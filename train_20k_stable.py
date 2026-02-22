#!/usr/bin/env python3
"""
Stable 20K Step Training with Full Metrics
==========================================
- Conservative hyperparameters to prevent NaN
- Full metrics logging (train/val loss & accuracy)
- Gradient monitoring and clipping
- Checkpoint every 500 steps
- Detailed progress tracking
"""

import os
import time
from pathlib import Path
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax

from config_hyena import HyenaFineTuneConfig
from hyena_data_hg38 import HG38DataLoader
from model_hybrid import create_hybrid_model, count_parameters
from checkpoint_utils import save_checkpoint, save_metrics_to_csv
from mamba_optim import (
    OptimizedTrainState,
    train_step_fused,
    estimate_loss_async,
    warmup_jit_compilation,
    cast_to_bfloat16
)


# =============================================================================
# STABLE CONFIGURATION FOR 20K STEPS
# =============================================================================

STABLE_CONFIG = HyenaFineTuneConfig(
    # Model Architecture
    vocab_size=12,
    d_model=256,              # Medium size for stability
    n_layers=4,               # 4 layers
    d_state=16,
    d_conv=4,
    expand=2,
    mode="tustin",

    # Pre-trained Model
    pretrained_model="LongSafari/hyenadna-medium-160k-seqlen",
    load_embeddings=True,
    load_layer_norms=True,

    # Training - CONSERVATIVE for stability
    learning_rate=5e-5,       # Lower LR to prevent NaN
    batch_size=16,            # Smaller batch for stability
    seq_len=1024,             # Shorter sequences
    max_steps=20000,          # 20K steps total
    warmup_steps=1000,        # Gradual warmup

    # Two-Phase Training
    phase1_steps=4000,        # 20% for embedding freeze
    phase2_steps=16000,       # 80% for full training
    freeze_embeddings_phase1=True,

    # Optimization - STABLE settings
    use_bfloat16=True,
    gradient_clip_norm=0.5,   # Aggressive clipping to prevent NaN
    weight_decay=0.05,        # Lower weight decay
    gradient_accumulation_steps=1,

    # Evaluation & Checkpointing
    eval_interval=500,        # Evaluate every 500 steps
    eval_iters=20,            # 20 batches for evaluation
    save_interval=500,        # Checkpoint every 500 steps
    log_interval=50,          # Log every 50 steps

    # Data Loading
    num_workers=4,
    prefetch_batches=2,
    max_tokens=20_000_000,    # 20M tokens for 20K steps

    # Directories
    checkpoint_dir="./checkpoints/stable_20k",
    results_dir="./results/stable_20k",
    cache_dir="./data/hyenadna"
)


def check_for_nan(params, step):
    """Check if any parameters are NaN and raise error."""
    def has_nan(x):
        if isinstance(x, jnp.ndarray):
            return jnp.any(jnp.isnan(x))
        return False

    nan_leaves = [has_nan(x) for x in jax.tree_util.tree_leaves(params)]

    if any(nan_leaves):
        raise ValueError(f"NaN detected in parameters at step {step}!")


def print_metrics_table(step, metrics, phase_name="Training"):
    """Print metrics in a nice table format."""
    print(f"\n{'='*70}")
    print(f"{phase_name} - Step {step}")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Train':<15} {'Validation':<15}")
    print(f"{'-'*70}")
    print(f"{'Loss':<20} {metrics['train_loss']:<15.6f} {metrics['val_loss']:<15.6f}")
    print(f"{'Accuracy':<20} {metrics['train_acc']:<15.4f} {metrics['val_acc']:<15.4f}")
    print(f"{'='*70}\n")


def train_with_metrics(
    state: OptimizedTrainState,
    loader: HG38DataLoader,
    config: HyenaFineTuneConfig,
    start_step: int = 0
):
    """
    Training loop with full metrics tracking and NaN detection.

    Args:
        state: Training state
        loader: Data loader
        config: Configuration
        start_step: Starting step (for resume)
    """
    print("\n" + "="*70)
    print("STABLE 20K STEP TRAINING")
    print("="*70)
    print(f"Start Step: {start_step}")
    print(f"Max Steps: {config.max_steps}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Sequence Length: {config.seq_len}")
    print(f"Gradient Clip: {config.gradient_clip_norm}")
    print("="*70 + "\n")

    # Initial evaluation
    print("Initial evaluation...")
    initial_metrics = estimate_loss_async(state, loader, eval_iters=config.eval_iters)
    print_metrics_table(start_step, initial_metrics, "Initial State")

    # Save initial metrics
    save_metrics_to_csv(config.checkpoint_dir, start_step, initial_metrics)

    # Training loop
    pbar = tqdm(range(start_step, config.max_steps),
                desc="Training",
                initial=start_step,
                total=config.max_steps)

    running_loss = 0.0
    log_count = 0

    for step in pbar:
        # Get batch
        x, y, mask = loader.get_batch('train')

        # Training step
        state, loss = train_step_fused(state, x, y, mask)

        # Check for NaN
        loss_value = float(loss)
        if jnp.isnan(loss_value) or jnp.isinf(loss_value):
            print(f"\n❌ NaN/Inf detected at step {step}!")
            print(f"Loss value: {loss_value}")

            # Save emergency checkpoint
            print("Saving emergency checkpoint...")
            save_checkpoint(state, step, config.checkpoint_dir, config, prefix="nan_error_")

            raise ValueError(f"Training failed: NaN/Inf loss at step {step}")

        # Accumulate loss for logging
        running_loss += loss_value
        log_count += 1

        # Periodic logging
        if (step + 1) % config.log_interval == 0:
            avg_loss = running_loss / log_count
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'step': f'{step+1}/{config.max_steps}'
            })
            running_loss = 0.0
            log_count = 0

        # Periodic evaluation
        if (step + 1) % config.eval_interval == 0 or (step + 1) == config.max_steps:
            print(f"\n{'─'*70}")
            print(f"Evaluation at step {step + 1}")
            print(f"{'─'*70}")

            # Compute full metrics
            metrics = estimate_loss_async(state, loader, eval_iters=config.eval_iters)

            # Check for NaN in metrics
            if any(jnp.isnan(v) or jnp.isinf(v) for v in metrics.values()):
                print("\n❌ NaN/Inf detected in evaluation metrics!")
                print(f"Metrics: {metrics}")
                save_checkpoint(state, step + 1, config.checkpoint_dir, config, prefix="nan_eval_")
                raise ValueError(f"NaN/Inf in evaluation metrics at step {step + 1}")

            # Print metrics table
            print_metrics_table(step + 1, metrics, "Evaluation")

            # Save metrics
            save_metrics_to_csv(config.checkpoint_dir, step + 1, metrics)

            # Save checkpoint
            print(f"Saving checkpoint at step {step + 1}...")
            save_checkpoint(
                state,
                step + 1,
                config.checkpoint_dir,
                config,
                metrics=metrics,
                keep_last_n=10  # Keep last 10 checkpoints
            )
            print(f"✓ Checkpoint saved\n")

    return state


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("HYENADNA → TUSTIN MAMBA - STABLE 20K STEP TRAINING")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Model: {STABLE_CONFIG.d_model}d × {STABLE_CONFIG.n_layers} layers")
    print(f"  Mode: {STABLE_CONFIG.mode}")
    print(f"  Steps: {STABLE_CONFIG.max_steps}")
    print(f"  Learning Rate: {STABLE_CONFIG.learning_rate}")
    print(f"  Batch Size: {STABLE_CONFIG.batch_size}")
    print(f"  Sequence Length: {STABLE_CONFIG.seq_len}")
    print(f"  Gradient Clip: {STABLE_CONFIG.gradient_clip_norm}")
    print(f"  Checkpoints: {STABLE_CONFIG.checkpoint_dir}")
    print("="*70 + "\n")

    # Create directories
    os.makedirs(STABLE_CONFIG.checkpoint_dir, exist_ok=True)
    os.makedirs(STABLE_CONFIG.results_dir, exist_ok=True)

    # Initialize JAX
    print("Initializing JAX...")
    try:
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        if devices[0].platform == 'gpu':
            print(f"✓ Running on GPU: {devices[0].device_kind}")
        else:
            print("✓ Running on CPU")
    except Exception as e:
        print(f"Device detection: {e}")

    rng = jax.random.PRNGKey(42)

    # Initialize data loader
    print("\nInitializing HG38 data loader...")
    loader = HG38DataLoader(STABLE_CONFIG)

    # Create model
    print("\nCreating hybrid model...")
    from model_hybrid import create_hybrid_model

    model, params, pretrained_weights = create_hybrid_model(STABLE_CONFIG, rng)

    # Cast to bfloat16
    if STABLE_CONFIG.use_bfloat16:
        params = cast_to_bfloat16(params)
        print("✓ Parameters cast to bfloat16")

    # Count parameters
    n_params = count_parameters(params)
    print(f"✓ Total parameters: {n_params:,}")

    # Create optimizer with gradient clipping
    print(f"\nCreating optimizer (LR={STABLE_CONFIG.learning_rate}, clip={STABLE_CONFIG.gradient_clip_norm})...")

    tx = optax.chain(
        optax.clip_by_global_norm(STABLE_CONFIG.gradient_clip_norm),  # Gradient clipping
        optax.adamw(
            STABLE_CONFIG.learning_rate,
            weight_decay=STABLE_CONFIG.weight_decay
        )
    )

    opt_state = tx.init(params)

    # Create training state
    state = OptimizedTrainState(
        step=0,
        params=params,
        opt_state=opt_state,
        tx=tx,
        apply_fn=model.apply
    )

    print("✓ Training state created")

    # JIT warmup
    print("\nWarming up JIT compilation...")
    warmup_start = time.time()
    state = warmup_jit_compilation(state, loader, verbose=False)
    warmup_time = time.time() - warmup_start
    print(f"✓ JIT warmup completed in {warmup_time:.2f}s")

    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    train_start = time.time()

    try:
        state = train_with_metrics(state, loader, STABLE_CONFIG, start_step=0)

        # Training complete
        train_time = time.time() - train_start

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Total time: {train_time/3600:.2f} hours")
        print(f"Steps/sec: {STABLE_CONFIG.max_steps/train_time:.2f}")
        print(f"Checkpoints saved to: {STABLE_CONFIG.checkpoint_dir}")
        print(f"Metrics saved to: {STABLE_CONFIG.checkpoint_dir}/metrics.csv")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Checkpoints saved - you can resume training later")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
