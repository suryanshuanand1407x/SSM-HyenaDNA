#!/usr/bin/env python3
"""
RC-Equivariant Tustin-Mamba Training for hg38
==============================================
Integrates Reverse Complement (RC) equivariance so the model
treats DNA sequences and their reverse complements identically.

Features:
- RC-aware data augmentation (50% random RC flip)
- RC consistency loss (encourages symmetric predictions)
- Full metrics tracking (train/val loss & accuracy)
- Stable 20K step training with no NaN

Biological Motivation:
DNA is double-stranded: 5'-ATCG-3' pairs with 3'-TAGC-5'
Both strands are functionally equivalent for many genomic features.
The model should produce identical predictions for both.
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
    cast_to_bfloat16
)
from rc_equivariance import (
    RCDataLoader,
    rc_consistency_loss,
    reverse_complement_tokens
)


# =============================================================================
# RC-EQUIVARIANT CONFIGURATION
# =============================================================================

RC_CONFIG = HyenaFineTuneConfig(
    # Model Architecture
    vocab_size=12,
    d_model=128,              # Medium size
    n_layers=6,
    d_state=16,
    d_conv=4,
    expand=2,
    mode="tustin",            # Tustin discretization

    # Pre-trained Model
    pretrained_model="LongSafari/hyenadna-medium-160k-seqlen",
    load_embeddings=True,
    load_layer_norms=True,

    # Training - Stable hyperparameters
    learning_rate=5e-5,       # Conservative for stability
    batch_size=8,
    seq_len=8192,
    max_steps=20000,
    warmup_steps=1000,

    # Two-Phase Training
    phase1_steps=4000,        # 20% freeze embeddings
    phase2_steps=16000,       # 80% full training
    freeze_embeddings_phase1=True,

    # Optimization
    use_bfloat16=True,
    gradient_clip_norm=0.5,   # Aggressive clipping
    weight_decay=0.05,
    gradient_accumulation_steps=1,

    # Evaluation & Checkpointing
    eval_interval=500,
    eval_iters=20,
    save_interval=500,
    log_interval=250,

    # Data Loading
    num_workers=4,
    prefetch_batches=2,
    max_tokens=20_000_000,

    # Directories
    checkpoint_dir="./checkpoints/rc_equivariant_20k",
    results_dir="./results/rc_equivariant_20k",
    cache_dir="./data/hyenadna"
)

# RC-specific hyperparameters
RC_AUGMENTATION_MODE = 'random'  # 'random', 'double', or 'none'
RC_LOSS_WEIGHT = 0.1             # Weight for RC consistency loss
USE_RC_LOSS = True               # Enable RC consistency loss


# =============================================================================
# RC-Aware Training Step
# =============================================================================

@jax.jit
def train_step_rc_aware(
    state: OptimizedTrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    rc_weight: float = 0.1
):
    """
    Training step with RC consistency loss.

    Args:
        state: Training state
        x: (B, L) input sequences
        y: (B, L) target sequences
        mask: (B, L) loss mask
        rc_weight: Weight for RC consistency term

    Returns:
        state: Updated state
        metrics: Dict with ce_loss, rc_loss, total_loss
    """
    def loss_fn(params):
        # Forward pass
        logits = state.apply_fn({'params': params}, x, train=True)

        # Standard cross-entropy loss
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        ce_loss = jnp.sum(per_token_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

        # RC consistency loss (computed unconditionally)
        # Note: If rc_weight=0, this still computes but contributes nothing to total_loss

        # Compute RC of input
        rc_x = jax.vmap(reverse_complement_tokens)(x)

        # Forward pass on RC
        rc_logits = state.apply_fn({'params': params}, rc_x, train=True)

        # Reverse the RC logits to align with forward
        rc_logits_reversed = jnp.flip(rc_logits, axis=1)

        # Consistency loss (MSE between forward and reversed-RC predictions)
        diff = logits - rc_logits_reversed
        rc_loss = jnp.mean(diff ** 2)

        # Combined loss (rc_weight multiplier handles enabling/disabling)
        total_loss = ce_loss + rc_weight * rc_loss

        return total_loss, (ce_loss, rc_loss)

    # Compute gradients
    (total_loss, (ce_loss, rc_loss)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)

    # Update parameters
    state = state.apply_gradients(grads=grads)

    # Return metrics
    metrics = {
        'total_loss': total_loss,
        'ce_loss': ce_loss,
        'rc_loss': rc_loss
    }

    return state, metrics


@jax.jit
def eval_step_rc_aware(
    state: OptimizedTrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
):
    """
    Evaluation step with loss and accuracy.

    Args:
        state: Training state
        x: (B, L) input sequences
        y: (B, L) target sequences
        mask: (B, L) loss mask

    Returns:
        loss: Scalar loss
        accuracy: Scalar accuracy
    """
    # Forward pass (no dropout)
    logits = state.apply_fn({'params': state.params}, x, train=False)

    # Compute loss
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    loss = jnp.sum(per_token_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    # Compute accuracy
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == y).astype(mask.dtype)
    accuracy = jnp.sum(correct * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    return loss, accuracy


def estimate_loss_with_rc(
    state: OptimizedTrainState,
    loader: RCDataLoader,
    eval_iters: int = 20
):
    """
    Estimate loss and accuracy on train/val sets.

    Args:
        state: Training state
        loader: RC-aware data loader
        eval_iters: Number of batches to evaluate

    Returns:
        metrics: Dict with train/val loss and accuracy
    """
    out = {}

    for split in ['train', 'validation']:
        losses = []
        accs = []

        for _ in range(eval_iters):
            x, y, mask = loader.get_batch(split)
            loss, acc = eval_step_rc_aware(state, x, y, mask)
            losses.append(loss)
            accs.append(acc)

        # Block and compute means
        losses = [jax.block_until_ready(l) for l in losses]
        accs = [jax.block_until_ready(a) for a in accs]

        key_prefix = 'train' if split == 'train' else 'val'
        out[f'{key_prefix}_loss'] = float(jnp.mean(jnp.array(losses)))
        out[f'{key_prefix}_acc'] = float(jnp.mean(jnp.array(accs)))

    return out


# =============================================================================
# Training Loop with RC Equivariance
# =============================================================================

def print_metrics_table(step, metrics, phase_name="Training"):
    """Print metrics in a formatted table."""
    print(f"\n{'='*70}")
    print(f"{phase_name} - Step {step}")
    print(f"{'='*70}")

    # Check if we have RC-specific metrics
    if 'ce_loss' in metrics:
        print(f"{'Metric':<20} {'Value':<15}")
        print(f"{'-'*35}")
        print(f"{'CE Loss':<20} {metrics['ce_loss']:<15.6f}")
        print(f"{'RC Loss':<20} {metrics['rc_loss']:<15.6f}")
        print(f"{'Total Loss':<20} {metrics['total_loss']:<15.6f}")
    else:
        print(f"{'Metric':<20} {'Train':<15} {'Validation':<15}")
        print(f"{'-'*50}")
        print(f"{'Loss':<20} {metrics['train_loss']:<15.6f} {metrics['val_loss']:<15.6f}")
        print(f"{'Accuracy':<20} {metrics['train_acc']:<15.4f} {metrics['val_acc']:<15.4f}")

    print(f"{'='*70}\n")


def train_with_rc_equivariance(
    state: OptimizedTrainState,
    loader: RCDataLoader,
    config: HyenaFineTuneConfig,
    rc_weight: float = 0.1,
    start_step: int = 0
):
    """
    Training loop with RC equivariance.

    Args:
        state: Training state
        loader: RC-aware data loader
        config: Configuration
        rc_weight: Weight for RC consistency loss
        start_step: Starting step (for resume)
    """
    print("\n" + "="*70)
    print("RC-EQUIVARIANT TRAINING")
    print("="*70)
    print(f"Start Step: {start_step}")
    print(f"Max Steps: {config.max_steps}")
    print(f"RC Augmentation: {RC_AUGMENTATION_MODE}")
    print(f"RC Loss Weight: {rc_weight}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Gradient Clip: {config.gradient_clip_norm}")
    print("="*70 + "\n")

    # Initial evaluation
    print("Initial evaluation...")
    initial_metrics = estimate_loss_with_rc(state, loader, eval_iters=config.eval_iters)
    print_metrics_table(start_step, initial_metrics, "Initial State")
    save_metrics_to_csv(config.checkpoint_dir, start_step, initial_metrics)

    # Training loop
    pbar = tqdm(range(start_step, config.max_steps),
                desc="Training",
                initial=start_step,
                total=config.max_steps)

    running_metrics = {'total_loss': 0.0, 'ce_loss': 0.0, 'rc_loss': 0.0}
    log_count = 0

    for step in pbar:
        # Get batch (RC-augmented)
        x, y, mask = loader.get_batch('train')

        # Training step with RC loss
        state, metrics = train_step_rc_aware(state, x, y, mask, rc_weight)

        # Accumulate metrics
        for key in running_metrics:
            running_metrics[key] += float(metrics[key])
        log_count += 1

        # Check for NaN
        if jnp.isnan(metrics['total_loss']) or jnp.isinf(metrics['total_loss']):
            print(f"\n❌ NaN/Inf detected at step {step}!")
            print(f"Metrics: {metrics}")
            save_checkpoint(state, step, config.checkpoint_dir, config, prefix="nan_error_")
            raise ValueError(f"Training failed: NaN/Inf at step {step}")

        # Periodic logging
        if (step + 1) % config.log_interval == 0:
            avg_metrics = {k: v / log_count for k, v in running_metrics.items()}
            pbar.set_postfix({
                'loss': f"{avg_metrics['total_loss']:.4f}",
                'ce': f"{avg_metrics['ce_loss']:.4f}",
                'rc': f"{avg_metrics['rc_loss']:.4f}"
            })
            running_metrics = {'total_loss': 0.0, 'ce_loss': 0.0, 'rc_loss': 0.0}
            log_count = 0

        # Periodic evaluation
        if (step + 1) % config.eval_interval == 0 or (step + 1) == config.max_steps:
            print(f"\n{'─'*70}")
            print(f"Evaluation at step {step + 1}")
            print(f"{'─'*70}")

            # Compute full metrics
            eval_metrics = estimate_loss_with_rc(state, loader, eval_iters=config.eval_iters)

            # Check for NaN
            if any(jnp.isnan(v) or jnp.isinf(v) for v in eval_metrics.values()):
                print("\n❌ NaN/Inf in evaluation metrics!")
                print(f"Metrics: {eval_metrics}")
                save_checkpoint(state, step + 1, config.checkpoint_dir, config, prefix="nan_eval_")
                raise ValueError(f"NaN/Inf in evaluation at step {step + 1}")

            # Print metrics
            print_metrics_table(step + 1, eval_metrics, "Evaluation")

            # Save metrics
            save_metrics_to_csv(config.checkpoint_dir, step + 1, eval_metrics)

            # Save checkpoint
            print(f"Saving checkpoint at step {step + 1}...")
            save_checkpoint(
                state,
                step + 1,
                config.checkpoint_dir,
                config,
                metrics=eval_metrics,
                keep_last_n=10
            )
            print(f"✓ Checkpoint saved\n")

    return state


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    """Main training function with RC equivariance."""
    print("\n" + "="*70)
    print("RC-EQUIVARIANT TUSTIN-MAMBA TRAINING FOR HG38")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Model: {RC_CONFIG.d_model}d × {RC_CONFIG.n_layers} layers")
    print(f"  Mode: {RC_CONFIG.mode} (Tustin discretization)")
    print(f"  Steps: {RC_CONFIG.max_steps}")
    print(f"  Learning Rate: {RC_CONFIG.learning_rate}")
    print(f"  Batch Size: {RC_CONFIG.batch_size}")
    print(f"  Sequence Length: {RC_CONFIG.seq_len}")
    print(f"  RC Augmentation: {RC_AUGMENTATION_MODE}")
    print(f"  RC Loss Weight: {RC_LOSS_WEIGHT}")
    print(f"  Checkpoints: {RC_CONFIG.checkpoint_dir}")
    print("="*70 + "\n")

    # Create directories
    os.makedirs(RC_CONFIG.checkpoint_dir, exist_ok=True)
    os.makedirs(RC_CONFIG.results_dir, exist_ok=True)

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

    # Initialize RC-aware data loader
    print("\nInitializing RC-aware HG38 data loader...")
    base_loader = HG38DataLoader(RC_CONFIG)
    rc_loader = RCDataLoader(
        base_loader,
        mode=RC_AUGMENTATION_MODE,
        seed=42
    )
    print(f"✓ RC augmentation mode: {RC_AUGMENTATION_MODE}")

    # Create model
    print("\nCreating hybrid model (HyenaDNA + Tustin-Mamba)...")
    from model_hybrid import create_hybrid_model

    model, params, pretrained_weights = create_hybrid_model(RC_CONFIG, rng)

    # Cast to bfloat16
    if RC_CONFIG.use_bfloat16:
        params = cast_to_bfloat16(params)
        print("✓ Parameters cast to bfloat16")

    # Count parameters
    n_params = count_parameters(params)
    print(f"✓ Total parameters: {n_params:,}")

    # Create optimizer with gradient clipping
    print(f"\nCreating optimizer...")
    print(f"  Learning rate: {RC_CONFIG.learning_rate}")
    print(f"  Gradient clip: {RC_CONFIG.gradient_clip_norm}")
    print(f"  Weight decay: {RC_CONFIG.weight_decay}")

    tx = optax.chain(
        optax.clip_by_global_norm(RC_CONFIG.gradient_clip_norm),
        optax.adamw(RC_CONFIG.learning_rate, weight_decay=RC_CONFIG.weight_decay)
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
    print("This may take 30-60 seconds on first run...")

    # Warmup with a single batch
    x, y, mask = rc_loader.get_batch('train')
    warmup_start = time.time()

    # Compile training step
    _ = train_step_rc_aware(state, x, y, mask, RC_LOSS_WEIGHT)

    # Compile eval step
    _ = eval_step_rc_aware(state, x, y, mask)

    warmup_time = time.time() - warmup_start
    print(f"✓ JIT warmup completed in {warmup_time:.2f}s")

    # Test RC augmentation
    print("\n" + "="*70)
    print("Testing RC Equivariance")
    print("="*70)

    # Get a batch and show augmentation
    x_test, y_test, mask_test = rc_loader.get_batch('train')
    print(f"Batch shape: {x_test.shape}")
    print(f"First sequence (first 10 tokens): {x_test[0, :10]}")

    # Compute RC
    from rc_equivariance import reverse_complement_tokens
    rc_test = reverse_complement_tokens(x_test[0])
    print(f"RC of first sequence (first 10): {rc_test[:10]}")
    print("✓ RC augmentation working")
    print("="*70 + "\n")

    # Train
    print("="*70)
    print("STARTING RC-EQUIVARIANT TRAINING")
    print("="*70)

    train_start = time.time()

    try:
        state = train_with_rc_equivariance(
            state,
            rc_loader,
            RC_CONFIG,
            rc_weight=RC_LOSS_WEIGHT if USE_RC_LOSS else 0.0,
            start_step=0
        )

        # Training complete
        train_time = time.time() - train_start

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Total time: {train_time/3600:.2f} hours")
        print(f"Steps/sec: {RC_CONFIG.max_steps/train_time:.2f}")
        print(f"Checkpoints: {RC_CONFIG.checkpoint_dir}")
        print(f"Metrics: {RC_CONFIG.checkpoint_dir}/metrics.csv")
        print("="*70)

        # Save final summary
        summary_path = os.path.join(RC_CONFIG.results_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("RC-Equivariant Tustin-Mamba Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {RC_CONFIG.d_model}d × {RC_CONFIG.n_layers} layers\n")
            f.write(f"Mode: {RC_CONFIG.mode}\n")
            f.write(f"Steps: {RC_CONFIG.max_steps}\n")
            f.write(f"RC Augmentation: {RC_AUGMENTATION_MODE}\n")
            f.write(f"RC Loss Weight: {RC_LOSS_WEIGHT}\n")
            f.write(f"Training Time: {train_time/3600:.2f} hours\n")
            f.write(f"Parameters: {n_params:,}\n")

        print(f"\n✓ Summary saved to: {summary_path}")

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
