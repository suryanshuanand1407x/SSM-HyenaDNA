"""
HyenaDNA → Mamba Fine-tuning Script
==================================
Two-phase fine-tuning: Phase 1 (freeze embeddings) → Phase 2 (full fine-tuning)
Comparison study: Tustin vs ZOH Mamba blocks
"""

# ==============================================================================
# HYPERPARAMETERS - EDIT HERE FOR QUICK TWEAKS
# ==============================================================================

# Model Architecture
MODEL_D_MODEL = 512              # Hidden dimension (128, 256, 512, 1024)
MODEL_N_LAYERS = 6               # Number of Mamba blocks (2, 4, 6, 8, 12)
MODEL_D_STATE = 16               # SSM state dimension (16, 32, 64)
MODEL_D_CONV = 4                 # Convolution width (4, 8)
MODEL_EXPAND = 2                 # Expansion factor (2, 4)
MODEL_MODE = "tustin"            # Discretization: "tustin" or "zoh"

# Training Strategy
LEARNING_RATE = 1e-4             # Learning rate (1e-5 to 1e-3)
BATCH_SIZE = 32                  # Batch size (8, 16, 32, 64)
SEQ_LEN = 4096                   # Sequence length (1024, 2048, 4096, 8192)
MAX_STEPS = 20000               # Total training steps
WARMUP_STEPS = 2000              # LR warmup steps

# Two-Phase Training
PHASE1_STEPS = 20000             # Phase 1 (freeze embeddings) - 20% of training
PHASE2_STEPS = 80000             # Phase 2 (full fine-tuning) - 80% of training
FREEZE_EMBEDDINGS_PHASE1 = True  # Whether to freeze embeddings in Phase 1

# Optimization
USE_BFLOAT16 = True              # Use bfloat16 (Tensor Cores) - recommended for RTX 5090
GRADIENT_CLIP_NORM = 1.0         # Gradient clipping (0.5 to 2.0)
WEIGHT_DECAY = 0.1               # AdamW weight decay (0.0 to 0.2)
GRADIENT_ACCUMULATION = 1        # Gradient accumulation steps (1, 2, 4)

# Evaluation & Checkpointing
EVAL_INTERVAL = 200              # Steps between evaluations (same as save interval for metrics)
EVAL_ITERS = 20                  # Number of batches for evaluation
SAVE_INTERVAL = 200              # Steps between checkpoints (SAVE EVERY 200 STEPS)
LOG_INTERVAL = 100               # Steps between log updates

# Data Loading (Multi-core)
NUM_WORKERS = 4                  # CPU workers for data loading (2, 4, 6, 8)
PREFETCH_BATCHES = 2             # Batches to prefetch (1, 2, 4)
ENABLE_PREFETCH = True           # Enable multi-core prefetching

# Pre-trained Model
PRETRAINED_MODEL = "LongSafari/hyenadna-medium-160k-seqlen"
LOAD_EMBEDDINGS = True           # Load pre-trained embeddings
LOAD_LAYER_NORMS = True          # Load pre-trained layer norms

# Dataset
MAX_TOKENS = 100_000_000         # Maximum training tokens (for medium-scale study)

# Directories (auto-created)
CHECKPOINT_DIR = "./checkpoints/hyena_mamba_custom"
RESULTS_DIR = "./results/custom_training"

# ==============================================================================
# END HYPERPARAMETERS
# ==============================================================================

import os
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax

from config_hyena import (
    HyenaFineTuneConfig,
    TUSTIN_CONFIG,
    ZOH_CONFIG,
    QUICK_CONFIG
)
from hyena_data_hg38 import HG38DataLoader
from model_hybrid import create_hybrid_model, freeze_embeddings, count_parameters
from checkpoint_utils import (
    save_checkpoint,
    auto_resume,
    save_phase_marker,
    load_phase_markers
)
from mamba_optim import (
    OptimizedTrainState,
    train_step_fused,
    eval_step_with_logits,
    estimate_loss_async,
    warmup_jit_compilation,
    cast_to_bfloat16
)
from mamba_metrics import compute_all_metrics
from mamba_viz import generate_performance_report


def create_config_from_hyperparameters() -> HyenaFineTuneConfig:
    """
    Create configuration from hyperparameters defined at top of file.

    This allows easy tweaking without editing config_hyena.py.
    """
    return HyenaFineTuneConfig(
        # Model Architecture
        vocab_size=12,  # HyenaDNA standard
        d_model=MODEL_D_MODEL,
        n_layers=MODEL_N_LAYERS,
        d_state=MODEL_D_STATE,
        d_conv=MODEL_D_CONV,
        expand=MODEL_EXPAND,
        mode=MODEL_MODE,

        # Pre-trained Model
        pretrained_model=PRETRAINED_MODEL,
        load_embeddings=LOAD_EMBEDDINGS,
        load_layer_norms=LOAD_LAYER_NORMS,

        # Training Strategy
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,

        # Two-Phase Training
        phase1_steps=PHASE1_STEPS,
        phase2_steps=PHASE2_STEPS,
        freeze_embeddings_phase1=FREEZE_EMBEDDINGS_PHASE1,

        # Optimization
        use_bfloat16=USE_BFLOAT16,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        # Evaluation & Checkpointing
        eval_interval=EVAL_INTERVAL,
        eval_iters=EVAL_ITERS,
        save_interval=SAVE_INTERVAL,
        log_interval=LOG_INTERVAL,

        # Data Loading
        num_workers=NUM_WORKERS,
        prefetch_batches=PREFETCH_BATCHES,
        max_tokens=MAX_TOKENS,

        # Directories
        checkpoint_dir=CHECKPOINT_DIR,
        results_dir=RESULTS_DIR,
    )


def create_masked_optimizer(learning_rate: float, freeze_mask: dict):
    """
    Create optimizer with frozen parameters.

    Args:
        learning_rate: Learning rate
        freeze_mask: Dictionary indicating which params to freeze

    Returns:
        optimizer: Optax optimizer with masking
    """
    # AdamW optimizer
    optimizer = optax.adamw(learning_rate, weight_decay=0.1)

    # Apply mask: frozen params get zero gradients
    mask_fn = optax.masked(optimizer, freeze_mask)

    return mask_fn


def create_train_state_with_pretrained(
    rng: jax.random.PRNGKey,
    config: HyenaFineTuneConfig,
    freeze_phase1: bool = True
) -> tuple:
    """
    Create training state with HyenaDNA pre-trained weights.

    Args:
        rng: Random key
        config: Configuration
        freeze_phase1: Whether to freeze embeddings (Phase 1)

    Returns:
        state: Training state
        model: Model instance
        pretrained_weights: Original HyenaDNA weights
    """
    print("\n" + "=" * 60)
    print("Creating Hybrid Model (HyenaDNA + Mamba)")
    print("=" * 60)

    # Create hybrid model with HyenaDNA weights
    model, params, pretrained_weights = create_hybrid_model(config, rng)

    # Cast to bfloat16 for Tensor Cores
    if config.use_bfloat16:
        params = cast_to_bfloat16(params)
        print("✓ Parameters cast to bfloat16")

    # Count parameters
    n_params = count_parameters(params)
    print(f"✓ Total parameters: {n_params:,}")

    # Create optimizer (with or without freezing)
    if freeze_phase1:
        freeze_mask = freeze_embeddings(params)
        tx = create_masked_optimizer(config.learning_rate, freeze_mask)
        print("✓ Optimizer created with FROZEN EMBEDDINGS (Phase 1)")
    else:
        tx = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
        print("✓ Optimizer created with ALL PARAMETERS TRAINABLE (Phase 2)")

    # Initialize optimizer state
    opt_state = tx.init(params)

    # Create training state
    state = OptimizedTrainState(
        step=0,
        params=params,
        opt_state=opt_state,
        tx=tx,
        apply_fn=model.apply
    )

    return state, model, pretrained_weights


def train_phase(
    state: OptimizedTrainState,
    loader: HG38DataLoader,
    config: HyenaFineTuneConfig,
    phase_name: str,
    start_step: int,
    end_step: int,
    checkpoint_dir: str
) -> OptimizedTrainState:
    """
    Train for one phase (Phase 1 or Phase 2).

    Args:
        state: Training state
        loader: Data loader
        config: Configuration
        phase_name: "Phase 1" or "Phase 2"
        start_step: Starting step
        end_step: Ending step
        checkpoint_dir: Where to save checkpoints

    Returns:
        state: Updated training state
    """
    print(f"\n{'=' * 60}")
    print(f"{phase_name}: Steps {start_step} → {end_step}")
    print(f"{'=' * 60}\n")

    # Save phase marker
    save_phase_marker(checkpoint_dir, phase_name.lower().replace(' ', '_'), start_step)

    # Training loop with progress bar
    pbar = tqdm(range(start_step, end_step), desc=phase_name)

    for step in pbar:
        # Get batch
        x, y, mask = loader.get_batch('train')

        # Training step (fused forward + backward)
        state, loss = train_step_fused(state, x, y, mask)

        # Update progress bar
        if step % config.log_interval == 0:
            # Block to get actual loss value
            loss_value = float(loss)
            pbar.set_postfix({'loss': f'{loss_value:.4f}'})

        # Periodic evaluation
        if (step + 1) % config.eval_interval == 0:
            print(f"\n--- Evaluation at step {step + 1} ---")
            metrics = estimate_loss_async(
                state,
                loader,
                eval_iters=config.eval_iters
            )

            train_loss = float(metrics['train_loss'])
            val_loss = float(metrics['val_loss'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print()

        # Periodic checkpointing (with metrics)
        if (step + 1) % config.save_interval == 0:
            # Compute metrics for checkpoint
            print(f"\n--- Computing metrics for checkpoint at step {step + 1} ---")
            checkpoint_metrics = estimate_loss_async(
                state,
                loader,
                eval_iters=config.eval_iters
            )

            # Convert to regular Python floats
            metrics_dict = {
                'train_loss': float(checkpoint_metrics['train_loss']),
                'train_acc': float(checkpoint_metrics.get('train_acc', 0.0)),
                'val_loss': float(checkpoint_metrics['val_loss']),
                'val_acc': float(checkpoint_metrics.get('val_acc', 0.0))
            }

            # Save checkpoint with metrics (keep last 20 checkpoints = 4000 steps of history)
            save_checkpoint(
                state,
                step + 1,
                checkpoint_dir,
                config=config,
                metrics=metrics_dict,
                keep_last_n=20  # Keep more checkpoints since we save more frequently
            )

            # Also save metrics to CSV for easy tracking
            from checkpoint_utils import save_metrics_to_csv
            save_metrics_to_csv(checkpoint_dir, step + 1, metrics_dict)

    return state


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Fine-tune HyenaDNA with Mamba blocks')
    parser.add_argument(
        '--config',
        type=str,
        default='quick',
        choices=['quick', 'tustin', 'zoh', 'custom'],
        help='Configuration preset (use "custom" to use hyperparameters from top of file)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest checkpoint'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Override checkpoint directory'
    )
    args = parser.parse_args()

    # Load configuration
    if args.config == 'custom':
        print("\n" + "=" * 60)
        print("Using CUSTOM hyperparameters from top of train_hyena.py")
        print("=" * 60)
        config = create_config_from_hyperparameters()
    elif args.config == 'quick':
        config = QUICK_CONFIG
    elif args.config == 'tustin':
        config = TUSTIN_CONFIG
    else:
        config = ZOH_CONFIG

    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir

    print("\n" + "=" * 60)
    print(f"HyenaDNA → Mamba Fine-tuning ({args.config.upper()})")
    print("=" * 60)
    print(f"Mode: {config.mode}")
    print(f"Model: {config.d_model}d × {config.n_layers} layers")
    print(f"Sequence Length: {config.seq_len}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Total Steps: {config.max_steps}")
    print(f"Phase 1 Steps: {config.phase1_steps} (freeze embeddings: {config.freeze_embeddings_phase1})")
    print(f"Phase 2 Steps: {config.phase2_steps} (full fine-tuning)")
    print(f"Data Workers: {config.num_workers} (prefetch: {config.prefetch_batches} batches)")
    print(f"Checkpoint Dir: {config.checkpoint_dir}")
    print()

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # Initialize JAX with GPU/CPU auto-detection
    print("Initializing JAX...")
    try:
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        if devices[0].platform == 'gpu':
            print(f"✓ Running on GPU: {devices[0].device_kind}")
        else:
            print("✓ Running on CPU (GPU not available or cuDNN issue)")
    except Exception as e:
        print(f"Device detection warning: {e}")
        print("Continuing with default device...")

    rng = jax.random.PRNGKey(42)

    # Initialize data loader (HG38 real genomic data)
    print("\nInitializing HG38 data loader (real genomic data)...")
    loader = HG38DataLoader(config)

    # Check for existing checkpoints
    current_step = 0
    phase_markers = load_phase_markers(config.checkpoint_dir)

    if args.resume:
        print("\nAttempting to resume from checkpoint...")
        # We'll handle resume after creating the state

    # Determine which phase we're in
    if current_step < config.phase1_steps:
        current_phase = 1
        freeze_embeddings_flag = True
    else:
        current_phase = 2
        freeze_embeddings_flag = False

    # Create training state
    print(f"\nCreating training state for Phase {current_phase}...")
    state, model, pretrained_weights = create_train_state_with_pretrained(
        rng,
        config,
        freeze_phase1=freeze_embeddings_flag
    )

    # Resume from checkpoint if requested
    if args.resume:
        state, current_step, _ = auto_resume(
            config.checkpoint_dir,
            state,
            config
        )

        # Re-determine phase based on loaded step
        if current_step < config.phase1_steps:
            current_phase = 1
        else:
            current_phase = 2

    # JIT warmup
    print("\nWarming up JIT compilation...")
    warmup_start = time.time()
    state = warmup_jit_compilation(state, loader)
    warmup_time = time.time() - warmup_start
    print(f"✓ JIT warmup completed in {warmup_time:.2f}s")

    # Training Phase 1: Freeze embeddings
    if current_step < config.phase1_steps:
        state = train_phase(
            state,
            loader,
            config,
            "Phase 1 (Freeze Embeddings)",
            current_step,
            config.phase1_steps,
            config.checkpoint_dir
        )
        current_step = config.phase1_steps

        # Save checkpoint at phase transition
        save_checkpoint(state, current_step, config.checkpoint_dir, config)

        print("\n" + "=" * 60)
        print("Phase 1 Complete → Transitioning to Phase 2")
        print("=" * 60)
        print("Re-creating optimizer with UNFROZEN parameters...")

        # Re-create optimizer without freezing
        tx = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
        opt_state = tx.init(state.params)
        state = state.replace(tx=tx, opt_state=opt_state)
        print("✓ Optimizer re-created with all parameters trainable\n")

    # Training Phase 2: Full fine-tuning
    state = train_phase(
        state,
        loader,
        config,
        "Phase 2 (Full Fine-tuning)",
        current_step,
        config.max_steps,
        config.checkpoint_dir
    )

    # Final checkpoint
    save_checkpoint(state, config.max_steps, config.checkpoint_dir, config)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    final_metrics = estimate_loss_async(
        state,
        loader,
        eval_iters=50  # More iterations for accurate final metrics
    )

    print("\nFinal Metrics:")
    print(f"  Train Loss: {float(final_metrics['train_loss']):.4f}")
    print(f"  Val Loss:   {float(final_metrics['val_loss']):.4f}")

    # Generate performance report
    print("\nGenerating performance report...")
    # Note: generate_performance_report expects training history
    # For now, just save final metrics
    report_path = os.path.join(config.results_dir, 'final_metrics.txt')
    with open(report_path, 'w') as f:
        f.write(f"Configuration: {args.config}\n")
        f.write(f"Mode: {config.mode}\n")
        f.write(f"Steps: {config.max_steps}\n")
        f.write(f"Final Train Loss: {float(final_metrics['train_loss']):.4f}\n")
        f.write(f"Final Val Loss: {float(final_metrics['val_loss']):.4f}\n")

    print(f"✓ Report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
