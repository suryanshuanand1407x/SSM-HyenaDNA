"""
HyenaDNA → Mamba Fine-tuning Script
==================================
Two-phase fine-tuning: Phase 1 (freeze embeddings) → Phase 2 (full fine-tuning)
Comparison study: Tustin vs ZOH Mamba blocks
"""

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
from hyena_data import HyenaDNALoader
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
        tx=tx
    )

    return state, model, pretrained_weights


def train_phase(
    state: OptimizedTrainState,
    loader: HyenaDNALoader,
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

        # Periodic checkpointing
        if (step + 1) % config.save_interval == 0:
            save_checkpoint(
                state,
                step + 1,
                checkpoint_dir,
                config=config
            )

    return state


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Fine-tune HyenaDNA with Mamba blocks')
    parser.add_argument(
        '--config',
        type=str,
        default='quick',
        choices=['quick', 'tustin', 'zoh'],
        help='Configuration preset'
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
    if args.config == 'quick':
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
    print(f"Total Steps: {config.max_steps}")
    print(f"Phase 1 Steps: {config.phase1_steps} (freeze embeddings)")
    print(f"Phase 2 Steps: {config.phase2_steps} (full fine-tuning)")
    print(f"Checkpoint Dir: {config.checkpoint_dir}")
    print()

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # Initialize JAX
    print("Initializing JAX...")
    print(f"JAX devices: {jax.devices()}")
    rng = jax.random.PRNGKey(42)

    # Initialize data loader
    print("\nInitializing data loader...")
    loader = HyenaDNALoader(config)

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
