"""
Checkpoint Utilities
==================
Save/load/resume training state with atomic writes and auto-recovery.
"""

import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import jax.numpy as jnp
import numpy as np

from mamba_optim import OptimizedTrainState


def save_checkpoint(
    state: OptimizedTrainState,
    step: int,
    checkpoint_dir: str,
    config: Any = None,
    metrics: Optional[Dict[str, float]] = None,
    keep_last_n: int = 3
):
    """
    Save checkpoint with atomic write (write to temp, then rename).

    Args:
        state: Training state (params + optimizer)
        step: Current training step
        checkpoint_dir: Directory to save checkpoint
        config: Configuration object (for reproducibility)
        metrics: Optional dict with train_loss, train_acc, val_loss, val_acc
        keep_last_n: Keep only last N checkpoints to save disk space
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Checkpoint filename
    checkpoint_name = f"checkpoint_{step:08d}.pkl"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    temp_path = checkpoint_path + ".tmp"

    # Prepare checkpoint data
    checkpoint_data = {
        'step': step,
        'params': state.params,
        'opt_state': state.opt_state,
        'config': config,
        'metrics': metrics or {},
    }

    # Write to temporary file first (atomic write)
    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        # Atomic rename
        shutil.move(temp_path, checkpoint_path)

        # Print checkpoint info with metrics
        if metrics:
            print(f"✓ Checkpoint saved: {checkpoint_path}")
            print(f"  Metrics: train_loss={metrics.get('train_loss', 'N/A'):.4f}, "
                  f"train_acc={metrics.get('train_acc', 'N/A'):.4f}, "
                  f"val_loss={metrics.get('val_loss', 'N/A'):.4f}, "
                  f"val_acc={metrics.get('val_acc', 'N/A'):.4f}")
        else:
            print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Clean up old checkpoints
        cleanup_old_checkpoints(checkpoint_dir, keep_last_n)

    except Exception as e:
        print(f"ERROR: Failed to save checkpoint: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        checkpoint_data: Dictionary with params, opt_state, step, config
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  Step: {checkpoint_data.get('step', 'unknown')}")

        return checkpoint_data

    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        raise


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the most recent checkpoint in directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        checkpoint_path: Path to latest checkpoint, or None if no checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Find all checkpoint files
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_') and f.endswith('.pkl')
    ]

    if not checkpoints:
        return None

    # Sort by step number (embedded in filename)
    checkpoints.sort()
    latest = checkpoints[-1]

    return os.path.join(checkpoint_dir, latest)


def auto_resume(
    checkpoint_dir: str,
    state: OptimizedTrainState,
    config: Any = None
) -> Tuple[OptimizedTrainState, int, Any]:
    """
    Automatically resume from latest checkpoint if available.

    Args:
        checkpoint_dir: Directory to check for checkpoints
        state: Default training state (used if no checkpoint found)
        config: Default configuration (used if no checkpoint found)

    Returns:
        state: Training state (restored or default)
        step: Training step (restored or 0)
        config: Configuration (restored or default)
    """
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint is None:
        print("No checkpoint found, starting from scratch")
        return state, 0, config

    print(f"Resuming from checkpoint: {latest_checkpoint}")

    # Load checkpoint
    checkpoint_data = load_checkpoint(latest_checkpoint)

    # Restore training state
    state = state.replace(
        params=checkpoint_data['params'],
        opt_state=checkpoint_data['opt_state']
    )

    step = checkpoint_data.get('step', 0)
    restored_config = checkpoint_data.get('config', config)

    print(f"✓ Resumed from step {step}")

    return state, step, restored_config


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 3):
    """
    Remove old checkpoints, keeping only the last N.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return

    # Find all checkpoint files
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_') and f.endswith('.pkl')
    ]

    if len(checkpoints) <= keep_last_n:
        return  # Nothing to clean up

    # Sort by modification time
    checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))

    # Remove oldest checkpoints
    to_remove = checkpoints[:-keep_last_n]
    for checkpoint in to_remove:
        path = os.path.join(checkpoint_dir, checkpoint)
        try:
            os.remove(path)
            print(f"  Removed old checkpoint: {checkpoint}")
        except Exception as e:
            print(f"  Warning: Could not remove {checkpoint}: {e}")


def save_phase_marker(checkpoint_dir: str, phase: str, step: int):
    """
    Save a marker file indicating training phase transition.

    Args:
        checkpoint_dir: Checkpoint directory
        phase: Phase name (e.g., "phase1", "phase2")
        step: Step at which phase started
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    marker_path = os.path.join(checkpoint_dir, f"{phase}_step_{step}.marker")

    with open(marker_path, 'w') as f:
        f.write(f"Phase: {phase}\nStep: {step}\n")

    print(f"✓ Phase marker saved: {phase} at step {step}")


def load_phase_markers(checkpoint_dir: str) -> Dict[str, int]:
    """
    Load all phase markers from checkpoint directory.

    Returns:
        phases: Dictionary mapping phase name to step
    """
    if not os.path.exists(checkpoint_dir):
        return {}

    phases = {}
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.marker'):
            # Parse filename: "phase1_step_10000.marker"
            parts = f.replace('.marker', '').split('_step_')
            if len(parts) == 2:
                phase_name = parts[0]
                step = int(parts[1])
                phases[phase_name] = step

    return phases


def save_metrics_to_csv(
    checkpoint_dir: str,
    step: int,
    metrics: Dict[str, float],
    create_header: bool = False
):
    """
    Save metrics to CSV file for easy tracking and plotting.

    Args:
        checkpoint_dir: Directory to save metrics CSV
        step: Current training step
        metrics: Dictionary with train_loss, train_acc, val_loss, val_acc
        create_header: Whether to create CSV header (first write)
    """
    import csv

    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_csv = os.path.join(checkpoint_dir, "training_metrics.csv")

    # Check if file exists to determine if we need header
    file_exists = os.path.exists(metrics_csv)

    try:
        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if file is new
            if not file_exists or create_header:
                writer.writerow(['step', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

            # Write metrics
            writer.writerow([
                step,
                metrics.get('train_loss', ''),
                metrics.get('train_acc', ''),
                metrics.get('val_loss', ''),
                metrics.get('val_acc', '')
            ])

    except Exception as e:
        print(f"Warning: Could not save metrics to CSV: {e}")
