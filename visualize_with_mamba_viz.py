#!/usr/bin/env python3
"""
Visualize Genomics Training with mamba_viz.py
==============================================
Converts training_metrics.csv to mamba_viz format and generates visualizations.

Usage:
    python visualize_with_mamba_viz.py ./checkpoints/rc_equivariant_20k/
"""

import sys
import pandas as pd
from pathlib import Path
from mamba_viz import plot_training_progress, setup_results_directory


def load_training_history(checkpoint_dir: str) -> dict:
    """
    Load training history from metrics CSV.

    Args:
        checkpoint_dir: Directory containing training_metrics.csv

    Returns:
        Dict with training history in mamba_viz format
    """
    checkpoint_path = Path(checkpoint_dir)

    # Try both possible filenames
    metrics_csv = checkpoint_path / "training_metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = checkpoint_path / "metrics.csv"

    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"No metrics file found in {checkpoint_dir}\n"
            f"Expected: training_metrics.csv or metrics.csv"
        )

    print(f"Loading metrics from: {metrics_csv}")
    df = pd.read_csv(metrics_csv)

    # Convert to mamba_viz format
    training_history = {
        'step': df['step'].tolist(),
        'train_loss': df['train_loss'].tolist(),
        'val_loss': df['val_loss'].tolist(),
        'train_acc': df['train_acc'].tolist() if 'train_acc' in df.columns else [0] * len(df),
        'val_acc': df['val_acc'].tolist() if 'val_acc' in df.columns else [0] * len(df),
    }

    print(f"✓ Loaded {len(df)} checkpoints")
    print(f"  Steps: {df['step'].min()} → {df['step'].max()}")

    return training_history


def visualize_training(checkpoint_dir: str, output_dir: str = None):
    """
    Generate training visualizations using mamba_viz.

    Args:
        checkpoint_dir: Directory containing training_metrics.csv
        output_dir: Output directory for plots (default: checkpoint_dir/../results/)
    """
    checkpoint_path = Path(checkpoint_dir)

    # Determine output directory
    if output_dir is None:
        # Use results directory parallel to checkpoints
        output_dir = checkpoint_path.parent.parent / "results" / checkpoint_path.name

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING TRAINING VISUALIZATIONS (mamba_viz)")
    print("="*70)
    print(f"Checkpoint Dir: {checkpoint_path}")
    print(f"Output Dir:     {output_path}")
    print("="*70 + "\n")

    # Load training history
    training_history = load_training_history(checkpoint_dir)

    # Generate visualization using mamba_viz
    print("Generating training progress plot...")
    plot_training_progress(
        training_history=training_history,
        results_dir=str(output_path),
        filename="training_progress.png"
    )

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"Output saved to: {output_path}")
    print(f"\nGenerated files:")
    print(f"  - training_progress.png   (Loss & Accuracy curves)")
    print("="*70 + "\n")

    # Print summary stats
    print_summary_stats(training_history)


def print_summary_stats(history: dict):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("TRAINING SUMMARY STATISTICS")
    print("="*70)

    steps = history['step']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']

    print(f"\nTotal Steps: {steps[-1]}")
    print(f"Checkpoints: {len(steps)}")

    print(f"\nInitial Metrics (Step {steps[0]}):")
    print(f"  Train Loss: {train_loss[0]:.6f}")
    print(f"  Val Loss:   {val_loss[0]:.6f}")
    print(f"  Train Acc:  {train_acc[0]:.4f}")
    print(f"  Val Acc:    {val_acc[0]:.4f}")

    print(f"\nFinal Metrics (Step {steps[-1]}):")
    print(f"  Train Loss: {train_loss[-1]:.6f}")
    print(f"  Val Loss:   {val_loss[-1]:.6f}")
    print(f"  Train Acc:  {train_acc[-1]:.4f}")
    print(f"  Val Acc:    {val_acc[-1]:.4f}")

    # Find best validation
    best_val_idx = val_loss.index(min(val_loss))
    best_val_step = steps[best_val_idx]
    best_val_loss = val_loss[best_val_idx]

    best_acc_idx = val_acc.index(max(val_acc))
    best_acc_step = steps[best_acc_idx]
    best_acc = val_acc[best_acc_idx]

    print(f"\nBest Validation:")
    print(f"  Best Val Loss: {best_val_loss:.6f} (Step {best_val_step})")
    print(f"  Best Val Acc:  {best_acc:.4f} (Step {best_acc_step})")

    print(f"\nImprovement:")
    loss_improvement = train_loss[0] - train_loss[-1]
    loss_pct = (loss_improvement / train_loss[0]) * 100
    print(f"  Train Loss: {train_loss[0]:.4f} → {train_loss[-1]:.4f} ({loss_pct:.1f}% reduction)")

    acc_improvement = val_acc[-1] - val_acc[0]
    acc_pct = (acc_improvement / (1 - val_acc[0])) * 100 if val_acc[0] < 1 else 0
    print(f"  Val Acc:    {val_acc[0]:.4f} → {val_acc[-1]:.4f} (+{acc_improvement:.4f})")

    print(f"\nGeneralization Gap (Final):")
    gap = abs(train_loss[-1] - val_loss[-1])
    print(f"  Loss Gap: {gap:.6f}")
    gap_acc = abs(train_acc[-1] - val_acc[-1])
    print(f"  Acc Gap:  {gap_acc:.4f}")

    print("="*70 + "\n")


def main():
    """Main CLI."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_with_mamba_viz.py <checkpoint_dir> [output_dir]")
        print("\nExamples:")
        print("  python visualize_with_mamba_viz.py ./checkpoints/rc_equivariant_20k/")
        print("  python visualize_with_mamba_viz.py ./checkpoints/stable_20k/ ./my_results/")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(checkpoint_dir).exists():
        print(f"Error: Directory not found: {checkpoint_dir}")
        sys.exit(1)

    visualize_training(checkpoint_dir, output_dir)


if __name__ == "__main__":
    main()
