"""
Plot Training Metrics from CSV
================================
Visualize training and validation loss/accuracy curves from training_metrics.csv
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_metrics(checkpoint_dir: str, save_path: str = None):
    """
    Plot training metrics from CSV file.

    Args:
        checkpoint_dir: Directory containing training_metrics.csv
        save_path: Optional path to save plot (if None, display interactively)
    """
    # Load metrics CSV
    metrics_csv = os.path.join(checkpoint_dir, "training_metrics.csv")

    if not os.path.exists(metrics_csv):
        print(f"ERROR: Metrics CSV not found: {metrics_csv}")
        print("Make sure you've run training with the updated checkpoint code.")
        sys.exit(1)

    # Read CSV
    df = pd.read_csv(metrics_csv)
    print(f"Loaded {len(df)} checkpoint records from {metrics_csv}")

    # Create figure with 2 subplots (loss and accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss curves
    ax1.plot(df['step'], df['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(df['step'], df['val_loss'], label='Val Loss', color='red', linewidth=2)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    ax2.plot(df['step'], df['train_acc'] * 100, label='Train Acc', color='blue', linewidth=2)
    ax2.plot(df['step'], df['val_acc'] * 100, label='Val Acc', color='red', linewidth=2)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total Steps: {df['step'].max()}")
    print(f"\nFinal Metrics (last checkpoint):")
    print(f"  Train Loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Train Acc:  {df['train_acc'].iloc[-1]*100:.2f}%")
    print(f"  Val Loss:   {df['val_loss'].iloc[-1]:.4f}")
    print(f"  Val Acc:    {df['val_acc'].iloc[-1]*100:.2f}%")
    print(f"\nBest Metrics:")
    best_val_loss_idx = df['val_loss'].idxmin()
    print(f"  Best Val Loss: {df['val_loss'].min():.4f} (step {df['step'].iloc[best_val_loss_idx]})")
    best_val_acc_idx = df['val_acc'].idxmax()
    print(f"  Best Val Acc:  {df['val_acc'].max()*100:.2f}% (step {df['step'].iloc[best_val_acc_idx]})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from CSV')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Directory containing training_metrics.csv'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save plot (if not specified, displays interactively)'
    )
    args = parser.parse_args()

    plot_metrics(args.checkpoint_dir, args.save)


if __name__ == '__main__':
    main()
