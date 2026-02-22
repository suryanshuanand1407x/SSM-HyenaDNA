#!/usr/bin/env python3
"""
Simple Training Metrics Plotter
================================
Quick visualization for metrics.csv from genomics training.

Usage:
    python plot_training.py ./checkpoints/rc_equivariant_20k/metrics.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def plot_metrics(csv_path: str):
    """Plot training metrics from CSV file."""

    # Load data
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} checkpoints from {csv_path}")
    print(f"Steps: {df['step'].min()} → {df['step'].max()}\n")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(df['step'], df['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
    ax.plot(df['step'], df['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=4)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best val loss
    best_idx = df['val_loss'].idxmin()
    best_step = df.loc[best_idx, 'step']
    best_loss = df.loc[best_idx, 'val_loss']
    ax.axvline(best_step, color='red', linestyle='--', alpha=0.5)
    ax.text(best_step, best_loss, f'  Best: {best_loss:.4f}', fontsize=9, color='red')

    # Plot 2: Accuracy curves
    ax = axes[0, 1]
    if 'train_acc' in df.columns and 'val_acc' in df.columns:
        ax.plot(df['step'], df['train_acc'], 'o-', label='Train Acc', linewidth=2, markersize=4)
        ax.plot(df['step'], df['val_acc'], 's-', label='Val Acc', linewidth=2, markersize=4)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mark best val acc
        best_idx = df['val_acc'].idxmax()
        best_step = df.loc[best_idx, 'step']
        best_acc = df.loc[best_idx, 'val_acc']
        ax.axvline(best_step, color='green', linestyle='--', alpha=0.5)
        ax.text(best_step, best_acc, f'  Best: {best_acc:.4f}', fontsize=9, color='green')
    else:
        ax.text(0.5, 0.5, 'No accuracy data', ha='center', va='center', fontsize=14)
        ax.set_title('Accuracy (No Data)', fontsize=14)

    # Plot 3: Loss gap (overfitting)
    ax = axes[1, 0]
    gap = df['val_loss'] - df['train_loss']
    ax.plot(df['step'], gap, 'o-', color='#E63946', linewidth=2, markersize=4)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(df['step'], 0, gap, where=(gap>0), alpha=0.3, color='red', label='Overfitting')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax.set_title('Generalization Gap', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary stats
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate stats
    final = df.iloc[-1]
    best_val_idx = df['val_loss'].idxmin()
    best_val = df.iloc[best_val_idx]

    stats_text = f"""
    TRAINING SUMMARY
    {'='*40}

    Total Steps: {int(df['step'].max())}
    Checkpoints: {len(df)}

    FINAL METRICS (Step {int(final['step'])}):
    Train Loss:  {final['train_loss']:.6f}
    Val Loss:    {final['val_loss']:.6f}
    """

    if 'train_acc' in df.columns:
        stats_text += f"    Train Acc:   {final['train_acc']:.4f}\n"
        stats_text += f"    Val Acc:     {final['val_acc']:.4f}\n"

    stats_text += f"""
    BEST VALIDATION (Step {int(best_val['step'])}):
    Best Val Loss: {best_val['val_loss']:.6f}
    """

    if 'val_acc' in df.columns:
        best_acc_idx = df['val_acc'].idxmax()
        best_acc = df.iloc[best_acc_idx]
        stats_text += f"    Best Val Acc:  {best_acc['val_acc']:.4f} (Step {int(best_acc['step'])})\n"

    stats_text += f"""
    IMPROVEMENT:
    Loss: {df['val_loss'].iloc[0]:.4f} → {final['val_loss']:.4f}
    """

    if 'val_acc' in df.columns:
        stats_text += f"    Acc:  {df['val_acc'].iloc[0]:.4f} → {final['val_acc']:.4f}\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save
    output_path = Path(csv_path).parent / "training_plots.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")

    plt.show()

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Steps: {int(df['step'].min())} → {int(df['step'].max())}")
    print(f"\nFinal Metrics (Step {int(final['step'])}):")
    print(f"  Train Loss: {final['train_loss']:.6f}")
    print(f"  Val Loss:   {final['val_loss']:.6f}")
    if 'train_acc' in df.columns:
        print(f"  Train Acc:  {final['train_acc']:.4f}")
        print(f"  Val Acc:    {final['val_acc']:.4f}")

    print(f"\nBest Validation (Step {int(best_val['step'])}):")
    print(f"  Val Loss: {best_val['val_loss']:.6f}")
    if 'val_acc' in df.columns:
        print(f"  Val Acc:  {df['val_acc'].max():.4f} (Step {int(df.loc[df['val_acc'].idxmax(), 'step'])})")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training.py <path/to/metrics.csv>")
        print("\nExample:")
        print("  python plot_training.py ./checkpoints/rc_equivariant_20k/metrics.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    plot_metrics(csv_path)
