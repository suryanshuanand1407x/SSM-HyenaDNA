#!/usr/bin/env python3
"""
Training Results Visualization
==============================
Comprehensive visualization tools for Tustin-Mamba training metrics.

Usage:
    # Visualize single run
    python visualize_results.py ./checkpoints/rc_equivariant_20k/metrics.csv

    # Compare multiple runs
    python visualize_results.py \
        --compare \
        ./checkpoints/stable_20k/metrics.csv \
        ./checkpoints/rc_equivariant_20k/metrics.csv

    # Generate full report
    python visualize_results.py --report ./checkpoints/rc_equivariant_20k/
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2


# =============================================================================
# Data Loading
# =============================================================================

def load_metrics(csv_path: str) -> pd.DataFrame:
    """
    Load metrics from CSV file.

    Args:
        csv_path: Path to metrics.csv

    Returns:
        DataFrame with metrics
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Metrics file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def get_run_name(csv_path: str) -> str:
    """Extract run name from path."""
    path = Path(csv_path)
    # Get parent directory name (e.g., "rc_equivariant_20k")
    return path.parent.name


# =============================================================================
# Single Run Visualization
# =============================================================================

def plot_loss_curves(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot training and validation loss curves.

    Args:
        df: DataFrame with 'step', 'train_loss', 'val_loss'
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step'], df['train_loss'], label='Train Loss', color='#2E86AB', linewidth=2)
    ax.plot(df['step'], df['val_loss'], label='Val Loss', color='#A23B72', linewidth=2)

    ax.set_xlabel('Training Step', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Training and Validation Loss', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add best validation loss marker
    best_val_idx = df['val_loss'].idxmin()
    best_val_step = df.loc[best_val_idx, 'step']
    best_val_loss = df.loc[best_val_idx, 'val_loss']

    ax.axvline(best_val_step, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(best_val_step, best_val_loss, f'  Best: {best_val_loss:.4f}\n  Step: {int(best_val_step)}',
            fontsize=10, color='red')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()


def plot_accuracy_curves(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot training and validation accuracy curves.

    Args:
        df: DataFrame with 'step', 'train_acc', 'val_acc'
        save_path: Optional path to save figure
    """
    if 'train_acc' not in df.columns or 'val_acc' not in df.columns:
        print("⚠️  No accuracy columns found in metrics")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step'], df['train_acc'], label='Train Accuracy', color='#06A77D', linewidth=2)
    ax.plot(df['step'], df['val_acc'], label='Val Accuracy', color='#D4A017', linewidth=2)

    ax.set_xlabel('Training Step', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Training and Validation Accuracy', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add best validation accuracy marker
    best_val_idx = df['val_acc'].idxmax()
    best_val_step = df.loc[best_val_idx, 'step']
    best_val_acc = df.loc[best_val_idx, 'val_acc']

    ax.axvline(best_val_step, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(best_val_step, best_val_acc, f'  Best: {best_val_acc:.4f}\n  Step: {int(best_val_step)}',
            fontsize=10, color='green')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()


def plot_combined_metrics(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot loss and accuracy in subplots.

    Args:
        df: DataFrame with metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Loss subplot
    ax1 = axes[0]
    ax1.plot(df['step'], df['train_loss'], label='Train Loss', color='#2E86AB', linewidth=2)
    ax1.plot(df['step'], df['val_loss'], label='Val Loss', color='#A23B72', linewidth=2)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy subplot
    if 'train_acc' in df.columns and 'val_acc' in df.columns:
        ax2 = axes[1]
        ax2.plot(df['step'], df['train_acc'], label='Train Accuracy', color='#06A77D', linewidth=2)
        ax2.plot(df['step'], df['val_acc'], label='Val Accuracy', color='#D4A017', linewidth=2)
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()


def plot_overfitting_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot train-val gap to analyze overfitting.

    Args:
        df: DataFrame with metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss gap
    ax1 = axes[0]
    loss_gap = df['val_loss'] - df['train_loss']
    ax1.plot(df['step'], loss_gap, color='#E63946', linewidth=2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.fill_between(df['step'], 0, loss_gap, where=(loss_gap > 0), alpha=0.3, color='#E63946', label='Overfitting')
    ax1.fill_between(df['step'], 0, loss_gap, where=(loss_gap < 0), alpha=0.3, color='#06A77D', label='Underfitting')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax1.set_title('Loss Gap (Overfitting Analysis)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy gap (if available)
    if 'train_acc' in df.columns and 'val_acc' in df.columns:
        ax2 = axes[1]
        acc_gap = df['train_acc'] - df['val_acc']
        ax2.plot(df['step'], acc_gap, color='#F77F00', linewidth=2)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(df['step'], 0, acc_gap, where=(acc_gap > 0), alpha=0.3, color='#F77F00', label='Overfitting')
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Train Acc - Val Acc', fontsize=12)
        ax2.set_title('Accuracy Gap (Overfitting Analysis)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()


def plot_smoothed_metrics(df: pd.DataFrame, window: int = 5, save_path: Optional[str] = None):
    """
    Plot smoothed loss curves for better visualization.

    Args:
        df: DataFrame with metrics
        window: Smoothing window size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Original curves (transparent)
    ax.plot(df['step'], df['train_loss'], color='#2E86AB', alpha=0.2, linewidth=1)
    ax.plot(df['step'], df['val_loss'], color='#A23B72', alpha=0.2, linewidth=1)

    # Smoothed curves
    train_smooth = df['train_loss'].rolling(window=window, center=True).mean()
    val_smooth = df['val_loss'].rolling(window=window, center=True).mean()

    ax.plot(df['step'], train_smooth, label=f'Train Loss (smoothed)', color='#2E86AB', linewidth=2.5)
    ax.plot(df['step'], val_smooth, label=f'Val Loss (smoothed)', color='#A23B72', linewidth=2.5)

    ax.set_xlabel('Training Step', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(f'Smoothed Loss Curves (window={window})', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()


# =============================================================================
# Comparison Visualization
# =============================================================================

def plot_comparison(csv_paths: List[str], save_path: Optional[str] = None):
    """
    Compare multiple training runs.

    Args:
        csv_paths: List of paths to metrics.csv files
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ['#2E86AB', '#A23B72', '#06A77D', '#D4A017', '#E63946', '#F77F00']

    for idx, csv_path in enumerate(csv_paths):
        df = load_metrics(csv_path)
        run_name = get_run_name(csv_path)
        color = colors[idx % len(colors)]

        # Train loss
        axes[0, 0].plot(df['step'], df['train_loss'], label=run_name, color=color, linewidth=2)

        # Val loss
        axes[0, 1].plot(df['step'], df['val_loss'], label=run_name, color=color, linewidth=2)

        # Train accuracy (if available)
        if 'train_acc' in df.columns:
            axes[1, 0].plot(df['step'], df['train_acc'], label=run_name, color=color, linewidth=2)

        # Val accuracy (if available)
        if 'val_acc' in df.columns:
            axes[1, 1].plot(df['step'], df['val_acc'], label=run_name, color=color, linewidth=2)

    # Configure subplots
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Step', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Step', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Step', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Step', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()


def plot_final_comparison_table(csv_paths: List[str], save_path: Optional[str] = None):
    """
    Create comparison table of final metrics.

    Args:
        csv_paths: List of paths to metrics.csv files
        save_path: Optional path to save figure
    """
    results = []

    for csv_path in csv_paths:
        df = load_metrics(csv_path)
        run_name = get_run_name(csv_path)

        final_row = df.iloc[-1]
        best_val_idx = df['val_loss'].idxmin()
        best_val_row = df.iloc[best_val_idx]

        results.append({
            'Run': run_name,
            'Final Train Loss': final_row['train_loss'],
            'Final Val Loss': final_row['val_loss'],
            'Best Val Loss': best_val_row['val_loss'],
            'Best Val Step': int(best_val_row['step']),
            'Final Train Acc': final_row.get('train_acc', np.nan),
            'Final Val Acc': final_row.get('val_acc', np.nan),
            'Best Val Acc': df['val_acc'].max() if 'val_acc' in df.columns else np.nan,
        })

    results_df = pd.DataFrame(results)

    # Create table plot
    fig, ax = plt.subplots(figsize=(14, len(results) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=results_df.values,
                     colLabels=results_df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(len(results_df.columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(results_df) + 1):
        for j in range(len(results_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')

    plt.title('Training Results Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.show()

    # Print table to console
    print("\n" + "=" * 80)
    print("FINAL METRICS COMPARISON")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)


# =============================================================================
# Full Report Generation
# =============================================================================

def generate_full_report(checkpoint_dir: str, output_dir: Optional[str] = None):
    """
    Generate complete visualization report for a training run.

    Args:
        checkpoint_dir: Directory containing metrics.csv
        output_dir: Directory to save plots (default: checkpoint_dir/plots)
    """
    checkpoint_path = Path(checkpoint_dir)
    metrics_csv = checkpoint_path / "metrics.csv"

    if not metrics_csv.exists():
        raise FileNotFoundError(f"No metrics.csv found in {checkpoint_dir}")

    # Create output directory
    if output_dir is None:
        output_dir = checkpoint_path / "plots"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Generating Full Training Report")
    print(f"{'='*80}")
    print(f"Metrics: {metrics_csv}")
    print(f"Output:  {output_path}")
    print(f"{'='*80}\n")

    # Load data
    df = load_metrics(str(metrics_csv))

    # Generate plots
    print("Generating plots...")

    plot_loss_curves(df, save_path=str(output_path / "loss_curves.png"))
    plot_accuracy_curves(df, save_path=str(output_path / "accuracy_curves.png"))
    plot_combined_metrics(df, save_path=str(output_path / "combined_metrics.png"))
    plot_overfitting_analysis(df, save_path=str(output_path / "overfitting_analysis.png"))
    plot_smoothed_metrics(df, window=5, save_path=str(output_path / "smoothed_loss.png"))

    print(f"\n{'='*80}")
    print(f"✓ Report generated successfully!")
    print(f"{'='*80}")
    print(f"Plots saved to: {output_path}")
    print(f"{'='*80}\n")


# =============================================================================
# Main CLI
# =============================================================================

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Visualize Tustin-Mamba training results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize single run
  python visualize_results.py ./checkpoints/rc_equivariant_20k/metrics.csv

  # Compare multiple runs
  python visualize_results.py --compare \
      ./checkpoints/stable_20k/metrics.csv \
      ./checkpoints/rc_equivariant_20k/metrics.csv

  # Generate full report
  python visualize_results.py --report ./checkpoints/rc_equivariant_20k/

  # Save plots
  python visualize_results.py metrics.csv --save ./plots/
        """
    )

    parser.add_argument(
        'paths',
        nargs='+',
        help='Path(s) to metrics.csv file(s) or checkpoint directory'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple runs'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate full report with all plots'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Directory to save plots'
    )
    parser.add_argument(
        '--smooth',
        type=int,
        default=5,
        help='Smoothing window size (default: 5)'
    )

    args = parser.parse_args()

    # Generate full report
    if args.report:
        for path in args.paths:
            generate_full_report(path, output_dir=args.save)
        return

    # Compare multiple runs
    if args.compare:
        if len(args.paths) < 2:
            print("❌ Need at least 2 metrics files for comparison")
            return

        save_path = None
        if args.save:
            save_path = Path(args.save) / "comparison.png"
            Path(args.save).mkdir(parents=True, exist_ok=True)

        plot_comparison(args.paths, save_path=save_path)
        plot_final_comparison_table(args.paths,
                                    save_path=Path(args.save) / "comparison_table.png" if args.save else None)
        return

    # Single run visualization
    csv_path = args.paths[0]
    df = load_metrics(csv_path)

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Visualizing: {csv_path}")
    print(f"{'='*80}\n")

    plot_combined_metrics(df, save_path=str(save_dir / "combined.png") if save_dir else None)
    plot_overfitting_analysis(df, save_path=str(save_dir / "overfitting.png") if save_dir else None)
    plot_smoothed_metrics(df, window=args.smooth,
                         save_path=str(save_dir / "smoothed.png") if save_dir else None)


if __name__ == "__main__":
    main()
