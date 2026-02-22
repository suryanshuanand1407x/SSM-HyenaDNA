#!/usr/bin/env python3
"""
Training Metrics Viewer
=======================
View and analyze training metrics from CSV file
"""

import pandas as pd
import sys
from pathlib import Path


def view_metrics(metrics_file: str):
    """
    Display training metrics in a nice format.

    Args:
        metrics_file: Path to metrics CSV file
    """
    if not Path(metrics_file).exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        print("\nMake sure training has started and saved at least one checkpoint.")
        return

    # Load metrics
    df = pd.read_csv(metrics_file)

    print("\n" + "="*80)
    print("TRAINING METRICS SUMMARY")
    print("="*80)

    # Summary statistics
    print(f"\nTotal checkpoints: {len(df)}")
    print(f"Steps recorded: {df['step'].min()} → {df['step'].max()}")

    print("\n" + "-"*80)
    print("LATEST METRICS (Most Recent Checkpoint)")
    print("-"*80)

    latest = df.iloc[-1]
    print(f"Step: {int(latest['step'])}")
    print(f"\n{'Metric':<20} {'Value':<15}")
    print(f"{'-'*35}")
    print(f"{'Train Loss':<20} {latest['train_loss']:<15.6f}")
    print(f"{'Train Accuracy':<20} {latest['train_acc']:<15.4f}")
    print(f"{'Val Loss':<20} {latest['val_loss']:<15.6f}")
    print(f"{'Val Accuracy':<20} {latest['val_acc']:<15.4f}")

    print("\n" + "-"*80)
    print("BEST METRICS (Across All Checkpoints)")
    print("-"*80)

    print(f"\n{'Metric':<25} {'Best Value':<15} {'At Step':<10}")
    print(f"{'-'*50}")
    print(f"{'Lowest Train Loss':<25} {df['train_loss'].min():<15.6f} {int(df.loc[df['train_loss'].idxmin(), 'step']):<10}")
    print(f"{'Highest Train Acc':<25} {df['train_acc'].max():<15.4f} {int(df.loc[df['train_acc'].idxmax(), 'step']):<10}")
    print(f"{'Lowest Val Loss':<25} {df['val_loss'].min():<15.6f} {int(df.loc[df['val_loss'].idxmin(), 'step']):<10}")
    print(f"{'Highest Val Acc':<25} {df['val_acc'].max():<15.4f} {int(df.loc[df['val_acc'].idxmax(), 'step']):<10}")

    print("\n" + "-"*80)
    print("FULL TRAINING HISTORY")
    print("-"*80)
    print()

    # Format the dataframe for display
    df_display = df.copy()
    df_display['step'] = df_display['step'].astype(int)
    df_display['train_loss'] = df_display['train_loss'].apply(lambda x: f"{x:.6f}")
    df_display['train_acc'] = df_display['train_acc'].apply(lambda x: f"{x:.4f}")
    df_display['val_loss'] = df_display['val_loss'].apply(lambda x: f"{x:.6f}")
    df_display['val_acc'] = df_display['val_acc'].apply(lambda x: f"{x:.4f}")

    print(df_display.to_string(index=False))

    print("\n" + "="*80)
    print()


def main():
    """Main function."""
    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]
    else:
        # Default location
        metrics_file = "./checkpoints/stable_20k/metrics.csv"

    print(f"\nLoading metrics from: {metrics_file}\n")
    view_metrics(metrics_file)


if __name__ == "__main__":
    main()
