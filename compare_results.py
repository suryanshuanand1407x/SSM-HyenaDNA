"""
Compare Tustin vs ZOH Results
============================
Analyze and visualize comparison between Tustin and ZOH Mamba variants
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_checkpoint(checkpoint_path):
    """Load checkpoint and extract metrics."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def parse_metrics_file(metrics_path):
    """Parse final_metrics.txt file."""
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    # Try to convert to float
                    try:
                        value = float(value)
                    except:
                        pass
                    metrics[key] = value
    return metrics


def compare_configurations(tustin_dir, zoh_dir, output_dir='./results/comparison'):
    """
    Compare Tustin and ZOH configurations.

    Args:
        tustin_dir: Tustin checkpoint directory
        zoh_dir: ZOH checkpoint directory
        output_dir: Where to save comparison results
    """
    print("\n" + "=" * 60)
    print("Tustin vs ZOH Comparison Analysis")
    print("=" * 60 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load metrics
    tustin_metrics_path = os.path.join(tustin_dir.replace('checkpoints', 'results'), 'final_metrics.txt')
    zoh_metrics_path = os.path.join(zoh_dir.replace('checkpoints', 'results'), 'final_metrics.txt')

    print("Loading metrics...")
    tustin_metrics = parse_metrics_file(tustin_metrics_path)
    zoh_metrics = parse_metrics_file(zoh_metrics_path)

    if not tustin_metrics:
        print(f"✗ Tustin metrics not found at: {tustin_metrics_path}")
        return

    if not zoh_metrics:
        print(f"✗ ZOH metrics not found at: {zoh_metrics_path}")
        return

    print("✓ Metrics loaded\n")

    # Print comparison table
    print("=" * 60)
    print("Final Metrics Comparison")
    print("=" * 60)
    print(f"{'Metric':<30} {'Tustin':<15} {'ZOH':<15} {'Winner':<10}")
    print("-" * 60)

    # Compare key metrics
    key_metrics = ['Final Train Loss', 'Final Val Loss']

    for metric in key_metrics:
        tustin_val = tustin_metrics.get(metric, None)
        zoh_val = zoh_metrics.get(metric, None)

        if tustin_val is not None and zoh_val is not None:
            # Lower is better for loss
            winner = "Tustin" if tustin_val < zoh_val else "ZOH"
            if abs(tustin_val - zoh_val) < 0.001:
                winner = "Tie"

            print(f"{metric:<30} {tustin_val:<15.4f} {zoh_val:<15.4f} {winner:<10}")

    print()

    # Calculate improvement
    if 'Final Val Loss' in tustin_metrics and 'Final Val Loss' in zoh_metrics:
        tustin_loss = tustin_metrics['Final Val Loss']
        zoh_loss = zoh_metrics['Final Val Loss']

        if tustin_loss < zoh_loss:
            improvement = ((zoh_loss - tustin_loss) / zoh_loss) * 100
            print(f"Tustin is {improvement:.2f}% better than ZOH (validation loss)")
        else:
            improvement = ((tustin_loss - zoh_loss) / tustin_loss) * 100
            print(f"ZOH is {improvement:.2f}% better than Tustin (validation loss)")

    print()

    # Save comparison report
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("Tustin vs ZOH Comparison Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("Tustin Metrics:\n")
        for key, value in tustin_metrics.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nZOH Metrics:\n")
        for key, value in zoh_metrics.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Conclusion:\n")
        f.write("=" * 60 + "\n")

        if 'Final Val Loss' in tustin_metrics and 'Final Val Loss' in zoh_metrics:
            tustin_loss = tustin_metrics['Final Val Loss']
            zoh_loss = zoh_metrics['Final Val Loss']

            if tustin_loss < zoh_loss:
                improvement = ((zoh_loss - tustin_loss) / zoh_loss) * 100
                f.write(f"Tustin discretization outperforms ZOH by {improvement:.2f}%\n")
                f.write("Tustin provides better stability and accuracy for genomic sequences.\n")
            else:
                improvement = ((tustin_loss - zoh_loss) / tustin_loss) * 100
                f.write(f"ZOH discretization outperforms Tustin by {improvement:.2f}%\n")
                f.write("Standard ZOH remains competitive for genomic sequences.\n")

    print(f"✓ Comparison report saved to: {report_path}")

    # Create visualization
    print("\nGenerating comparison plots...")
    create_comparison_plots(tustin_metrics, zoh_metrics, output_dir)

    print(f"\n✓ All results saved to: {output_dir}")
    print("\n" + "=" * 60)


def create_comparison_plots(tustin_metrics, zoh_metrics, output_dir):
    """Create comparison visualization."""

    # Extract metrics for plotting
    metrics_to_plot = []
    labels = []

    if 'Final Train Loss' in tustin_metrics:
        metrics_to_plot.append(('Train Loss',
                                tustin_metrics['Final Train Loss'],
                                zoh_metrics['Final Train Loss']))

    if 'Final Val Loss' in tustin_metrics:
        metrics_to_plot.append(('Val Loss',
                                tustin_metrics['Final Val Loss'],
                                zoh_metrics['Final Val Loss']))

    if not metrics_to_plot:
        print("No metrics available for plotting")
        return

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    tustin_vals = [m[1] for m in metrics_to_plot]
    zoh_vals = [m[2] for m in metrics_to_plot]
    labels = [m[0] for m in metrics_to_plot]

    rects1 = ax.bar(x - width/2, tustin_vals, width, label='Tustin', color='#2E86AB')
    rects2 = ax.bar(x + width/2, zoh_vals, width, label='ZOH', color='#A23B72')

    ax.set_ylabel('Loss')
    ax.set_title('Tustin vs ZOH Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare Tustin and ZOH results')
    parser.add_argument(
        '--tustin_dir',
        type=str,
        default='./checkpoints/hyena_mamba_tustin',
        help='Tustin checkpoint directory'
    )
    parser.add_argument(
        '--zoh_dir',
        type=str,
        default='./checkpoints/hyena_mamba_zoh',
        help='ZOH checkpoint directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/comparison',
        help='Output directory for comparison results'
    )
    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(args.tustin_dir):
        print(f"✗ Tustin directory not found: {args.tustin_dir}")
        print("Make sure you've trained the Tustin model first:")
        print("  python train_hyena.py --config tustin")
        return

    if not os.path.exists(args.zoh_dir):
        print(f"✗ ZOH directory not found: {args.zoh_dir}")
        print("Make sure you've trained the ZOH model first:")
        print("  python train_hyena.py --config zoh")
        return

    compare_configurations(args.tustin_dir, args.zoh_dir, args.output_dir)


if __name__ == '__main__':
    main()
