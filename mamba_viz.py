"""
Mamba Visualization Module - Post-Training Analysis
====================================================
Matplotlib-based visualization suite for model performance analysis.

Creates three core plots:
1. Drift Plot: MSE vs Sequence Position (log-scale)
2. Recency Chart: Recall accuracy by position buckets (bar chart)
3. Density Heatmap: Accuracy vs KV-pair count and sequence length

All plots are saved as high-resolution PNGs in /results/ directory.

This module is CPU-based (matplotlib) and called AFTER training completes,
so it doesn't interfere with H100 training throughput.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import json


# =============================================================================
# Setup and Utilities
# =============================================================================

def setup_results_directory(results_dir: str = "./results"):
    """
    Create results directory if it doesn't exist.

    Args:
        results_dir: Path to results directory
    """
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def set_plot_style():
    """
    Set consistent plot style for all visualizations.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.dpi'] = 150  # High-res output


# =============================================================================
# 1. Drift Plot: MSE vs Sequence Position (Log-Scale)
# =============================================================================

def plot_numerical_drift(
    metrics_log: List[Dict],
    results_dir: str = "./results",
    filename: str = "drift_plot.png",
):
    """
    Create drift plot showing MSE vs sequence position.

    Shows how numerical errors accumulate over sequence length.
    Log-scale y-axis to handle wide range of MSE values.

    Args:
        metrics_log: List of metric dicts with 'drift_per_position' key
        results_dir: Directory to save plot
        filename: Output filename
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Extract drift data from all evaluation steps
    all_drifts = []
    step_labels = []

    for entry in metrics_log:
        if 'drift_per_position' in entry:
            drift = entry['drift_per_position']
            all_drifts.append(drift)

            # Get step number if available
            step = entry.get('step', len(all_drifts) - 1)
            step_labels.append(step)

    if not all_drifts:
        print("Warning: No drift data found in metrics_log. Skipping drift plot.")
        return

    # Convert to numpy array
    all_drifts = np.array(all_drifts)  # (num_evals, seq_len)
    seq_len = all_drifts.shape[1]
    positions = np.arange(seq_len)

    # Plot drift for each evaluation step (with transparency)
    num_evals = len(all_drifts)
    colors = plt.cm.viridis(np.linspace(0, 1, num_evals))

    for i, (drift, step, color) in enumerate(zip(all_drifts, step_labels, colors)):
        alpha = 0.3 if num_evals > 5 else 0.6
        label = f"Step {step}" if i % max(1, num_evals // 5) == 0 else None
        ax.plot(positions, drift, alpha=alpha, color=color, linewidth=1.5, label=label)

    # Plot mean drift (bold line)
    mean_drift = np.mean(all_drifts, axis=0)
    ax.plot(positions, mean_drift, color='red', linewidth=3, label='Mean Drift', zorder=10)

    # Formatting
    ax.set_xlabel('Sequence Position', fontweight='bold')
    ax.set_ylabel('Numerical Drift (MSE / Entropy)', fontweight='bold')
    ax.set_title('Numerical Drift vs Sequence Position\n(Lower is Better)', fontweight='bold', pad=20)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', framealpha=0.9)

    # Add statistics box
    textstr = f'Mean Drift: {mean_drift.mean():.4e}\n'
    textstr += f'Max Drift: {mean_drift.max():.4e}\n'
    # Avoid division by zero
    if mean_drift[0] > 1e-10:
        drift_growth = mean_drift[-1] / mean_drift[0]
        textstr += f'Drift Growth: {drift_growth:.2f}x'
    else:
        textstr += f'Drift Growth: N/A (initial drift ≈ 0)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    output_path = os.path.join(results_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Drift plot saved to: {output_path}")


# =============================================================================
# 2. Recency Chart: Recall Accuracy by Position Buckets (Bar Chart)
# =============================================================================

def plot_recall_by_position(
    metrics_log: List[Dict],
    results_dir: str = "./results",
    filename: str = "recency_chart.png",
):
    """
    Create bar chart showing recall accuracy by position buckets.

    Shows whether model struggles with early vs late recalls
    (recency bias test).

    Args:
        metrics_log: List of metric dicts with 'recall_by_position' key
        results_dir: Directory to save plot
        filename: Output filename
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Extract recall-by-position data (use final evaluation)
    recall_by_pos = None
    for entry in reversed(metrics_log):
        if 'recall_by_position' in entry:
            recall_by_pos = entry['recall_by_position']
            break

    if recall_by_pos is None:
        print("Warning: No recall_by_position data found. Skipping recency chart.")
        return

    num_buckets = len(recall_by_pos)
    bucket_labels = [f"{i*25}-{(i+1)*25}%" for i in range(num_buckets)]

    # Create bar chart
    x_pos = np.arange(num_buckets)
    bars = ax.bar(x_pos, recall_by_pos, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, recall_by_pos)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Color bars by performance (green=good, yellow=medium, red=poor)
    for bar, acc in zip(bars, recall_by_pos):
        if acc >= 0.8:
            bar.set_color('green')
            bar.set_alpha(0.7)
        elif acc >= 0.5:
            bar.set_color('orange')
            bar.set_alpha(0.7)
        else:
            bar.set_color('red')
            bar.set_alpha(0.7)

    # Formatting
    ax.set_xlabel('Relative Sequence Position', fontweight='bold')
    ax.set_ylabel('Recall Accuracy', fontweight='bold')
    ax.set_title('Recall Accuracy by Sequence Position\n(Recency Bias Analysis)', fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (≥80%)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (≥50%)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3, axis='y')

    # Add interpretation text
    recency_bias = recall_by_pos[-1] - recall_by_pos[0]
    bias_text = "Recency Bias" if recency_bias > 0.1 else "No Strong Bias"
    textstr = f'First Quarter: {recall_by_pos[0]:.3f}\n'
    textstr += f'Last Quarter: {recall_by_pos[-1]:.3f}\n'
    textstr += f'Bias: {bias_text} ({recency_bias:+.3f})'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    output_path = os.path.join(results_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Recency chart saved to: {output_path}")


# =============================================================================
# 3. Density Heatmap: Accuracy vs KV-Pair Count
# =============================================================================

def plot_information_density(
    metrics_log: List[Dict],
    results_dir: str = "./results",
    filename: str = "density_heatmap.png",
):
    """
    Create heatmap showing accuracy vs KV-pair count and training progress.

    Shows how model's working memory capacity evolves during training.

    Args:
        metrics_log: List of metric dicts with 'density_accuracies' key
        results_dir: Directory to save plot
        filename: Output filename
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Extract density data from all evaluation steps
    all_density_accs = []
    all_density_counts = []
    step_labels = []

    for entry in metrics_log:
        if 'density_accuracies' in entry:
            all_density_accs.append(entry['density_accuracies'])
            all_density_counts.append(entry['density_counts'])
            step_labels.append(entry.get('step', len(all_density_accs) - 1))

    if not all_density_accs:
        print("Warning: No density data found. Skipping density heatmap.")
        return

    # Convert to numpy arrays
    density_matrix = np.array(all_density_accs)  # (num_evals, max_kv_pairs)
    counts_matrix = np.array(all_density_counts)  # (num_evals, max_kv_pairs)

    # Filter out KV pairs with no data
    kv_has_data = np.sum(counts_matrix, axis=0) > 0
    density_matrix = density_matrix[:, kv_has_data]
    counts_matrix = counts_matrix[:, kv_has_data]
    active_kv_pairs = np.arange(len(kv_has_data))[kv_has_data]

    if density_matrix.shape[1] == 0:
        print("Warning: No KV pairs with data. Skipping density heatmap.")
        return

    # === Plot 1: Heatmap of accuracy over training ===
    im1 = ax1.imshow(density_matrix.T, aspect='auto', cmap='RdYlGn',
                     vmin=0, vmax=1, interpolation='nearest')

    ax1.set_xlabel('Training Step', fontweight='bold')
    ax1.set_ylabel('Number of KV Pairs', fontweight='bold')
    ax1.set_title('Accuracy vs Information Density\n(Across Training)', fontweight='bold', pad=20)

    # Set ticks
    if len(step_labels) <= 10:
        ax1.set_xticks(np.arange(len(step_labels)))
        ax1.set_xticklabels(step_labels, rotation=45)
    else:
        tick_indices = np.linspace(0, len(step_labels)-1, 10, dtype=int)
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels([step_labels[i] for i in tick_indices], rotation=45)

    ax1.set_yticks(np.arange(len(active_kv_pairs)))
    ax1.set_yticklabels(active_kv_pairs)

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Accuracy', rotation=270, labelpad=20, fontweight='bold')

    # === Plot 2: Final accuracy vs KV pairs (line plot) ===
    final_density = density_matrix[-1, :]
    final_counts = counts_matrix[-1, :]

    ax2.plot(active_kv_pairs, final_density, marker='o', linewidth=2.5,
             markersize=8, color='steelblue', label='Accuracy')

    # Add bar plot of counts (secondary y-axis)
    ax2_twin = ax2.twinx()
    ax2_twin.bar(active_kv_pairs, final_counts, alpha=0.3, color='gray', label='# Sequences')

    # Formatting
    ax2.set_xlabel('Number of KV Pairs', fontweight='bold')
    ax2.set_ylabel('Recall Accuracy', fontweight='bold', color='steelblue')
    ax2_twin.set_ylabel('Number of Sequences', fontweight='bold', color='gray')
    ax2.set_title('Final Performance vs Working Memory Load', fontweight='bold', pad=20)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2_twin.tick_params(axis='y', labelcolor='gray')

    # Add threshold lines
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    # Add legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Add capacity analysis text
    capacity_threshold = 0.7
    max_capacity_kv = None
    for kv_idx, acc in zip(active_kv_pairs, final_density):
        if acc < capacity_threshold:
            max_capacity_kv = kv_idx - 1 if kv_idx > 0 else 0
            break
    if max_capacity_kv is None:
        max_capacity_kv = active_kv_pairs[-1]

    textstr = f'Working Memory Capacity:\n'
    textstr += f'Maintains ≥{capacity_threshold:.0%} accuracy\n'
    textstr += f'up to {max_capacity_kv} KV pairs'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.02, textstr, transform=ax2.transAxes, fontsize=11,
             verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    output_path = os.path.join(results_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Density heatmap saved to: {output_path}")


# =============================================================================
# 4. Training Progress Overview (Loss and Accuracy Curves)
# =============================================================================

def plot_training_progress(
    training_history: Dict,
    results_dir: str = "./results",
    filename: str = "training_progress.png",
):
    """
    Create training progress plots showing loss and accuracy over time.

    Args:
        training_history: Dict with 'step', 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        results_dir: Directory to save plot
        filename: Output filename
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    steps = training_history.get('step', [])
    if not steps:
        print("Warning: No training history found. Skipping training progress plot.")
        return

    # === Plot 1: Loss curves ===
    ax1.plot(steps, training_history['train_loss'], label='Train Loss',
             linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    ax1.plot(steps, training_history['val_loss'], label='Val Loss',
             linewidth=2.5, marker='s', markersize=5, alpha=0.8)

    ax1.set_xlabel('Training Step', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontweight='bold', pad=15)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for loss

    # === Plot 2: Accuracy curves ===
    ax2.plot(steps, training_history['train_acc'], label='Train Accuracy',
             linewidth=2.5, marker='o', markersize=5, alpha=0.8, color='green')
    ax2.plot(steps, training_history['val_acc'], label='Val Accuracy',
             linewidth=2.5, marker='s', markersize=5, alpha=0.8, color='orange')

    ax2.set_xlabel('Training Step', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontweight='bold', pad=15)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Add target accuracy line
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target (80%)')

    # Add final metrics text
    final_train_acc = training_history['train_acc'][-1]
    final_val_acc = training_history['val_acc'][-1]
    textstr = f'Final Train Acc: {final_train_acc:.4f}\n'
    textstr += f'Final Val Acc: {final_val_acc:.4f}\n'
    textstr += f'Generalization Gap: {abs(final_train_acc - final_val_acc):.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    output_path = os.path.join(results_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Training progress plot saved to: {output_path}")


# =============================================================================
# Master Function: Generate Complete Performance Report
# =============================================================================

def generate_performance_report(
    metrics_log: List[Dict],
    training_history: Optional[Dict] = None,
    results_dir: str = "./results",
):
    """
    Generate complete performance report with all visualizations.

    This is the main entry point called from train.py after training completes.

    Args:
        metrics_log: List of metric dicts collected during training
        training_history: Optional dict with training loss/accuracy history
        results_dir: Directory to save all plots

    Creates:
        - drift_plot.png: Numerical stability analysis
        - recency_chart.png: Position-based recall analysis
        - density_heatmap.png: Working memory capacity analysis
        - training_progress.png: Loss and accuracy curves (if history provided)
        - metrics_summary.json: Numerical summary of all metrics
    """
    print("\n" + "=" * 60)
    print("GENERATING PERFORMANCE REPORT")
    print("=" * 60)

    # Setup results directory
    results_dir = setup_results_directory(results_dir)
    print(f"Results directory: {results_dir}")

    # Generate individual plots
    print("\nGenerating visualizations...")

    # 1. Drift plot
    plot_numerical_drift(metrics_log, results_dir, "drift_plot.png")

    # 2. Recency chart
    plot_recall_by_position(metrics_log, results_dir, "recency_chart.png")

    # 3. Density heatmap
    plot_information_density(metrics_log, results_dir, "density_heatmap.png")

    # 4. Training progress (if history provided)
    if training_history is not None:
        plot_training_progress(training_history, results_dir, "training_progress.png")

    # Save metrics summary as JSON
    print("\nSaving metrics summary...")
    summary_path = os.path.join(results_dir, "metrics_summary.json")

    def convert_to_serializable(obj):
        """Convert NumPy/JAX arrays to JSON-serializable types."""
        import numpy as np
        if isinstance(obj, (np.ndarray, np.generic)):
            if obj.ndim == 0:  # Scalar
                return obj.item()
            else:  # Array
                return obj.tolist()
        elif hasattr(obj, 'item'):  # JAX array or 0-d NumPy array
            try:
                return obj.item()
            except ValueError:  # Multi-element array
                return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        else:
            return str(obj)

    with open(summary_path, 'w') as f:
        json.dump(metrics_log, f, indent=2, default=convert_to_serializable)
    print(f"✓ Metrics summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT COMPLETE")
    print(f"All visualizations saved to: {results_dir}")
    print("=" * 60)


# =============================================================================
# Quick Summary Statistics
# =============================================================================

def print_metrics_summary(metrics_log: List[Dict]):
    """
    Print human-readable summary of final metrics.

    Args:
        metrics_log: List of metric dicts
    """
    if not metrics_log:
        print("No metrics to summarize.")
        return

    final_metrics = metrics_log[-1]

    print("\n" + "=" * 60)
    print("FINAL METRICS SUMMARY")
    print("=" * 60)

    # EM Accuracy
    if 'em_accuracy' in final_metrics:
        print(f"\nExact Match Accuracy: {final_metrics['em_accuracy']:.4f}")

    # Recall by Position
    if 'recall_by_position' in final_metrics:
        recall = final_metrics['recall_by_position']
        print(f"\nRecall by Position:")
        for i, acc in enumerate(recall):
            print(f"  Bucket {i} ({i*25}-{(i+1)*25}%): {acc:.4f}")

    # Numerical Drift
    if 'drift_mse' in final_metrics:
        print(f"\nNumerical Drift (MSE): {final_metrics['drift_mse']:.6e}")

    # Information Density
    if 'density_accuracies' in final_metrics:
        density_accs = final_metrics['density_accuracies']
        density_counts = final_metrics.get('density_counts', [])
        print(f"\nInformation Density (Active KV Pairs):")
        for kv_idx, (acc, count) in enumerate(zip(density_accs, density_counts)):
            if count > 0:
                print(f"  {kv_idx} KV pairs: {acc:.4f} ({int(count)} sequences)")

    print("=" * 60)
