"""
Training Monitor for RTX 5090
============================
Real-time monitoring of training progress, GPU usage, and metrics
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import argparse


def get_gpu_stats():
    """Get GPU utilization and memory stats."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                'gpu_util': float(stats[0]),
                'mem_used': float(stats[1]),
                'mem_total': float(stats[2]),
                'temp': float(stats[3]),
                'power': float(stats[4])
            }
    except Exception as e:
        return None
    return None


def get_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_') and f.endswith('.pkl')
    ]

    if not checkpoints:
        return None

    checkpoints.sort()
    latest = checkpoints[-1]

    # Extract step number
    step = int(latest.replace('checkpoint_', '').replace('.pkl', ''))

    return step, os.path.join(checkpoint_dir, latest)


def read_metrics_file(results_dir):
    """Read training metrics from results directory."""
    metrics_file = os.path.join(results_dir, 'final_metrics.txt')

    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            return f.read()
    return None


def format_bytes(bytes_val):
    """Format bytes to human readable."""
    for unit in ['MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def monitor_training(checkpoint_dir, results_dir, interval=10):
    """
    Monitor training progress.

    Args:
        checkpoint_dir: Directory containing checkpoints
        results_dir: Directory containing results
        interval: Update interval in seconds
    """
    print("\n" + "=" * 60)
    print("Training Monitor (RTX 5090)")
    print("=" * 60)
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"Results Dir: {results_dir}")
    print(f"Update Interval: {interval}s")
    print("\nPress Ctrl+C to exit")
    print("=" * 60 + "\n")

    last_step = 0
    last_time = time.time()

    try:
        while True:
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')

            print("=" * 60)
            print(f"Training Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60 + "\n")

            # GPU stats
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                print("GPU Status:")
                print(f"  Utilization: {gpu_stats['gpu_util']:.0f}%")
                print(f"  Memory:      {gpu_stats['mem_used']:.0f} / {gpu_stats['mem_total']:.0f} MB " +
                      f"({100 * gpu_stats['mem_used'] / gpu_stats['mem_total']:.1f}%)")
                print(f"  Temperature: {gpu_stats['temp']:.0f}°C")
                print(f"  Power:       {gpu_stats['power']:.0f}W")
            else:
                print("GPU Status: Not available")

            print()

            # Checkpoint info
            checkpoint_info = get_latest_checkpoint(checkpoint_dir)
            if checkpoint_info:
                current_step, checkpoint_path = checkpoint_info

                # Calculate throughput
                time_elapsed = time.time() - last_time
                steps_per_sec = (current_step - last_step) / time_elapsed if time_elapsed > 0 else 0

                print("Training Progress:")
                print(f"  Current Step: {current_step:,}")
                print(f"  Latest Checkpoint: {os.path.basename(checkpoint_path)}")

                if steps_per_sec > 0:
                    print(f"  Speed: {steps_per_sec:.2f} steps/sec")

                # File size
                file_size = os.path.getsize(checkpoint_path)
                print(f"  Checkpoint Size: {format_bytes(file_size / 1024 / 1024)}")

                last_step = current_step
                last_time = time.time()
            else:
                print("Training Progress: No checkpoints yet")

            print()

            # Metrics
            metrics = read_metrics_file(results_dir)
            if metrics:
                print("Latest Metrics:")
                for line in metrics.split('\n'):
                    if line.strip():
                        print(f"  {line}")
            else:
                print("Metrics: Not available yet")

            print("\n" + "=" * 60)
            print(f"Next update in {interval}s... (Ctrl+C to exit)")

            # Sleep
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument(
        '--config',
        type=str,
        default='tustin',
        choices=['quick', 'tustin', 'zoh', 'large'],
        help='Config to monitor'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Update interval in seconds'
    )
    args = parser.parse_args()

    # Determine directories based on config
    config_to_dirs = {
        'quick': ('checkpoints/hyena_mamba_quick', 'results/quick_test'),
        'tustin': ('checkpoints/hyena_mamba_tustin', 'results/tustin_comparison'),
        'zoh': ('checkpoints/hyena_mamba_zoh', 'results/zoh_comparison'),
        'large': ('checkpoints/hyena_mamba_large', 'results/large_scale'),
    }

    checkpoint_dir, results_dir = config_to_dirs[args.config]

    monitor_training(checkpoint_dir, results_dir, args.interval)


if __name__ == '__main__':
    main()
