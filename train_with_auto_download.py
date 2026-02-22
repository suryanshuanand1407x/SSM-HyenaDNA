#!/usr/bin/env python3
"""
HyenaDNA Training with Automatic Dataset Download
=================================================
This script automatically:
1. Downloads HG38 dataset if not present
2. Downloads HyenaDNA (medium) pretrained weights from HuggingFace
3. Swaps Hyena blocks with Tustin Mamba blocks
4. Trains the model

Usage:
    python train_with_auto_download.py                    # Quick test
    python train_with_auto_download.py --config tustin    # Full Tustin training
    python train_with_auto_download.py --config custom    # Custom hyperparameters
    python train_with_auto_download.py --resume           # Resume from checkpoint
"""

import sys
import os
from pathlib import Path

def check_and_install_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'jax',
        'flax',
        'optax',
        'transformers',
        'pyfaidx',
        'pandas',
        'tqdm',
        'numpy'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("=" * 70)
        print("Missing Required Packages")
        print("=" * 70)
        print("\nThe following packages are required but not installed:")
        for pkg in missing:
            print(f"  - {pkg}")

        print("\nPlease install them using:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr run the full installation script:")
        print("  bash install_requirements.sh")
        print("=" * 70)
        return False

    return True


def verify_setup():
    """Verify the training setup is ready."""
    print("=" * 70)
    print("HyenaDNA Training Setup Verification")
    print("=" * 70)

    # Check packages
    print("\n[1/3] Checking Python packages...")
    if not check_and_install_requirements():
        return False
    print("  ✓ All required packages installed")

    # Check dataset (will auto-download if missing)
    print("\n[2/3] Checking HG38 dataset...")
    try:
        from download_hg38 import ensure_dataset_downloaded
        data_dir = "./data/hg38"

        # This will download if not present
        if not ensure_dataset_downloaded(data_dir):
            print("  ✗ Dataset download failed")
            return False
        print("  ✓ HG38 dataset ready")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    # Check model files
    print("\n[3/3] Checking model configuration...")
    try:
        from config_hyena import HyenaFineTuneConfig, QUICK_CONFIG
        from model_hybrid import create_hybrid_model
        import jax
        print("  ✓ Model configuration ready")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    print("\n" + "=" * 70)
    print("✓ Setup verification complete! Ready to train.")
    print("=" * 70)
    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train HyenaDNA with Tustin Mamba blocks (automatic setup)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (small model, 1000 steps)
  python train_with_auto_download.py

  # Full Tustin training (512d, 6 layers, 100K steps)
  python train_with_auto_download.py --config tustin

  # Custom hyperparameters (edit train_hyena.py)
  python train_with_auto_download.py --config custom

  # Resume from checkpoint
  python train_with_auto_download.py --resume --config tustin

The script will automatically:
  1. Download HG38 dataset (~3.1 GB) if not present
  2. Download HyenaDNA-medium weights from HuggingFace
  3. Initialize Tustin Mamba blocks
  4. Train with 2-phase fine-tuning
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='quick',
        choices=['quick', 'tustin', 'zoh', 'custom'],
        help='Configuration preset (default: quick for testing)'
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
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip setup verification (faster startup)'
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "=" * 70)
    print("HyenaDNA → Tustin Mamba Training")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Preset: {args.config}")
    print(f"  Resume: {args.resume}")
    if args.checkpoint_dir:
        print(f"  Checkpoint Dir: {args.checkpoint_dir}")
    print()

    # Verify setup (unless skipped)
    if not args.skip_verify:
        if not verify_setup():
            print("\n❌ Setup verification failed. Please fix errors above.")
            sys.exit(1)
    else:
        print("Skipping setup verification...")

    # Import and run training
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    try:
        # Import the main training script
        import train_hyena

        # Build arguments for train_hyena
        sys.argv = ['train_hyena.py', '--config', args.config]
        if args.resume:
            sys.argv.append('--resume')
        if args.checkpoint_dir:
            sys.argv.extend(['--checkpoint_dir', args.checkpoint_dir])

        # Run training
        train_hyena.main()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
