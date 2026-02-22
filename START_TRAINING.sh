#!/bin/bash
#
# Quick Start Training Script
# ============================
# This script automatically downloads data and starts training
#

set -e  # Exit on error

echo "=========================================================================="
echo "HyenaDNA → Tustin Mamba Training"
echo "=========================================================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Determine training mode
MODE="${1:-quick}"

case "$MODE" in
    "quick"|"test")
        CONFIG="quick"
        echo "Mode: Quick Test (small model, 1000 steps)"
        ;;
    "tustin"|"full")
        CONFIG="tustin"
        echo "Mode: Full Tustin Training (512d, 6 layers, 100K steps)"
        ;;
    "zoh")
        CONFIG="zoh"
        echo "Mode: ZOH Training (for comparison)"
        ;;
    "custom")
        CONFIG="custom"
        echo "Mode: Custom (edit hyperparameters in train_hyena.py)"
        ;;
    "resume")
        CONFIG="${2:-tustin}"
        RESUME_FLAG="--resume"
        echo "Mode: Resume Training (config: $CONFIG)"
        ;;
    *)
        echo "Usage: bash START_TRAINING.sh [MODE]"
        echo ""
        echo "Available modes:"
        echo "  quick      - Quick test (default)"
        echo "  tustin     - Full Tustin training"
        echo "  zoh        - ZOH training (comparison)"
        echo "  custom     - Custom hyperparameters"
        echo "  resume     - Resume from checkpoint"
        echo ""
        echo "Examples:"
        echo "  bash START_TRAINING.sh              # Quick test"
        echo "  bash START_TRAINING.sh tustin       # Full training"
        echo "  bash START_TRAINING.sh resume       # Resume training"
        exit 1
        ;;
esac

echo "Config: $CONFIG"
echo ""

# Run training with automatic setup
echo "=========================================================================="
echo "Starting Training (dataset will auto-download if needed)"
echo "=========================================================================="
echo ""

python3 train_with_auto_download.py --config "$CONFIG" $RESUME_FLAG

echo ""
echo "=========================================================================="
echo "Training Complete!"
echo "=========================================================================="
