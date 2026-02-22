#!/bin/bash
# Automated Experiment Runner for RTX 5090
# =========================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
EXPERIMENT=${1:-"quick"}
RESUME=${2:-""}

echo -e "${BLUE}=========================================="
echo "HyenaDNA → Mamba Experiment Runner"
echo -e "==========================================${NC}"
echo ""

# Validate experiment type
if [[ ! "$EXPERIMENT" =~ ^(quick|tustin|zoh|large|compare)$ ]]; then
    echo -e "${RED}Invalid experiment: $EXPERIMENT${NC}"
    echo ""
    echo "Usage: bash run_experiment.sh [EXPERIMENT] [--resume]"
    echo ""
    echo "Experiments:"
    echo "  quick    - Quick validation (5 min)"
    echo "  tustin   - Full Tustin training (3-4 hours)"
    echo "  zoh      - Full ZOH training (3-4 hours)"
    echo "  large    - Large-scale training (8-12 hours)"
    echo "  compare  - Compare Tustin vs ZOH results"
    echo ""
    echo "Examples:"
    echo "  bash run_experiment.sh quick"
    echo "  bash run_experiment.sh tustin"
    echo "  bash run_experiment.sh tustin --resume"
    echo "  bash run_experiment.sh compare"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Comparison mode
if [[ "$EXPERIMENT" == "compare" ]]; then
    echo -e "${BLUE}Running comparison analysis...${NC}"
    echo ""

    # Check if both models are trained
    if [[ ! -d "checkpoints/hyena_mamba_tustin" ]]; then
        echo -e "${RED}✗ Tustin model not found${NC}"
        echo "Train it first: bash run_experiment.sh tustin"
        exit 1
    fi

    if [[ ! -d "checkpoints/hyena_mamba_zoh" ]]; then
        echo -e "${RED}✗ ZOH model not found${NC}"
        echo "Train it first: bash run_experiment.sh zoh"
        exit 1
    fi

    python compare_results.py

    echo ""
    echo -e "${GREEN}✓ Comparison complete!${NC}"
    echo "Results saved to: results/comparison/"
    echo ""
    echo "View report:"
    echo "  cat results/comparison/comparison_report.txt"
    echo ""
    echo "View plot:"
    echo "  open results/comparison/comparison_plot.png"
    exit 0
fi

# Source environment
if [[ -f ".env" ]]; then
    echo -e "${YELLOW}Loading environment variables...${NC}"
    source .env
    echo -e "${GREEN}✓ Environment loaded${NC}"
    echo ""
fi

# Check GPU
echo "Checking GPU availability..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo -e "${RED}✗ nvidia-smi not found${NC}"
    echo "Make sure you're running on a machine with NVIDIA GPU"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo -e "${GREEN}✓ GPU detected: $GPU_INFO${NC}"
echo ""

# Training mode
LOG_FILE="logs/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}Starting ${EXPERIMENT} training...${NC}"
echo "Log file: $LOG_FILE"
echo ""

# Build command
CMD="python train_hyena.py --config $EXPERIMENT"

if [[ "$RESUME" == "--resume" ]]; then
    CMD="$CMD --resume"
    echo -e "${YELLOW}Resuming from checkpoint${NC}"
fi

# Print configuration
echo "Configuration:"
case $EXPERIMENT in
    quick)
        echo "  - Model: 128d × 2 layers"
        echo "  - Sequence: 1024 tokens"
        echo "  - Steps: 1000"
        echo "  - Expected time: ~5 minutes"
        echo "  - Memory: ~4GB"
        ;;
    tustin)
        echo "  - Model: 512d × 6 layers"
        echo "  - Sequence: 4096 tokens"
        echo "  - Steps: 100,000"
        echo "  - Expected time: ~3-4 hours"
        echo "  - Memory: ~20GB"
        ;;
    zoh)
        echo "  - Model: 512d × 6 layers"
        echo "  - Sequence: 4096 tokens"
        echo "  - Steps: 100,000"
        echo "  - Expected time: ~3-4 hours"
        echo "  - Memory: ~20GB"
        ;;
    large)
        echo "  - Model: 1024d × 12 layers"
        echo "  - Sequence: 8192 tokens"
        echo "  - Steps: 200,000"
        echo "  - Expected time: ~8-12 hours"
        echo "  - Memory: ~30GB"
        ;;
esac
echo ""

# Ask for confirmation (except for quick)
if [[ "$EXPERIMENT" != "quick" ]]; then
    read -p "Continue? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Show monitoring tip
echo ""
echo -e "${YELLOW}Tip: Monitor training in another terminal:${NC}"
echo "  python monitor_training.py --config $EXPERIMENT"
echo ""
echo -e "${GREEN}Starting training...${NC}"
echo ""

# Run training
$CMD 2>&1 | tee $LOG_FILE

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "✓ Training complete!"
    echo -e "==========================================${NC}"
    echo ""
    echo "Results:"
    echo "  - Checkpoints: checkpoints/hyena_mamba_${EXPERIMENT}/"
    echo "  - Metrics: results/${EXPERIMENT}*/final_metrics.txt"
    echo "  - Log: $LOG_FILE"
    echo ""

    if [[ "$EXPERIMENT" == "tustin" ]] || [[ "$EXPERIMENT" == "zoh" ]]; then
        echo "Next steps:"
        if [[ "$EXPERIMENT" == "tustin" ]]; then
            echo "  1. Train ZOH: bash run_experiment.sh zoh"
        else
            echo "  1. Train Tustin: bash run_experiment.sh tustin"
        fi
        echo "  2. Compare results: bash run_experiment.sh compare"
    fi

    echo ""
else
    echo ""
    echo -e "${RED}=========================================="
    echo "✗ Training failed!"
    echo -e "==========================================${NC}"
    echo ""
    echo "Check the log file: $LOG_FILE"
    echo ""
    echo "Common issues:"
    echo "  - Out of memory: Reduce batch_size or seq_len in config_hyena.py"
    echo "  - CUDA errors: Check GPU with nvidia-smi"
    echo "  - Data errors: Check internet connection for HuggingFace download"
    echo ""
    exit 1
fi
