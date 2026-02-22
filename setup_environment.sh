#!/bin/bash
# Environment Setup Script for RTX 5090
# =====================================

set -e  # Exit on error

echo "=========================================="
echo "HyenaDNA → Mamba Setup (RTX 5090)"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on GPU machine
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi not found${NC}"
    echo "This script should be run on a machine with NVIDIA GPU"
    echo "If you're setting up locally (Mac), skip GPU checks with: bash setup_environment.sh --skip-gpu"
    if [[ "$1" != "--skip-gpu" ]]; then
        exit 1
    fi
    echo -e "${YELLOW}⚠ Skipping GPU checks (--skip-gpu flag)${NC}"
    SKIP_GPU=1
fi

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo -e "${RED}✗ Python $REQUIRED_VERSION or higher required${NC}"
    echo "  Current: $PYTHON_VERSION"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check CUDA version (if GPU available)
if [[ -z "$SKIP_GPU" ]]; then
    echo ""
    echo "Checking CUDA installation..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
        echo -e "${GREEN}✓ CUDA $CUDA_VERSION${NC}"
    else
        echo -e "${YELLOW}⚠ nvcc not found in PATH${NC}"
        echo "  Make sure CUDA is installed and in PATH"
    fi

    # Check GPU
    echo ""
    echo "Detecting GPUs..."
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    echo -e "${GREEN}$GPU_INFO${NC}"
fi

# Create conda environment (optional)
echo ""
read -p "Create new conda environment 'hyena'? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating conda environment..."
    conda create -n hyena python=3.10 -y
    echo -e "${GREEN}✓ Conda environment created${NC}"
    echo "Activate with: conda activate hyena"
else
    echo "Skipping conda environment creation"
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."

# Detect if we should install CUDA or CPU version of JAX
if [[ -z "$SKIP_GPU" ]]; then
    echo "Installing JAX with CUDA support..."
    pip install --upgrade pip
    pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "Installing JAX (CPU version)..."
    pip install --upgrade pip
    pip install --upgrade jax jaxlib
fi

echo "Installing other dependencies..."
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p results
mkdir -p data
echo -e "${GREEN}✓ Directories created${NC}"

# Set environment variables
echo ""
echo "Setting up environment variables..."

ENV_FILE=".env"
cat > $ENV_FILE << 'EOF'
# JAX/XLA Configuration for RTX 5090
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=true --xla_gpu_enable_async_collectives=true"

# HuggingFace cache
export HF_HOME=./data/huggingface

# CUDA settings
export TF_CUDNN_USE_AUTOTUNE=1
EOF

echo -e "${GREEN}✓ Environment file created: $ENV_FILE${NC}"
echo "  Source it with: source .env"

# Verify setup
echo ""
echo "Verifying setup..."
if [[ -z "$SKIP_GPU" ]]; then
    python3 setup_gpu.py
else
    echo -e "${YELLOW}Skipping GPU verification (no GPU detected)${NC}"
fi

# Final instructions
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Source environment: source .env"
if [[ ! -z "$SKIP_GPU" ]]; then
    echo "  2. Transfer to GPU machine with: rsync -avz . user@gpu-server:/path/to/p2"
    echo "  3. SSH to GPU machine and run setup again"
else
    echo "  2. Run tests: python test_hyena_data.py"
    echo "  3. Start training:"
    echo "     - Quick test:  python train_hyena.py --config quick"
    echo "     - Full Tustin: python train_hyena.py --config tustin"
    echo "     - Full ZOH:    python train_hyena.py --config zoh"
fi
echo ""
