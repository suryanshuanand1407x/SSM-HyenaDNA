#!/bin/bash
# Complete Requirements Installation for RTX 5090
# ==============================================

set -e

echo "=========================================="
echo "Installing Requirements for RTX 5090"
echo "=========================================="
echo ""

# Detect platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="mac"
    echo "⚠️  Warning: Detected macOS - JAX will install CPU version"
    echo "   For GPU training, run this on your RTX 5090 machine"
    echo ""
else
    PLATFORM="unknown"
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install JAX with CUDA support (for Linux/RTX 5090)
if [[ "$PLATFORM" == "linux" ]] && command -v nvidia-smi &> /dev/null; then
    echo "Detected NVIDIA GPU - Installing JAX with CUDA 12 support..."

    # Uninstall existing JAX first
    pip uninstall -y jax jaxlib 2>/dev/null || true

    # Install JAX with CUDA 12
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    echo "✓ JAX with CUDA support installed"
else
    echo "Installing JAX (CPU version)..."
    pip install --upgrade jax jaxlib
    echo "✓ JAX (CPU) installed"

    if [[ "$PLATFORM" == "linux" ]]; then
        echo ""
        echo "⚠️  No NVIDIA GPU detected"
        echo "   If you have a GPU, make sure nvidia-smi works"
    fi
fi

echo ""

# Install remaining dependencies
echo "Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ All dependencies installed"
echo ""

# Verify installation
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

# Check JAX
python3 -c "import jax; print(f'JAX version: {jax.__version__}')"
python3 -c "import jax; print(f'JAX devices: {jax.devices()}')"
python3 -c "import jax; print(f'JAX backend: {jax.default_backend()}')"

echo ""

# Check other key packages
python3 -c "import flax; print(f'Flax version: {flax.__version__}')"
python3 -c "import optax; print(f'Optax version: {optax.__version__}')"
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python3 -c "import datasets; print(f'Datasets version: {datasets.__version__}')"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""

if [[ "$PLATFORM" == "linux" ]] && command -v nvidia-smi &> /dev/null; then
    echo "Next steps:"
    echo "  1. Verify GPU setup: python setup_gpu.py"
    echo "  2. Run tests: python test_hyena_data.py"
    echo "  3. Start training: bash run_experiment.sh quick"
else
    echo "⚠️  CPU version installed"
    echo ""
    echo "For GPU training on RTX 5090:"
    echo "  1. Transfer files to GPU machine"
    echo "  2. Run this script on GPU machine"
    echo "  3. Verify with: python setup_gpu.py"
fi

echo ""
