#!/bin/bash
#
# Install Required Dependencies for HyenaDNA Training
# ====================================================

set -e

echo "========================================================================"
echo "Installing HyenaDNA Training Dependencies"
echo "========================================================================"
echo ""

# Core dependencies
echo "Installing core packages..."
pip install -q \
    jax \
    flax \
    optax \
    transformers \
    pyfaidx \
    pandas \
    tqdm \
    numpy

echo "✓ Core packages installed"
echo ""

# Optional: JAX with CUDA support (for GPU training)
read -p "Install JAX with CUDA support for GPU training? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing JAX with CUDA 12 support..."
    pip install --upgrade "jax[cuda12]" -q
    echo "✓ JAX with CUDA installed"
fi

echo ""
echo "========================================================================"
echo "✓ Dependencies installed successfully!"
echo "========================================================================"
echo ""
echo "Test installation:"
echo "  python test_dataset_download.py"
echo ""
echo "Start training:"
echo "  bash START_TRAINING.sh"
echo "========================================================================"
