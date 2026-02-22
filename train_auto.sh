#!/bin/bash
#
# Auto-detect and run training (GPU if available, CPU as fallback)
#

set -e

echo "=========================================="
echo "HyenaDNA Training - Auto GPU/CPU Detection"
echo "=========================================="
echo ""

# Try GPU first
echo "Testing GPU availability..."
if python -c "import jax; jax.devices()" 2>&1 | grep -q "gpu\|cuda"; then
    echo "✓ GPU detected, attempting GPU training..."
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

    # Test if GPU actually works
    if python -c "import jax; import jax.numpy as jnp; x = jnp.array([1])" 2>/dev/null; then
        echo "✓ GPU working! Running training on GPU..."
        echo ""
        python train_hyena.py "$@"
        exit 0
    else
        echo "⚠ GPU detected but cuDNN initialization failed"
        echo "  This is a known issue with RTX 5090 and JAX 0.9.x"
        echo "  Falling back to CPU..."
        echo ""
    fi
fi

# Fallback to CPU
echo "Running training on CPU..."
echo "  (This will be slower but works reliably)"
echo ""
export JAX_PLATFORMS=cpu
python train_hyena.py "$@"
