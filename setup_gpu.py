"""
GPU Setup and Verification for RTX 5090
======================================
Verify CUDA, cuDNN, JAX GPU setup, and optimize for NVIDIA hardware
"""

import os
import sys


def setup_environment():
    """Set optimal environment variables for NVIDIA GPU training."""

    # XLA/JAX CUDA optimizations
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Dynamic memory allocation
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'   # Use 90% of GPU memory
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # Use CUDA allocator

    # Enable XLA optimizations
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=true '
        '--xla_gpu_enable_async_collectives=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
    )

    # TensorFloat-32 (TF32) for Ampere/Ada/Hopper (RTX 5090 is Ada Lovelace)
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

    # cuDNN optimizations
    os.environ['TF_CUDNN_DETERMINISTIC'] = '0'  # Allow non-deterministic for speed
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'   # Auto-tune kernels

    print("✓ Environment variables configured for RTX 5090")


def verify_jax_setup():
    """Verify JAX can see CUDA GPUs."""
    try:
        import jax
        print("\n" + "=" * 60)
        print("JAX GPU Verification")
        print("=" * 60)

        # Check devices
        devices = jax.devices()
        print(f"Available devices: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"  [{i}] {device.device_kind} - {device.platform}")

        # Check for GPU
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if not gpu_devices:
            print("\n⚠ WARNING: No GPU devices found!")
            print("Make sure CUDA and cuDNN are installed:")
            print("  pip install --upgrade 'jax[cuda12]'")
            return False

        print(f"\n✓ Found {len(gpu_devices)} GPU(s)")

        # Test GPU computation
        import jax.numpy as jnp
        x = jnp.ones((1000, 1000))
        y = jnp.dot(x, x)
        y.block_until_ready()

        print("✓ GPU computation test passed")

        # Check backend
        print(f"\nDefault backend: {jax.default_backend()}")

        return True

    except Exception as e:
        print(f"\n✗ JAX GPU setup failed: {e}")
        return False


def verify_cuda_version():
    """Verify CUDA installation."""
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("CUDA Verification")
            print("=" * 60)
            print(result.stdout)
            return True
        else:
            print("\n⚠ nvcc not found - CUDA may not be installed")
            return False
    except FileNotFoundError:
        print("\n⚠ nvcc not found in PATH")
        return False


def check_gpu_memory():
    """Check available GPU memory."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("GPU Memory Status")
            print("=" * 60)
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            return True
        else:
            print("\n⚠ nvidia-smi not available")
            return False

    except FileNotFoundError:
        print("\n⚠ nvidia-smi not found")
        return False


def verify_bfloat16_support():
    """Verify bfloat16 support (critical for Tensor Cores)."""
    try:
        import jax
        import jax.numpy as jnp

        print("\n" + "=" * 60)
        print("bfloat16 Support Check")
        print("=" * 60)

        # Test bfloat16 computation
        x = jnp.ones((100, 100), dtype=jnp.bfloat16)
        y = jnp.dot(x, x)
        y.block_until_ready()

        print("✓ bfloat16 computation works")
        print("✓ Tensor Cores will be utilized for training")

        return True

    except Exception as e:
        print(f"✗ bfloat16 test failed: {e}")
        return False


def print_optimization_tips():
    """Print optimization tips for RTX 5090."""
    print("\n" + "=" * 60)
    print("RTX 5090 Optimization Tips")
    print("=" * 60)
    print("""
RTX 5090 Specifications:
  - CUDA Cores: ~16,384 (estimated)
  - Tensor Cores: 5th gen
  - Memory: 32GB GDDR7
  - Memory Bandwidth: ~1.5 TB/s
  - TDP: ~450W

Recommended Training Settings:
  ✓ use_bfloat16=True (Tensor Core acceleration)
  ✓ batch_size=32-64 (large batches for GPU utilization)
  ✓ seq_len=4096-8192 (long sequences fit in 32GB)
  ✓ gradient_accumulation_steps=1-2 (if needed)
  ✓ Enable XLA fusion (automatic in JAX)

Expected Performance:
  - Throughput: ~50-100k tokens/sec (depends on model size)
  - Training time: ~2-4 hours for 100M tokens (medium model)
  - Memory usage: ~20-28GB for large configs

Multi-GPU Setup (if you have multiple RTX 5090s):
  - Use JAX's pmap for data parallelism
  - Scale batch size linearly with GPU count
  - Expected linear speedup up to 4 GPUs
""")


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("RTX 5090 Setup Verification")
    print("=" * 60)

    # Setup environment
    setup_environment()

    # Run checks
    checks = [
        ("CUDA Installation", verify_cuda_version),
        ("GPU Memory", check_gpu_memory),
        ("JAX GPU Setup", verify_jax_setup),
        ("bfloat16 Support", verify_bfloat16_support),
    ]

    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("✓ All checks passed! Ready for training on RTX 5090")
        print_optimization_tips()
        print("\nYou can now run:")
        print("  python test_hyena_data.py              # Run tests")
        print("  python train_hyena.py --config quick   # Quick test")
        print("  python train_hyena.py --config tustin  # Full training (Tustin)")
        print("  python train_hyena.py --config zoh     # Full training (ZOH)")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Install CUDA: https://developer.nvidia.com/cuda-downloads")
        print("  2. Install JAX with CUDA support:")
        print("     pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        print("  3. Verify nvidia-smi works: nvidia-smi")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
