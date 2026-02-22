# HyenaDNA → Mamba Fine-tuning

**Comparison Study: Tustin vs ZOH Mamba Blocks**

Optimized for NVIDIA RTX 5090 (32GB GDDR7)

## Overview

This project fine-tunes the pre-trained **HyenaDNA** genomic language model by replacing Hyena operators with **Mamba blocks** (Tustin or ZOH discretization). The goal is to compare the performance of different Mamba variants on genomic sequence modeling.

### Architecture

```
HyenaDNA Embeddings (pre-trained)
         ↓
   Mamba Blocks (random init, Tustin or ZOH)
         ↓
   Output Head (pre-trained or random)
```

### Two-Phase Fine-tuning Strategy

1. **Phase 1 (20% of training)**: Freeze embeddings, train only Mamba blocks
   - Prevents noisy gradients from destroying pre-trained DNA representations
   - Linear probing to learn state-space dynamics

2. **Phase 2 (80% of training)**: Unfreeze all parameters
   - Full fine-tuning for domain adaptation
   - Embeddings adjust to better serve Mamba logic

## Quick Start (RTX 5090)

### 1. SSH Setup

```bash
# Connect to your RTX 5090 machine
ssh user@your-gpu-server

# Navigate to project
cd /path/to/p2
```

### 2. Environment Setup

```bash
# Create conda environment
conda create -n hyena python=3.10
conda activate hyena

# Install JAX with CUDA support (for RTX 5090)
pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Verify JAX can see GPU
python setup_gpu.py
```

Expected output:
```
✓ Found 1 GPU(s)
✓ GPU computation test passed
✓ bfloat16 computation works
✓ All checks passed! Ready for training on RTX 5090
```

### 4. Run Tests

```bash
# Unit tests (validates data pipeline and model)
python test_hyena_data.py
```

### 5. Start Training

```bash
# Quick validation (1K steps, ~5 minutes)
python train_hyena.py --config quick

# Full Tustin training (~2-4 hours)
python train_hyena.py --config tustin

# Full ZOH training (for comparison)
python train_hyena.py --config zoh

# Resume from checkpoint
python train_hyena.py --config tustin --resume
```

## Configuration Presets

### QUICK_CONFIG
- **Purpose**: Fast validation
- **Model**: 128d × 2 layers
- **Sequence**: 1024 tokens
- **Time**: ~5 minutes on RTX 5090
- **Memory**: ~4GB VRAM

### TUSTIN_CONFIG
- **Purpose**: Production comparison study
- **Model**: 512d × 6 layers (~50M params)
- **Sequence**: 4096 tokens (long context)
- **Batch**: 32
- **Steps**: 100K (~3-4 hours)
- **Memory**: ~18-22GB VRAM

### ZOH_CONFIG
- **Purpose**: Baseline comparison
- **Model**: 512d × 6 layers
- **Sequence**: 4096 tokens
- **Batch**: 32
- **Steps**: 100K (~3-4 hours)
- **Memory**: ~18-22GB VRAM

### LARGE_CONFIG
- **Purpose**: Push RTX 5090 to limits
- **Model**: 1024d × 12 layers (~200M params)
- **Sequence**: 8192 tokens (very long)
- **Batch**: 16
- **Steps**: 200K (~8-12 hours)
- **Memory**: ~28-30GB VRAM

## Expected Performance

### RTX 5090 Benchmarks

| Config | Throughput | Memory | Training Time |
|--------|-----------|---------|---------------|
| QUICK  | ~80K tok/s | 4GB     | 5 min         |
| TUSTIN | ~50K tok/s | 20GB    | 3-4 hours     |
| ZOH    | ~50K tok/s | 20GB    | 3-4 hours     |
| LARGE  | ~25K tok/s | 30GB    | 8-12 hours    |

### Validation Metrics (Target)

- **Perplexity**: < 2.0 (good model)
- **Accuracy**: > 80% (vs 25% random baseline)
- **Val Loss**: < 1.0

## Comparison Study Workflow

### 1. Train Both Variants

```bash
# Terminal 1: Train Tustin
python train_hyena.py --config tustin

# Terminal 2: Train ZOH (parallel if you have 2x RTX 5090)
python train_hyena.py --config zoh
```

### 2. Monitor Training

Checkpoints saved every 2K steps:
```
checkpoints/hyena_mamba_tustin/checkpoint_00002000.pkl
checkpoints/hyena_mamba_tustin/checkpoint_00004000.pkl
...
```

Logs saved to:
```
results/tustin_comparison/final_metrics.txt
results/zoh_comparison/final_metrics.txt
```

### 3. Compare Results

```python
# Load final metrics
with open('results/tustin_comparison/final_metrics.txt') as f:
    tustin_metrics = f.read()

with open('results/zoh_comparison/final_metrics.txt') as f:
    zoh_metrics = f.read()

# Compare validation loss, perplexity, accuracy
```

## File Structure

```
p2/
├── mamba_core.py           # Mamba model (Tustin/ZOH blocks)
├── mamba_optim.py          # Training infrastructure
├── mamba_metrics.py        # Metrics computation
├── mamba_viz.py            # Visualization
│
├── config_hyena.py         # Configuration presets
├── hyena_data.py           # DNA data loader
├── model_hybrid.py         # HyenaDNA + Mamba hybrid
├── checkpoint_utils.py     # Checkpoint management
├── train_hyena.py          # Main training script
│
├── test_hyena_data.py      # Unit tests
├── setup_gpu.py            # GPU verification
└── README_HYENA.md         # This file

checkpoints/                # Training checkpoints
├── hyena_mamba_tustin/
├── hyena_mamba_zoh/
└── hyena_mamba_quick/

results/                    # Training results
├── tustin_comparison/
├── zoh_comparison/
└── quick_test/

data/                       # Dataset cache
└── hyenadna/              # HuggingFace cache
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python train_hyena.py --config tustin  # Edit config_hyena.py: batch_size=16

# Or reduce sequence length
# Edit config_hyena.py: seq_len=2048
```

### Slow Training

- **Check GPU utilization**: `nvidia-smi -l 1`
  - Should be ~95-100% during training
- **Enable bfloat16**: Already enabled in configs (Tensor Core acceleration)
- **Check data loading**: Should not be bottleneck with prefetching

### CUDA Errors

```bash
# Reinstall JAX with correct CUDA version
pip uninstall jax jaxlib
pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Check CUDA version
nvcc --version
```

### HuggingFace Download Issues

```bash
# Set HF cache directory
export HF_HOME=/path/to/large/disk/huggingface

# Or manually download model
huggingface-cli download LongSafari/hyenadna-medium-160k-seqlen
```

## Advanced Usage

### Multi-GPU Training

```python
# Edit train_hyena.py to use JAX pmap
# Scale batch_size linearly with GPU count

# Example for 2x RTX 5090:
batch_size = 64  # 32 per GPU
```

### Custom Dataset

```python
# Edit hyena_data.py
def _load_genomic_dataset(self):
    dataset = load_dataset("your/custom/dataset")
    return dataset
```

### Adjust Phase Split

```python
# Edit config_hyena.py
phase1_steps = 30000  # 30% freeze
phase2_steps = 70000  # 70% full finetune
```

## Citation

If you use this code, please cite:

- **Mamba**: Gu & Dao (2023) - Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **HyenaDNA**: Nguyen et al. (2023) - HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution
- **Tustin Discretization**: Your paper (when published)

## License

MIT License (or your chosen license)

## Contact

For issues or questions, please open a GitHub issue or contact [your email].

---

**Ready for RTX 5090 training! 🚀**
