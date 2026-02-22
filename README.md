# HyenaDNA → Mamba Fine-tuning (Tustin vs ZOH Comparison)

**Optimized for NVIDIA RTX 5090**

## Overview

Fine-tune pre-trained **HyenaDNA** genomic language model by replacing Hyena operators with **Mamba blocks** to compare **Tustin** vs **ZOH** discretization methods on genomic sequences.

### Architecture
```
┌─────────────────────────────────────┐
│   HyenaDNA Token Embeddings         │  ← Pre-trained (frozen in Phase 1)
├─────────────────────────────────────┤
│   Mamba Block 1 (Tustin or ZOH)    │  ← Random init, trainable
│   Mamba Block 2                     │
│   ...                               │
│   Mamba Block N                     │
├─────────────────────────────────────┤
│   Output Head → DNA predictions     │
└─────────────────────────────────────┘
```

### Two-Phase Training
1. **Phase 1 (20%)**: Freeze embeddings, train Mamba blocks only (linear probing)
2. **Phase 2 (80%)**: Unfreeze all parameters (domain adaptation)

## Quick Start

### 1. Installation (on RTX 5090 machine)

```bash
# Install dependencies
bash install_requirements.sh

# Verify GPU
python setup_gpu.py
```

📖 **Detailed install guide**: [INSTALL.md](INSTALL.md)

### 2. Validation

```bash
# Run tests
python test_hyena_data.py

# Quick training test (5 min)
bash run_experiment.sh quick
```

### 3. Full Training & Comparison

```bash
# Train Tustin (3-4 hours)
bash run_experiment.sh tustin

# Train ZOH (3-4 hours)
bash run_experiment.sh zoh

# Compare results
bash run_experiment.sh compare
```

📖 **Quick start guide**: [QUICKSTART.md](QUICKSTART.md)

## Documentation

| Document | Description |
|----------|-------------|
| **[QUICKSTART.md](QUICKSTART.md)** | 5-minute setup and training guide |
| **[INSTALL.md](INSTALL.md)** | Detailed installation instructions |
| **[README_HYENA.md](README_HYENA.md)** | Comprehensive technical documentation |
| **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** | Implementation details and architecture |

## Project Structure

```
p2/
├── README.md                    # This file
├── QUICKSTART.md               # Quick start guide
├── INSTALL.md                  # Installation guide
├── README_HYENA.md             # Full documentation
├── IMPLEMENTATION_SUMMARY.md   # Technical details
│
├── Core Implementation
│   ├── config_hyena.py         # Configuration presets
│   ├── hyena_data.py          # DNA data loader
│   ├── model_hybrid.py        # HyenaDNA + Mamba hybrid
│   ├── train_hyena.py         # Training script
│   └── checkpoint_utils.py    # Checkpoint management
│
├── Mamba Core (existing)
│   ├── mamba_core.py          # Mamba architecture (Tustin/ZOH)
│   ├── mamba_optim.py         # Training infrastructure
│   ├── mamba_metrics.py       # Metrics
│   └── mamba_viz.py           # Visualization
│
├── Utilities
│   ├── setup_gpu.py           # GPU verification
│   ├── monitor_training.py    # Real-time monitoring
│   ├── compare_results.py     # Result comparison
│   └── test_hyena_data.py     # Unit tests
│
└── Scripts
    ├── install_requirements.sh    # Dependency installation
    ├── setup_environment.sh       # Environment setup
    └── run_experiment.sh          # Experiment runner
```

## Configuration Presets

| Config | Model | Seq Len | Batch | Time | Memory | Purpose |
|--------|-------|---------|-------|------|--------|---------|
| **quick** | 128d×2 | 1024 | 16 | 5m | 4GB | Validation |
| **tustin** | 512d×6 | 4096 | 32 | 3-4h | 20GB | Comparison |
| **zoh** | 512d×6 | 4096 | 32 | 3-4h | 20GB | Baseline |
| **large** | 1024d×12 | 8192 | 16 | 8-12h | 30GB | Max scale |

## Features

### GPU Optimizations
- ✅ **bfloat16 precision** - Tensor Core acceleration
- ✅ **XLA fusion** - Single-kernel forward+backward
- ✅ **Dynamic memory** - Efficient VRAM usage
- ✅ **Gradient accumulation** - Handle large batches

### Robustness
- ✅ **Atomic checkpoints** - No corruption on crash
- ✅ **Auto-resume** - Continue from latest checkpoint
- ✅ **Phase tracking** - Automatic phase transitions
- ✅ **Gradient clipping** - Training stability
- ✅ **Checkpoint every 200 steps** - Detailed training history
- ✅ **Metrics in checkpoints** - Train/val loss + accuracy saved

### Monitoring & Analysis
- ✅ **Real-time monitoring** - GPU/progress tracking
- ✅ **Automatic comparison** - Tustin vs ZOH analysis
- ✅ **Visualization** - Training curves and plots
- ✅ **Paper-ready metrics** - Publication-quality results

## Expected Results

### Performance Targets
- **Validation Loss**: < 1.0 (perplexity < 2.7)
- **Accuracy**: > 80% (vs 25% random baseline)
- **GPU Utilization**: > 95%
- **Throughput**: ~50K tokens/sec (TUSTIN/ZOH config)

### Comparison Metrics
- Loss reduction: Tustin vs ZOH
- Training stability: Gradient norms
- Convergence speed: Steps to target accuracy
- Genomic performance: Biological plausibility

## Usage Examples

### Basic Training
```bash
# Tustin training
python train_hyena.py --config tustin

# ZOH training
python train_hyena.py --config zoh

# Resume from checkpoint
python train_hyena.py --config tustin --resume
```

### Monitoring
```bash
# Real-time monitoring (separate terminal)
python monitor_training.py --config tustin --interval 10

# Check GPU
watch -n 1 nvidia-smi

# View logs
tail -f logs/tustin_*.log
```

### Analysis
```bash
# Plot training curves
python plot_training_metrics.py --checkpoint_dir ./checkpoints/hyena_mamba_tustin

# Compare Tustin vs ZOH
python compare_results.py

# View results
cat results/comparison/comparison_report.txt
open results/comparison/comparison_plot.png
```

### Metrics Tracking
Every checkpoint (saved every 200 steps) includes:
- Training loss & accuracy
- Validation loss & accuracy
- Automatic CSV logging for analysis

See **[CHECKPOINT_METRICS.md](CHECKPOINT_METRICS.md)** for details.

## Requirements

- **Hardware**: NVIDIA RTX 5090 (or any CUDA GPU with 16GB+ VRAM)
- **Software**:
  - CUDA 12.0+
  - Python 3.10+
  - 20GB disk space

**Full dependency list**: [requirements.txt](requirements.txt)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `batch_size` or `seq_len` in config |
| CUDA not found | Install CUDA 12+ and run `install_requirements.sh` |
| Slow training | Check GPU utilization (should be >95%) |
| Loss → NaN | Lower learning rate, check data |

See [INSTALL.md](INSTALL.md) for detailed troubleshooting.

## Development

### Running Tests
```bash
# All tests
python test_hyena_data.py

# GPU verification
python setup_gpu.py

# Single test
python -c "from test_hyena_data import test_dna_tokenization; test_dna_tokenization()"
```

### Adding New Configs
Edit `config_hyena.py`:
```python
CUSTOM_CONFIG = HyenaFineTuneConfig(
    mode="tustin",
    d_model=768,
    n_layers=8,
    seq_len=6144,
    batch_size=24,
    # ... other params
)
```

## Citation

If you use this code, please cite:

```bibtex
@article{your_paper,
  title={Tustin vs ZOH Discretization for Genomic Mamba Models},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

Also cite:
- **Mamba**: Gu & Dao (2023) - Mamba: Linear-Time Sequence Modeling
- **HyenaDNA**: Nguyen et al. (2023) - HyenaDNA: Long-Range Genomic Sequence Modeling

## License

MIT License (or your chosen license)

## Contact

For questions or issues:
- Open a GitHub issue
- Email: [your-email]

---

**Status**: ✅ **Ready for RTX 5090 Training**

**Time to first results**: ~4 hours (one config)
**Time to full comparison**: ~6-8 hours (both configs)

🚀 **Get started**: [QUICKSTART.md](QUICKSTART.md)
