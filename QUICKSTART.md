# Quick Start Guide - RTX 5090

**Goal**: Fine-tune HyenaDNA with Mamba blocks and compare Tustin vs ZOH

## 🚀 5-Minute Setup (On RTX 5090 Machine)

```bash
# 1. Clone/transfer project
cd /path/to/p2

# 2. Setup environment
bash setup_environment.sh

# 3. Activate environment
source .env
conda activate hyena  # if using conda

# 4. Verify GPU
python setup_gpu.py
```

Expected output:
```
✓ Found 1 GPU(s)
✓ bfloat16 computation works
✓ All checks passed!
```

## ✅ Validation (5 minutes)

```bash
# Run tests
python test_hyena_data.py

# Quick training test (1000 steps)
python train_hyena.py --config quick
```

## 🔬 Full Comparison Study

### Option 1: Sequential Training (~6-8 hours total)

```bash
# Train Tustin (3-4 hours)
python train_hyena.py --config tustin

# Train ZOH (3-4 hours)
python train_hyena.py --config zoh

# Compare results
python compare_results.py
```

### Option 2: Parallel Training (if you have 2x RTX 5090)

```bash
# Terminal 1: Tustin on GPU 0
CUDA_VISIBLE_DEVICES=0 python train_hyena.py --config tustin

# Terminal 2: ZOH on GPU 1
CUDA_VISIBLE_DEVICES=1 python train_hyena.py --config zoh

# After both finish:
python compare_results.py
```

### Option 3: Background Training with Monitoring

```bash
# Start training in background
nohup python train_hyena.py --config tustin > logs/tustin.log 2>&1 &

# Monitor in another terminal
python monitor_training.py --config tustin --interval 10

# Check logs
tail -f logs/tustin.log
```

## 📊 Expected Results

| Config | Model Size | Seq Len | Time | Memory | Val Loss |
|--------|-----------|---------|------|--------|----------|
| QUICK  | 2.5M     | 1024    | 5m   | 4GB    | ~1.5     |
| TUSTIN | 50M      | 4096    | 3-4h | 20GB   | <1.0     |
| ZOH    | 50M      | 4096    | 3-4h | 20GB   | <1.0     |

**Target Performance**:
- Validation perplexity < 2.0
- Next-nucleotide accuracy > 80%
- Either Tustin or ZOH should show clear advantage

## 🔍 Monitoring Training

### Real-time Monitor
```bash
python monitor_training.py --config tustin
```

### Manual Checks
```bash
# GPU utilization (should be ~95-100%)
nvidia-smi -l 1

# Latest checkpoint
ls -lht checkpoints/hyena_mamba_tustin/

# Training logs
tail -f logs/tustin.log
```

## 📈 Analyzing Results

### After Training Completes

```bash
# Compare Tustin vs ZOH
python compare_results.py

# Results saved to:
# - results/comparison/comparison_report.txt
# - results/comparison/comparison_plot.png
```

### Load Checkpoint for Inference

```python
import pickle

# Load checkpoint
with open('checkpoints/hyena_mamba_tustin/checkpoint_00100000.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

params = checkpoint['params']
step = checkpoint['step']

print(f"Loaded checkpoint from step {step}")
```

## 🐛 Troubleshooting

### Out of Memory

```python
# Edit config_hyena.py
batch_size = 16  # Reduce from 32
seq_len = 2048   # Reduce from 4096
```

### Slow Training

- Check GPU utilization: `nvidia-smi` (should be >95%)
- Check bfloat16 is enabled: `use_bfloat16=True` in config
- Enable XLA optimizations: `source .env`

### Training Diverges (Loss → NaN)

```python
# Edit config_hyena.py
learning_rate = 5e-5  # Reduce from 1e-4
gradient_clip_norm = 0.5  # Reduce from 1.0
```

### HuggingFace Download Fails

```bash
# Manually download
huggingface-cli login  # if private model
huggingface-cli download LongSafari/hyenadna-medium-160k-seqlen

# Or use mirror
export HF_ENDPOINT=https://hf-mirror.com
```

## 💾 Checkpoints & Storage

### Disk Space Requirements

- Checkpoints: ~2GB per checkpoint × 3 kept = ~6GB
- Dataset cache: ~5-10GB (HuggingFace)
- Results: ~100MB
- **Total**: ~15-20GB per experiment

### Checkpoint Management

```bash
# Keep only last 3 checkpoints (automatic)
# Controlled by keep_last_n=3 in checkpoint_utils.py

# Manually clean old checkpoints
rm checkpoints/hyena_mamba_tustin/checkpoint_000*.pkl

# Backup important checkpoints
cp checkpoints/hyena_mamba_tustin/checkpoint_00100000.pkl backups/
```

## 📝 Paper-Ready Results

### Generate Comparison Table

```bash
python compare_results.py
cat results/comparison/comparison_report.txt
```

### Generate Plots

```python
# All plots saved automatically:
# - results/comparison/comparison_plot.png
# - results/tustin_comparison/training_curves.png (if implemented)
# - results/zoh_comparison/training_curves.png
```

## 🚨 Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch_size or seq_len |
| `JAX compilation slow` | Normal for first run, cached after |
| `Loss is NaN` | Lower learning rate, check data |
| `GPU at 0%` | Check code is actually running |
| `Checkpoint not found` | Check checkpoint_dir path |

## 🎯 Next Steps After Training

1. **Compare Results**: `python compare_results.py`
2. **Fine-tune Hyperparameters**: Adjust learning rate, batch size
3. **Scale Up**: Try LARGE_CONFIG for better accuracy
4. **Evaluate on Benchmarks**: Add downstream tasks
5. **Write Paper**: Use comparison_report.txt for results section

## 📞 Need Help?

- Check README_HYENA.md for detailed documentation
- Run `python setup_gpu.py` to verify setup
- Check logs in `logs/` directory
- Verify GPU with `nvidia-smi`

---

**Ready to train! 🚀**

Estimated time to first results: **~4 hours** (TUSTIN or ZOH config)
