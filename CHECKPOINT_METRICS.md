# Checkpoint & Metrics Tracking System

## Overview

Your fine-tuning now **saves checkpoints every 200 steps** with complete training metrics (train/val loss and accuracy). This allows you to:

- Track training progress in detail
- Resume from any checkpoint
- Analyze training curves
- Identify overfitting early
- Compare different runs

---

## What Gets Saved

### 1. **Checkpoint Files** (every 200 steps)

Location: `checkpoints/<config_name>/checkpoint_XXXXXXXX.pkl`

Each checkpoint contains:
```python
{
    'step': 200,                    # Training step
    'params': {...},                # Model weights (bfloat16)
    'opt_state': {...},             # Optimizer state (Adam momentum, etc.)
    'config': HyenaFineTuneConfig,  # Full configuration
    'metrics': {
        'train_loss': 1.234,        # Training loss
        'train_acc': 0.456,         # Training accuracy
        'val_loss': 1.345,          # Validation loss
        'val_acc': 0.423            # Validation accuracy
    }
}
```

### 2. **Metrics CSV** (cumulative log)

Location: `checkpoints/<config_name>/training_metrics.csv`

Format:
```csv
step,train_loss,train_acc,val_loss,val_acc
200,1.234,0.456,1.345,0.423
400,1.123,0.478,1.256,0.445
600,1.089,0.492,1.198,0.461
...
```

This CSV file is **perfect for plotting** and analyzing training curves.

---

## Checkpoint Management

### Automatic Cleanup

- **Keeps last 20 checkpoints** by default (= 4000 steps of history)
- Automatically deletes older checkpoints to save disk space
- For a 100K step training run, you'll keep checkpoints from step 96,000 onward

### Disk Space

- **Per checkpoint**: ~500MB - 2GB (depending on model size)
- **20 checkpoints**: ~10-40GB total
- **CSV file**: <1MB (negligible)

### Modify Retention

Edit `train_hyena.py` to keep more/fewer checkpoints:

```python
save_checkpoint(
    state, step + 1, checkpoint_dir,
    config=config, metrics=metrics_dict,
    keep_last_n=50  # Keep 50 checkpoints instead of 20
)
```

---

## Usage Examples

### 1. **Train with Metrics Tracking**

```bash
# Start training (automatically saves every 200 steps)
python train_hyena.py --config tustin

# Resume from latest checkpoint
python train_hyena.py --config tustin --resume
```

### 2. **Plot Training Curves**

```bash
# Interactive plot (opens window)
python plot_training_metrics.py --checkpoint_dir ./checkpoints/hyena_mamba_tustin

# Save to file
python plot_training_metrics.py \
    --checkpoint_dir ./checkpoints/hyena_mamba_tustin \
    --save ./results/tustin_training_curves.png
```

### 3. **Load Specific Checkpoint**

```python
import pickle

# Load checkpoint
with open('checkpoints/hyena_mamba_tustin/checkpoint_00010000.pkl', 'rb') as f:
    ckpt = pickle.load(f)

# Access metrics
print(f"Step: {ckpt['step']}")
print(f"Train Loss: {ckpt['metrics']['train_loss']:.4f}")
print(f"Train Acc:  {ckpt['metrics']['train_acc']*100:.2f}%")
print(f"Val Loss:   {ckpt['metrics']['val_loss']:.4f}")
print(f"Val Acc:    {ckpt['metrics']['val_acc']*100:.2f}%")

# Access model weights
params = ckpt['params']
```

### 4. **Analyze Metrics CSV**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv('checkpoints/hyena_mamba_tustin/training_metrics.csv')

# Find best validation loss
best_val_step = df.loc[df['val_loss'].idxmin(), 'step']
best_val_loss = df['val_loss'].min()
print(f"Best val loss: {best_val_loss:.4f} at step {best_val_step}")

# Plot custom curves
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['train_loss'], label='Train Loss')
plt.plot(df['step'], df['val_loss'], label='Val Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.savefig('my_custom_plot.png')
```

### 5. **Compare Tustin vs ZOH**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load both runs
tustin_df = pd.read_csv('checkpoints/hyena_mamba_tustin/training_metrics.csv')
zoh_df = pd.read_csv('checkpoints/hyena_mamba_zoh/training_metrics.csv')

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(tustin_df['step'], tustin_df['val_loss'], label='Tustin', linewidth=2)
plt.plot(zoh_df['step'], zoh_df['val_loss'], label='ZOH', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('Tustin vs ZOH: Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(tustin_df['step'], tustin_df['val_acc']*100, label='Tustin', linewidth=2)
plt.plot(zoh_df['step'], zoh_df['val_acc']*100, label='ZOH', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.title('Tustin vs ZOH: Validation Accuracy')

plt.tight_layout()
plt.savefig('tustin_vs_zoh_comparison.png', dpi=300)
```

---

## Configuration

### Update Save Interval

All configs now save every **200 steps** by default. To change:

**Option 1: Edit `config_hyena.py`**
```python
TUSTIN_CONFIG = HyenaFineTuneConfig(
    ...
    save_interval=500,  # Save every 500 steps instead
    eval_interval=500,  # Should match save_interval for metrics
    ...
)
```

**Option 2: Edit hyperparameters in `train_hyena.py`**
```python
SAVE_INTERVAL = 200   # Change this value
EVAL_INTERVAL = 200   # Should match SAVE_INTERVAL
```

---

## Monitoring During Training

### Real-time Progress

Training loop displays:
```
Phase 2: 100%|██████████| 80000/80000 [2:34:12<00:00, 8.62it/s, loss=0.8234]

--- Evaluation at step 10000 ---
Train Loss: 0.8234
Val Loss:   0.8567

--- Computing metrics for checkpoint at step 10000 ---
✓ Checkpoint saved: checkpoints/hyena_mamba_tustin/checkpoint_00010000.pkl
  Metrics: train_loss=0.8234, train_acc=0.6543, val_loss=0.8567, val_acc=0.6321
  Removed old checkpoint: checkpoint_00008000.pkl
```

### Check Latest Metrics

```bash
# View last 10 checkpoints
tail -10 checkpoints/hyena_mamba_tustin/training_metrics.csv

# Watch metrics in real-time during training
watch -n 5 'tail -1 checkpoints/hyena_mamba_tustin/training_metrics.csv'
```

---

## Expected Results

### QUICK_CONFIG (1000 steps)
- **Checkpoints**: 5 total (steps 200, 400, 600, 800, 1000)
- **Training time**: ~5 minutes
- **Disk space**: ~2GB

### TUSTIN_CONFIG / ZOH_CONFIG (100K steps)
- **Checkpoints**: 20 kept (steps 96,200 to 100,000)
- **CSV records**: 500 rows (one per 200 steps)
- **Training time**: ~3-4 hours
- **Disk space**: ~20-30GB

### LARGE_CONFIG (200K steps)
- **Checkpoints**: 20 kept (steps 196,200 to 200,000)
- **CSV records**: 1000 rows
- **Training time**: ~8-12 hours
- **Disk space**: ~40-50GB

---

## Troubleshooting

### Issue: "Out of disk space"

**Solution**: Reduce `keep_last_n` or increase `save_interval`

```python
# Option 1: Keep fewer checkpoints
keep_last_n=10  # Only keep 10 checkpoints

# Option 2: Save less frequently
save_interval=500  # Save every 500 steps instead of 200
```

### Issue: "Metrics CSV is empty"

**Cause**: Training hasn't reached first checkpoint yet (200 steps)

**Solution**: Wait until step 200, then check again

### Issue: "Want to save ALL checkpoints (no cleanup)"

```python
# In train_hyena.py, set:
keep_last_n=99999  # Keep all checkpoints (WARNING: disk space!)
```

---

## Best Practices

### 1. **Always Plot After Training**
```bash
python plot_training_metrics.py --checkpoint_dir ./checkpoints/hyena_mamba_tustin
```

### 2. **Save Plots for Paper**
```bash
python plot_training_metrics.py \
    --checkpoint_dir ./checkpoints/hyena_mamba_tustin \
    --save ./paper_figures/tustin_training.png
```

### 3. **Back Up Important Checkpoints**
```bash
# Backup best checkpoint
mkdir -p backups
cp checkpoints/hyena_mamba_tustin/checkpoint_00045600.pkl backups/best_tustin.pkl
```

### 4. **Export Metrics for Analysis**
```bash
# Copy CSV to results directory
cp checkpoints/hyena_mamba_tustin/training_metrics.csv \
   results/tustin_metrics.csv
```

---

## Summary

✅ **Checkpoints saved every 200 steps**
✅ **Complete metrics in every checkpoint** (train/val loss + accuracy)
✅ **CSV log for easy plotting**
✅ **Automatic old checkpoint cleanup** (keeps last 20)
✅ **Resume training from any checkpoint**
✅ **Visualization script included** (`plot_training_metrics.py`)

**Result**: You have full visibility into your training process and can analyze performance at every stage!
