"""
Configuration for HyenaDNA → Mamba Fine-tuning
==============================================
Optimized for NVIDIA RTX 5090 (32GB GDDR7)
Comparison study: Tustin vs ZOH Mamba blocks with pre-trained HyenaDNA embeddings
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HyenaFineTuneConfig:
    """Configuration for fine-tuning HyenaDNA with Mamba blocks."""

    # Model Architecture
    vocab_size: int = 12  # HyenaDNA uses 12 tokens (4 bases + special tokens)
    d_model: int = 256    # Match HyenaDNA-medium-160k
    n_layers: int = 4     # Number of Mamba blocks to use
    d_state: int = 16     # Mamba SSM state dimension
    d_conv: int = 4       # Mamba convolution width
    expand: int = 2       # Mamba expansion factor
    mode: str = "tustin"  # "tustin" or "zoh" for comparison

    # Pre-trained Model
    pretrained_model: str = "LongSafari/hyenadna-medium-160k-seqlen"
    load_embeddings: bool = True
    load_layer_norms: bool = True

    # Training Strategy (2-Phase) - Optimized for RTX 5090
    learning_rate: float = 1e-4
    batch_size: int = 32  # Larger batch for RTX 5090 (32GB VRAM)
    seq_len: int = 2048   # Longer sequences for better context
    max_steps: int = 50000  # ~100M tokens at batch_size=32, seq_len=2048
    warmup_steps: int = 2000

    # Phase 1: Linear Probing (freeze embeddings)
    phase1_steps: int = 10000  # First 20% of training
    freeze_embeddings_phase1: bool = True

    # Phase 2: Full Fine-tuning (unfreeze all)
    phase2_steps: int = 40000  # Remaining 80%

    # Optimization
    use_bfloat16: bool = True
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 1

    # Evaluation & Checkpointing
    eval_interval: int = 200  # Evaluate every 200 steps
    eval_iters: int = 20
    save_interval: int = 200  # Save checkpoint every 200 steps
    log_interval: int = 100

    # Data Loading
    dataset_name: str = "LongSafari/hyenadna-genomic-benchmark"
    num_workers: int = 4
    prefetch_batches: int = 2
    max_tokens: int = 100_000_000  # 100M tokens

    # Directories
    checkpoint_dir: str = "./checkpoints/hyena_mamba"
    results_dir: str = "./results/hyena_training"
    cache_dir: str = "./data/hyenadna"


# Preset Configurations - Optimized for RTX 5090

TUSTIN_CONFIG = HyenaFineTuneConfig(
    mode="tustin",
    d_model=512,      # Larger model for RTX 5090
    n_layers=6,
    batch_size=32,
    seq_len=4096,     # Long context (fits in 32GB)
    max_steps=100000,
    phase1_steps=20000,
    phase2_steps=80000,
    checkpoint_dir="./checkpoints/hyena_mamba_tustin",
    results_dir="./results/tustin_comparison"
)

ZOH_CONFIG = HyenaFineTuneConfig(
    mode="zoh",
    d_model=512,      # Larger model for RTX 5090
    n_layers=6,
    batch_size=32,
    seq_len=4096,     # Long context (fits in 32GB)
    max_steps=100000,
    phase1_steps=20000,
    phase2_steps=80000,
    checkpoint_dir="./checkpoints/hyena_mamba_zoh",
    results_dir="./results/zoh_comparison"
)

# For quick validation (small model, fast iteration)
QUICK_CONFIG = HyenaFineTuneConfig(
    d_model=128,
    n_layers=2,
    batch_size=16,
    seq_len=1024,
    max_steps=1000,
    phase1_steps=200,
    phase2_steps=800,
    eval_interval=100,
    save_interval=200,  # Save every 200 steps
    max_tokens=1_000_000,  # 1M tokens for quick test
    checkpoint_dir="./checkpoints/hyena_mamba_quick",
    results_dir="./results/quick_test"
)

# Large-scale configuration (pushes RTX 5090 to limits)
LARGE_CONFIG = HyenaFineTuneConfig(
    mode="tustin",
    d_model=1024,     # Very large model
    n_layers=12,
    batch_size=16,    # Reduced batch for memory
    seq_len=8192,     # Very long context
    max_steps=200000,
    phase1_steps=40000,
    phase2_steps=160000,
    learning_rate=5e-5,  # Lower LR for stability
    checkpoint_dir="./checkpoints/hyena_mamba_large",
    results_dir="./results/large_scale"
)
