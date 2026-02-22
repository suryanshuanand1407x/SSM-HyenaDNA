"""
HyenaDNA Data Loader
==================
DNA sequence loading and tokenization for genomic language modeling.

Features:
- Multi-core CPU data loading
- Prefetching to overlap with GPU computation
- Thread-safe batch generation
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
from datasets import load_dataset, Dataset
from config_hyena import HyenaFineTuneConfig
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time


class HyenaDNALoader:
    """
    Data loader for DNA sequences with HyenaDNA tokenization.

    Provides the get_batch(split) interface required by mamba_optim.py.
    """

    def __init__(self, config: HyenaFineTuneConfig, enable_prefetch: bool = True):
        """
        Initialize the data loader.

        Args:
            config: Fine-tuning configuration
            enable_prefetch: Enable multi-core prefetching (recommended for GPU training)
        """
        self.config = config
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size

        # DNA tokenization: Matches HyenaDNA convention
        # Standard bases: A=0, C=1, G=2, T=3
        # Special tokens: N=4 (unknown), PAD=5, ...
        self.base_to_token = {
            'A': 0, 'a': 0,
            'C': 1, 'c': 1,
            'G': 2, 'g': 2,
            'T': 3, 't': 3,
            'N': 4, 'n': 4,  # Unknown base
        }
        self.pad_token = 5

        # Load dataset
        print(f"Loading genomic dataset...")
        self.dataset = self._load_genomic_dataset()
        print(f"Dataset loaded: {len(self.dataset['train'])} training sequences")

        # Track position in dataset (for streaming)
        self.train_idx = 0
        self.val_idx = 0
        self.test_idx = 0

        # Multi-core data loading with prefetching
        self.enable_prefetch = enable_prefetch
        self.num_workers = config.num_workers
        self.prefetch_batches = config.prefetch_batches

        if self.enable_prefetch:
            print(f"✓ Prefetching enabled: {self.num_workers} workers, {self.prefetch_batches} batches ahead")
            self._setup_prefetch()
        else:
            print("⚠ Prefetching disabled (single-threaded mode)")

    def _load_genomic_dataset(self):
        """Load and prepare the genomic dataset."""
        import sys

        # Try multiple real genomic datasets in order of preference
        datasets_to_try = [
            # Primary: HyenaDNA benchmark
            ("LongSafari/hyenadna-genomic-benchmark", {}),
            # Fallback 1: Human genome reference
            ("LongSafari/human-genome-hg38", {}),
            # Fallback 2: Nucleotide transformer human reference
            ("InstaDeepAI/human_reference_genome", {}),
            # Fallback 3: Standard genomic benchmark
            ("rajpurkar/genomic_benchmarks", {"name": "human_nontata_promoters"}),
        ]

        for dataset_name, kwargs in datasets_to_try:
            try:
                print(f"Attempting to load: {dataset_name}...")
                dataset = load_dataset(
                    dataset_name,
                    cache_dir=self.config.cache_dir,
                    **kwargs
                )
                print(f"✓ Successfully loaded: {dataset_name}")
                return dataset
            except Exception as e:
                print(f"  Failed to load {dataset_name}: {str(e)[:100]}")
                continue

        # If all real datasets fail, STOP and inform user
        print("\n" + "=" * 70)
        print("ERROR: Could not load any real genomic datasets!")
        print("=" * 70)
        print("\nTried the following datasets:")
        for dataset_name, _ in datasets_to_try:
            print(f"  - {dataset_name}")
        print("\nPossible solutions:")
        print("  1. Check your internet connection")
        print("  2. Install datasets library: pip install --upgrade datasets")
        print("  3. Manually download a genomic dataset")
        print("  4. Set HF_TOKEN environment variable if using private datasets")
        print("\nREFUSING TO USE SYNTHETIC DATA (as requested)")
        print("=" * 70)
        sys.exit(1)

    def _create_synthetic_dataset(self):
        """Create synthetic DNA sequences for testing."""
        np.random.seed(42)

        def generate_dna_sequences(n_sequences: int, seq_len: int):
            sequences = []
            bases = ['A', 'C', 'G', 'T']
            for _ in range(n_sequences):
                seq = ''.join(np.random.choice(bases, size=seq_len))
                sequences.append({'sequence': seq})
            return sequences

        train_data = generate_dna_sequences(1000, self.seq_len * 2)
        val_data = generate_dna_sequences(100, self.seq_len * 2)
        test_data = generate_dna_sequences(100, self.seq_len * 2)

        return {
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        }

    def tokenize_dna(self, sequence: str) -> np.ndarray:
        """
        Tokenize DNA sequence to integer array.

        Args:
            sequence: DNA string (e.g., "ACGTACGT")

        Returns:
            tokens: Integer array (A=0, C=1, G=2, T=3, N=4, PAD=5)
        """
        tokens = []
        for base in sequence:
            token = self.base_to_token.get(base, self.pad_token)
            tokens.append(token)
        return np.array(tokens, dtype=np.int32)

    def get_batch(self, split: str = 'train') -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get a batch of DNA sequences for training/evaluation.

        Required interface for mamba_optim.py functions.

        Uses multi-core prefetching if enabled (recommended for GPU training),
        otherwise falls back to synchronous generation.

        Args:
            split: 'train', 'validation', or 'test'

        Returns:
            x: (B, L) int32 input tokens
            y: (B, L) int32 target tokens (shifted by 1)
            mask: (B, L) float32 loss mask (1=valid, 0=padding)
        """
        # Use prefetching if enabled
        if self.enable_prefetch:
            return self.get_batch_prefetch(split)

        # Otherwise, synchronous generation (original implementation)
        x_batch, y_batch, mask_batch = self._generate_batch_sync(split)

        # Convert to JAX arrays
        return (
            jnp.array(x_batch, dtype=jnp.int32),
            jnp.array(y_batch, dtype=jnp.int32),
            jnp.array(mask_batch, dtype=jnp.float32)
        )

    def reset_indices(self):
        """Reset dataset indices (e.g., for new epoch)."""
        self.train_idx = 0
        self.val_idx = 0
        self.test_idx = 0

    def _setup_prefetch(self):
        """Setup multi-core prefetching infrastructure."""
        # Prefetch queues for each split
        self.prefetch_queues = {
            'train': Queue(maxsize=self.prefetch_batches),
            'validation': Queue(maxsize=self.prefetch_batches),
            'test': Queue(maxsize=self.prefetch_batches)
        }

        # Thread pool for parallel batch generation
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

        # Background threads for each split
        self.prefetch_threads = {}
        self.stop_prefetch = threading.Event()

        for split in ['train', 'validation', 'test']:
            thread = threading.Thread(
                target=self._prefetch_worker,
                args=(split,),
                daemon=True
            )
            thread.start()
            self.prefetch_threads[split] = thread

    def _prefetch_worker(self, split: str):
        """Background worker that continuously generates batches."""
        while not self.stop_prefetch.is_set():
            try:
                # Generate batch (this is CPU-intensive)
                batch = self._generate_batch_sync(split)

                # Put in queue (blocks if queue is full)
                self.prefetch_queues[split].put(batch, timeout=1.0)

            except Exception as e:
                # Log error with full traceback for debugging
                if not self.stop_prefetch.is_set():
                    import traceback
                    print(f"\nWarning: Prefetch error for {split}:")
                    print("-" * 60)
                    traceback.print_exc()
                    print("-" * 60)
                    time.sleep(0.1)

    def _generate_batch_sync(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a single batch synchronously (called by prefetch workers).

        Args:
            split: 'train', 'validation', or 'test'

        Returns:
            x, y, mask as numpy arrays (not JAX arrays yet)
        """
        # Select dataset split
        if split == 'train':
            data = self.dataset['train']
            idx = self.train_idx
        elif split == 'validation':
            data = self.dataset.get('validation', self.dataset.get('valid', self.dataset['train']))
            idx = self.val_idx
        else:
            data = self.dataset.get('test', self.dataset['train'])
            idx = self.test_idx

        # Prepare batch arrays
        x_batch = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        y_batch = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        mask_batch = np.zeros((self.batch_size, self.seq_len), dtype=np.float32)

        # Fill batch
        for b in range(self.batch_size):
            # Get sequence (with wraparound)
            seq_idx = (idx + b) % len(data)
            sample = data[seq_idx]

            # Extract sequence string
            if 'sequence' in sample:
                sequence = sample['sequence']
            elif 'text' in sample:
                sequence = sample['text']
            else:
                # Try the first string field
                sequence = list(sample.values())[0]

            # Tokenize
            tokens = self.tokenize_dna(sequence)

            # Extract window of seq_len + 1 (for causal shift)
            if len(tokens) >= self.seq_len + 1:
                # Random start position for variety
                start = np.random.randint(0, len(tokens) - self.seq_len)
                window = tokens[start:start + self.seq_len + 1]
            else:
                # Pad short sequences
                window = np.pad(
                    tokens,
                    (0, self.seq_len + 1 - len(tokens)),
                    constant_values=self.pad_token
                )

            # Causal language modeling: input = [0:L], target = [1:L+1]
            x_batch[b] = window[:self.seq_len]
            y_batch[b] = window[1:self.seq_len + 1]

            # Mask out padding positions
            mask_batch[b] = (x_batch[b] != self.pad_token).astype(np.float32)

        # Update index
        if split == 'train':
            self.train_idx = (self.train_idx + self.batch_size) % len(data)
        elif split == 'validation':
            self.val_idx = (self.val_idx + self.batch_size) % len(data)
        else:
            self.test_idx = (self.test_idx + self.batch_size) % len(data)

        return x_batch, y_batch, mask_batch

    def get_batch_prefetch(self, split: str = 'train') -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get a pre-generated batch from prefetch queue (fast, no blocking).

        Args:
            split: 'train', 'validation', or 'test'

        Returns:
            x: (B, L) int32 input tokens
            y: (B, L) int32 target tokens
            mask: (B, L) float32 loss mask
        """
        try:
            # Get from queue (blocks until batch is ready)
            x_batch, y_batch, mask_batch = self.prefetch_queues[split].get(timeout=5.0)

            # Convert to JAX arrays (this is fast)
            return (
                jnp.array(x_batch, dtype=jnp.int32),
                jnp.array(y_batch, dtype=jnp.int32),
                jnp.array(mask_batch, dtype=jnp.float32)
            )

        except Exception as e:
            print(f"Warning: Prefetch queue timeout for {split}, falling back to sync")
            # Fallback to synchronous generation
            x, y, mask = self._generate_batch_sync(split)
            return (
                jnp.array(x, dtype=jnp.int32),
                jnp.array(y, dtype=jnp.int32),
                jnp.array(mask, dtype=jnp.float32)
            )

    def shutdown_prefetch(self):
        """Shutdown prefetch threads (call when done training)."""
        if self.enable_prefetch:
            self.stop_prefetch.set()
            # Give threads time to finish
            time.sleep(0.5)
            self.executor.shutdown(wait=False)
            print("✓ Prefetch threads stopped")


# Utility functions
def get_dna_vocab_size() -> int:
    """Return DNA vocabulary size to match HyenaDNA (12 tokens for compatibility)."""
    return 12  # Match HyenaDNA tokenizer for pre-trained weight compatibility


def decode_dna_tokens(tokens: np.ndarray) -> str:
    """
    Convert token array back to DNA string.

    Args:
        tokens: Integer array

    Returns:
        DNA sequence string
    """
    token_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N', 5: '_'}
    return ''.join(token_to_base.get(t, '?') for t in tokens)
