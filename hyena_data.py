"""
HyenaDNA Data Loader
==================
DNA sequence loading and tokenization for genomic language modeling.
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
from datasets import load_dataset, Dataset
from config_hyena import HyenaFineTuneConfig


class HyenaDNALoader:
    """
    Data loader for DNA sequences with HyenaDNA tokenization.

    Provides the get_batch(split) interface required by mamba_optim.py.
    """

    def __init__(self, config: HyenaFineTuneConfig):
        """
        Initialize the data loader.

        Args:
            config: Fine-tuning configuration
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

    def _load_genomic_dataset(self):
        """Load and prepare the genomic dataset."""
        try:
            # Try loading HyenaDNA benchmark dataset
            dataset = load_dataset(
                self.config.dataset_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            return dataset
        except:
            # Fallback: Use human genome from HuggingFace
            print("Using human genome reference dataset...")
            try:
                dataset = load_dataset(
                    "LongSafari/human-genome-hg38",
                    cache_dir=self.config.cache_dir,
                    streaming=True
                )
                return dataset
            except:
                # Last fallback: Create synthetic dataset for testing
                print("WARNING: Using synthetic DNA data for testing")
                return self._create_synthetic_dataset()

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

        Args:
            split: 'train', 'validation', or 'test'

        Returns:
            x: (B, L) int32 input tokens
            y: (B, L) int32 target tokens (shifted by 1)
            mask: (B, L) float32 loss mask (1=valid, 0=padding)
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

        # Convert to JAX arrays (optimized for CUDA)
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
