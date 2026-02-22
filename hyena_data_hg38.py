"""
HyenaDNA Data Loader - HG38 Human Reference Genome
==================================================
Uses real genomic data from FASTA/BED files (no synthetic data).
Adapted from HyenaDNA official data loader.
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple
from pathlib import Path
from pyfaidx import Fasta
import pandas as pd
from config_hyena import HyenaFineTuneConfig
from random import randrange
from download_hg38 import ensure_dataset_downloaded


class HG38DataLoader:
    """
    Data loader for HG38 human reference genome using FASTA + BED files.

    This is the REAL HyenaDNA dataset - no synthetic data.
    """

    def __init__(self, config: HyenaFineTuneConfig):
        """
        Initialize HG38 data loader.

        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size

        # DNA tokenization: Matches HyenaDNA convention
        # A=0, C=1, G=2, T=3, N=4 (unknown), pad=5
        self.base_to_token = {
            'A': 0, 'a': 0,
            'C': 1, 'c': 1,
            'G': 2, 'g': 2,
            'T': 3, 't': 3,
            'N': 4, 'n': 4,  # Unknown base
        }
        self.pad_token = 5

        # Load HG38 data files
        print(f"Loading HG38 human reference genome...")

        # Determine data directory
        data_dir = Path("./data/hg38")
        if not data_dir.exists():
            data_dir = Path(config.cache_dir) / "hg38"

        # Automatically download dataset if not present
        print(f"Checking dataset at: {data_dir.absolute()}")
        if not ensure_dataset_downloaded(str(data_dir)):
            raise RuntimeError(
                f"Failed to download HG38 dataset to {data_dir}\n"
                f"Please try manual download: bash download_hg38_data.sh"
            )

        self.fasta_file = data_dir / "hg38.ml.fa"
        self.bed_file = data_dir / "human-sequences.bed"

        # Check files exist
        if not self.fasta_file.exists():
            raise FileNotFoundError(
                f"FASTA file not found: {self.fasta_file}\n"
                f"Please run: ./download_hg38_data.sh"
            )
        if not self.bed_file.exists():
            raise FileNotFoundError(
                f"BED file not found: {self.bed_file}\n"
                f"Please run: ./download_hg38_data.sh"
            )

        # Load FASTA file (human genome sequences)
        print(f"  Loading FASTA: {self.fasta_file}")
        self.seqs = Fasta(str(self.fasta_file))
        print(f"  ✓ Loaded {len(self.seqs.keys())} chromosomes")

        # Load BED file (intervals: chr, start, end, split)
        print(f"  Loading BED: {self.bed_file}")
        df_raw = pd.read_csv(
            str(self.bed_file),
            sep='\t',
            names=['chr_name', 'start', 'end', 'split']
        )

        # Split into train/validation/test
        self.train_df = df_raw[df_raw['split'] == 'train'].reset_index(drop=True)
        self.val_df = df_raw[df_raw['split'] == 'valid'].reset_index(drop=True)
        self.test_df = df_raw[df_raw['split'] == 'test'].reset_index(drop=True)

        print(f"  ✓ Train intervals: {len(self.train_df):,}")
        print(f"  ✓ Valid intervals: {len(self.val_df):,}")
        print(f"  ✓ Test intervals: {len(self.test_df):,}")

        # Track position in dataset
        self.train_idx = 0
        self.val_idx = 0
        self.test_idx = 0

        print(f"✓ HG38 dataset loaded successfully\n")

    def extract_sequence(self, chr_name: str, start: int, end: int) -> str:
        """
        Extract DNA sequence from FASTA file.

        Args:
            chr_name: Chromosome name (e.g., 'chr1')
            start: Start position (0-indexed)
            end: End position (exclusive)

        Returns:
            DNA sequence string
        """
        chromosome = self.seqs[chr_name]
        chromosome_length = len(chromosome)

        # Adjust window to fit max_length
        interval_length = end - start

        if interval_length > self.seq_len:
            # If interval too long, take random window
            max_start = end - self.seq_len
            start = randrange(start, max(start + 1, max_start))
            end = start + self.seq_len
        elif interval_length < self.seq_len:
            # If interval too short, extend it
            extra_seq = self.seq_len - interval_length
            extra_left = extra_seq // 2
            extra_right = extra_seq - extra_left
            start = max(0, start - extra_left)
            end = min(chromosome_length, end + extra_right)

        # Extract sequence
        seq = str(chromosome[start:end])

        # Pad if needed (shouldn't happen often)
        if len(seq) < self.seq_len:
            seq = seq + ('N' * (self.seq_len - len(seq)))

        return seq[:self.seq_len]  # Ensure exact length

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

        Args:
            split: 'train', 'validation', or 'test'

        Returns:
            x: (B, L) int32 input tokens
            y: (B, L) int32 target tokens (shifted by 1)
            mask: (B, L) float32 loss mask (1=valid, 0=padding)
        """
        # Select dataset split
        if split == 'train':
            df = self.train_df
            idx = self.train_idx
        elif split == 'validation':
            df = self.val_df
            idx = self.val_idx
        else:
            df = self.test_df
            idx = self.test_idx

        # Prepare batch arrays
        x_batch = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        y_batch = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        mask_batch = np.zeros((self.batch_size, self.seq_len), dtype=np.float32)

        # Fill batch
        for b in range(self.batch_size):
            # Get interval (with wraparound)
            row_idx = (idx + b) % len(df)
            row = df.iloc[row_idx]
            chr_name, start, end = row['chr_name'], row['start'], row['end']

            # Extract sequence from FASTA
            sequence = self.extract_sequence(chr_name, start, end)

            # Tokenize
            tokens = self.tokenize_dna(sequence)

            # Causal language modeling: input = [0:L], target = [1:L+1]
            # We need L+1 tokens total
            if len(tokens) >= self.seq_len + 1:
                # Random start for variety
                offset = randrange(0, len(tokens) - self.seq_len)
                window = tokens[offset:offset + self.seq_len + 1]
            else:
                # Pad if needed
                window = np.pad(
                    tokens,
                    (0, self.seq_len + 1 - len(tokens)),
                    constant_values=self.pad_token
                )

            x_batch[b] = window[:self.seq_len]
            y_batch[b] = window[1:self.seq_len + 1]

            # Mask out padding positions
            mask_batch[b] = (x_batch[b] != self.pad_token).astype(np.float32)

        # Update index
        if split == 'train':
            self.train_idx = (self.train_idx + self.batch_size) % len(df)
        elif split == 'validation':
            self.val_idx = (self.val_idx + self.batch_size) % len(df)
        else:
            self.test_idx = (self.test_idx + self.batch_size) % len(df)

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
