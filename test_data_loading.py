#!/usr/bin/env python
"""Test script to diagnose data loading issues."""

import traceback
from hyena_data import HyenaDNALoader
from config_hyena import QUICK_CONFIG

print("Testing data loading without prefetch...")
print("=" * 60)

try:
    loader = HyenaDNALoader(QUICK_CONFIG, enable_prefetch=False)
    print("✓ Dataset loaded successfully")
    print(f"✓ Train split size: {len(loader.dataset['train'])}")
    print(f"✓ Dataset keys: {loader.dataset.keys()}")

    # Try to get first sample
    first_sample = loader.dataset['train'][0]
    print(f"✓ First sample keys: {list(first_sample.keys())}")
    print(f"✓ First sample preview: {str(first_sample)[:200]}")

    # Try to generate a batch
    print("\nTesting batch generation...")
    x, y, mask = loader.get_batch('train')
    print(f"✓ Batch generated successfully")
    print(f"  x shape: {x.shape}, dtype: {x.dtype}")
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    print(f"  mask shape: {mask.shape}, dtype: {mask.dtype}")

except Exception as e:
    print("✗ ERROR during data loading:")
    print("-" * 60)
    traceback.print_exc()
    print("-" * 60)
