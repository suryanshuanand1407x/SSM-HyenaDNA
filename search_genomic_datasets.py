#!/usr/bin/env python
"""Search for available genomic datasets on HuggingFace."""

from datasets import load_dataset
import sys

# List of known genomic datasets to test
test_datasets = [
    # DNA/Genomic datasets
    ("kuleshov-group/med-halt", {}),
    ("sagawa/BIOSSES", {}),
    ("tattabio/gUTR_Brendel", {}),
    ("multimolecule/rna-atlas", {}),
    ("multimolecule/trna", {}),
    ("gustavhartz/human_genome_test_split", {}),
    ("nucleotide_transformer/human_reference", {}),
    ("vivym/midrc", {}),
]

print("Searching for accessible genomic datasets...")
print("=" * 70)

working_datasets = []

for dataset_name, kwargs in test_datasets:
    try:
        print(f"\nTrying: {dataset_name}...")
        ds = load_dataset(dataset_name, split="train", streaming=True, **kwargs)
        # Try to get first example
        first = next(iter(ds))
        print(f"  ✓ SUCCESS! Keys: {list(first.keys())}")
        working_datasets.append(dataset_name)
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:80]}")

print("\n" + "=" * 70)
print(f"Found {len(working_datasets)} working datasets:")
for ds in working_datasets:
    print(f"  - {ds}")
