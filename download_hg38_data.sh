#!/bin/bash
#
# Download HyenaDNA HG38 Dataset
# Human reference genome (hg38) data for pretraining
#

set -e  # Exit on error

echo "====================================================================="
echo "Downloading HyenaDNA HG38 Dataset"
echo "====================================================================="
echo ""

# Create data directory
DATA_DIR="./data/hg38"
mkdir -p "$DATA_DIR"

echo "Data directory: $DATA_DIR"
echo ""

# Download FASTA file (human genome reference)
echo "Step 1/3: Downloading hg38.ml.fa.gz (Human Reference Genome) ..."
echo "  Source: https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz"
echo "  Size: ~800 MB compressed, ~3.1 GB uncompressed"
echo ""

if [ -f "$DATA_DIR/hg38.ml.fa" ]; then
    echo "  ✓ hg38.ml.fa already exists, skipping download"
else
    curl -# https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz -o "$DATA_DIR/hg38.ml.fa.gz"

    echo ""
    echo "Step 2/3: Decompressing hg38.ml.fa.gz ..."
    gunzip -f "$DATA_DIR/hg38.ml.fa.gz"
    echo "  ✓ Decompressed to hg38.ml.fa"
fi

echo ""

# Download BED file (sequence intervals)
echo "Step 3/3: Downloading human-sequences.bed (sequence intervals) ..."
echo "  Source: https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed"
echo "  Size: ~35 MB"
echo ""

curl -# https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed -o "$DATA_DIR/human-sequences.bed"

echo ""
echo "====================================================================="
echo "Download Complete!"
echo "====================================================================="
echo ""
echo "Files downloaded to $DATA_DIR:"
ls -lh "$DATA_DIR"
echo ""
echo "Dataset info:"
echo "  - hg38.ml.fa: Human reference genome (24 chromosomes merged)"
echo "  - human-sequences.bed: Sequence intervals (chr, start, end, split)"
echo ""
echo "You can now train with real HyenaDNA data!"
echo "====================================================================="
