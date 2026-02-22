"""
Automatic HG38 Dataset Downloader for HyenaDNA
===============================================
Automatically downloads the HG38 human reference genome dataset
if it doesn't exist locally.
"""

import os
import subprocess
import urllib.request
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for URL downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update progress bar.

        Args:
            b: Number of blocks transferred
            bsize: Size of each block (in bytes)
            tsize: Total size (in bytes)
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, description: str = "Downloading"):
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        output_path: Local path to save file
        description: Description for progress bar
    """
    print(f"\n{description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {output_path}")

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=description) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

    print(f"  ✓ Download complete!")


def decompress_gzip(gz_path: str, output_path: str):
    """
    Decompress a gzip file with progress bar.

    Args:
        gz_path: Path to .gz file
        output_path: Path for decompressed output
    """
    print(f"\nDecompressing {gz_path}...")

    # Get file size for progress bar
    file_size = os.path.getsize(gz_path)

    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Decompressing") as pbar:
                # Read in chunks
                chunk_size = 1024 * 1024  # 1 MB chunks
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.update(len(chunk))

    print(f"  ✓ Decompressed to {output_path}")


def download_hg38_dataset(data_dir: str = "./data/hg38", force: bool = False) -> bool:
    """
    Download HG38 dataset (FASTA + BED files) if not already present.

    This function:
    1. Creates the data directory if it doesn't exist
    2. Downloads hg38.ml.fa.gz (human genome reference) from Google Cloud
    3. Decompresses it to hg38.ml.fa
    4. Downloads human-sequences.bed (sequence intervals)
    5. Skips files that already exist (unless force=True)

    Args:
        data_dir: Directory to store dataset (default: ./data/hg38)
        force: Force re-download even if files exist

    Returns:
        success: True if dataset is ready, False if download failed
    """
    print("=" * 70)
    print("HyenaDNA HG38 Dataset Downloader")
    print("=" * 70)

    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"\nData directory: {data_path.absolute()}")

    # Define file paths
    fasta_file = data_path / "hg38.ml.fa"
    fasta_gz_file = data_path / "hg38.ml.fa.gz"
    bed_file = data_path / "human-sequences.bed"

    # URLs
    fasta_url = "https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz"
    bed_url = "https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed"

    try:
        # Step 1: Download and decompress FASTA file
        print("\n" + "-" * 70)
        print("Step 1/2: Human Reference Genome (hg38.ml.fa)")
        print("-" * 70)
        print("  Size: ~800 MB compressed, ~3.1 GB uncompressed")

        if fasta_file.exists() and not force:
            print(f"  ✓ File already exists: {fasta_file}")
            print("  Skipping download")
        else:
            # Download compressed file
            if fasta_gz_file.exists() and not force:
                print(f"  ✓ Compressed file already exists: {fasta_gz_file}")
            else:
                download_file(
                    fasta_url,
                    str(fasta_gz_file),
                    "Downloading hg38.ml.fa.gz"
                )

            # Decompress
            decompress_gzip(str(fasta_gz_file), str(fasta_file))

            # Remove compressed file to save space
            print(f"\nRemoving compressed file to save disk space...")
            fasta_gz_file.unlink()
            print(f"  ✓ Removed {fasta_gz_file}")

        # Step 2: Download BED file
        print("\n" + "-" * 70)
        print("Step 2/2: Sequence Intervals (human-sequences.bed)")
        print("-" * 70)
        print("  Size: ~35 MB")

        if bed_file.exists() and not force:
            print(f"  ✓ File already exists: {bed_file}")
            print("  Skipping download")
        else:
            download_file(
                bed_url,
                str(bed_file),
                "Downloading human-sequences.bed"
            )

        # Verify all files exist
        print("\n" + "=" * 70)
        print("Dataset Download Complete!")
        print("=" * 70)

        print(f"\nDataset location: {data_path.absolute()}")
        print("\nFiles:")
        for file in [fasta_file, bed_file]:
            if file.exists():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  ✓ {file.name}: {size_mb:.1f} MB")
            else:
                print(f"  ✗ {file.name}: MISSING")
                return False

        print("\n" + "=" * 70)
        print("Dataset is ready for training!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nYou can try manual download using:")
        print(f"  bash download_hg38_data.sh")
        return False


def verify_dataset(data_dir: str = "./data/hg38") -> bool:
    """
    Verify that the HG38 dataset exists and is complete.

    Args:
        data_dir: Directory containing dataset

    Returns:
        valid: True if dataset is complete and valid
    """
    data_path = Path(data_dir)

    fasta_file = data_path / "hg38.ml.fa"
    bed_file = data_path / "human-sequences.bed"

    # Check files exist
    if not fasta_file.exists():
        print(f"Missing FASTA file: {fasta_file}")
        return False

    if not bed_file.exists():
        print(f"Missing BED file: {bed_file}")
        return False

    # Check file sizes (basic validation)
    fasta_size = fasta_file.stat().st_size
    bed_size = bed_file.stat().st_size

    # Expected sizes (approximate)
    MIN_FASTA_SIZE = 3_000_000_000  # ~3 GB
    MIN_BED_SIZE = 1_000_000  # ~1 MB (38K sequences)

    if fasta_size < MIN_FASTA_SIZE:
        print(f"FASTA file seems incomplete: {fasta_size} bytes (expected >{MIN_FASTA_SIZE})")
        return False

    if bed_size < MIN_BED_SIZE:
        print(f"BED file seems incomplete: {bed_size} bytes (expected >{MIN_BED_SIZE})")
        return False

    return True


def ensure_dataset_downloaded(data_dir: str = "./data/hg38") -> bool:
    """
    Ensure HG38 dataset is downloaded. Download if missing.

    This is the main function to call from other scripts.

    Args:
        data_dir: Directory to store dataset

    Returns:
        success: True if dataset is ready
    """
    # First, try to verify existing dataset
    if verify_dataset(data_dir):
        print(f"✓ HG38 dataset verified at {data_dir}")
        return True

    # If not valid, download it
    print(f"HG38 dataset not found or incomplete. Downloading...")
    return download_hg38_dataset(data_dir)


if __name__ == "__main__":
    """
    Run this script directly to download the HG38 dataset.

    Usage:
        python download_hg38.py
        python download_hg38.py --force  # Re-download even if exists
    """
    import argparse

    parser = argparse.ArgumentParser(description='Download HyenaDNA HG38 dataset')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/hg38',
        help='Directory to store dataset'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )
    args = parser.parse_args()

    success = download_hg38_dataset(args.data_dir, force=args.force)

    if success:
        print("\n✓ Dataset download successful!")
        exit(0)
    else:
        print("\n❌ Dataset download failed!")
        exit(1)
