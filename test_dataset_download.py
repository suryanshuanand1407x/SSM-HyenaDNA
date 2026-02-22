"""
Test script to verify HG38 dataset download and data loading
"""

from download_hg38 import ensure_dataset_downloaded, verify_dataset
from pathlib import Path


def test_download():
    """Test automatic dataset download."""
    print("=" * 70)
    print("Testing HG38 Dataset Download")
    print("=" * 70)

    data_dir = "./data/hg38"

    # Ensure dataset is downloaded
    success = ensure_dataset_downloaded(data_dir)

    if success:
        print("\n✓ Dataset download test PASSED")

        # Verify dataset
        print("\nVerifying dataset integrity...")
        if verify_dataset(data_dir):
            print("✓ Dataset verification PASSED")

            # Show file info
            data_path = Path(data_dir)
            fasta_file = data_path / "hg38.ml.fa"
            bed_file = data_path / "human-sequences.bed"

            print("\nDataset files:")
            for file in [fasta_file, bed_file]:
                if file.exists():
                    size_gb = file.stat().st_size / (1024 ** 3)
                    print(f"  {file.name}: {size_gb:.2f} GB")
        else:
            print("✗ Dataset verification FAILED")
            return False
    else:
        print("\n✗ Dataset download test FAILED")
        return False

    return True


def test_data_loader():
    """Test HG38DataLoader with downloaded dataset."""
    print("\n" + "=" * 70)
    print("Testing HG38DataLoader")
    print("=" * 70)

    try:
        from config_hyena import QUICK_CONFIG
        from hyena_data_hg38 import HG38DataLoader

        print("\nInitializing data loader...")
        loader = HG38DataLoader(QUICK_CONFIG)

        print("\nTesting batch generation...")
        x, y, mask = loader.get_batch('train')

        print(f"✓ Batch shapes:")
        print(f"  x (input):  {x.shape} dtype={x.dtype}")
        print(f"  y (target): {y.shape} dtype={y.dtype}")
        print(f"  mask:       {mask.shape} dtype={mask.dtype}")

        print(f"\n✓ Token statistics:")
        print(f"  Min token: {x.min()}")
        print(f"  Max token: {x.max()}")
        print(f"  Unique tokens: {len(set(x.flatten().tolist()))}")

        print("\n✓ Data loader test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Data loader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nHyenaDNA HG38 Dataset Test Suite")
    print("=" * 70)

    # Test 1: Download
    download_ok = test_download()

    # Test 2: Data loader (only if download succeeded)
    if download_ok:
        loader_ok = test_data_loader()
    else:
        loader_ok = False

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"  Dataset Download: {'✓ PASS' if download_ok else '✗ FAIL'}")
    print(f"  Data Loader:      {'✓ PASS' if loader_ok else '✗ FAIL'}")
    print("=" * 70)

    if download_ok and loader_ok:
        print("\n✓ All tests passed! Ready for training.")
        exit(0)
    else:
        print("\n✗ Some tests failed. Please check errors above.")
        exit(1)
