"""
Simple script to analyze 2D spectroscopy HDF5 data.
Uses the data directory structure from config/paths.py.
"""

from pathlib import Path
from src.utils.save_and_load import load_2d_spectroscopy_data
from config.paths import DATA_DIR


def analyze_latest_file():
    """Load and analyze the most recent HDF5 file."""
    print("=" * 50)
    print("ANALYZING LATEST HDF5 FILE")
    print("=" * 50)

    # Find all HDF5 files
    all_h5_files = []
    for data_dir in [
        DATA_DIR / "raw" / "2d_spectroscopy",
        DATA_DIR / "processed" / "2d_spectroscopy",
    ]:
        if data_dir.exists():
            all_h5_files.extend(data_dir.glob("*.h5"))

    if not all_h5_files:
        print("No HDF5 files found. Run calc_2D_datas.py first.")
        return

    # Get the newest file
    latest_file = max(all_h5_files, key=lambda f: f.stat().st_mtime)
    print(f"File: {latest_file.name}")

    try:
        # Load the data
        data = load_2d_spectroscopy_data(str(latest_file))

        print(f"Times shape: {data['times'].shape}")
        print(f"Times_T shape: {data['times_T'].shape}")
        print(f"Number of 2D datasets: {len(data['two_d_datas'])}")

        # Show first dataset info
        if data["two_d_datas"] and data["two_d_datas"][0] is not None:
            first_data = data["two_d_datas"][0]
            print(f"First dataset shape: {first_data.shape}")
            print(f"First dataset size: {first_data.nbytes / (1024**2):.2f} MB")

        print(f"System parameters: {len(data['system_params'])} entries")

    except Exception as e:
        print(f"Error loading file: {e}")


def main():
    """Main function."""
    print(f"Data directory: {DATA_DIR}")
    analyze_latest_file()


if __name__ == "__main__":
    main()
