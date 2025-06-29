"""
1D Electronic Spectroscopy Data Plotting Script

This script provides a unified interface for plotting 1D electronic spectroscopy data
using the standardized data structure.

Usage:
    # Auto mode (latest from 1d_spectroscopy)
    python plot_1D_datas.py

    # Load specific files
    python plot_1D_datas.py --data-path "path/to/data.npz" --info-path "path/to/info.pkl"

    # Load from directory
    python plot_1D_datas.py --base-dir DIR
    python plot_1D_datas.py --latest-from DIR
"""

import sys
import argparse
from pathlib import Path

# Modern imports from reorganized package structure
import data
from qspectro2d.data import (
    load_all_data_from_directory,
    load_latest_data_from_directory,
    load_data_from_paths,
)
from qspectro2d.data.files import generate_base_sub_dir

from qspectro2d.visualization import plot_1d_data
from config.paths import DATA_DIR, FIGURES_PYTHON_DIR


def main():
    """Main function to run the 1D spectroscopy plotting."""
    parser = argparse.ArgumentParser(
        description="Plot 1D electronic spectroscopy data"
    )  # Input options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--base-dir", type=Path, help="Load newest from all subdirectories"
    )
    group.add_argument(
        "--latest-from", type=Path, help="Load latest single file from directory tree"
    )
    group.add_argument(
        "--data-path", type=str, help="Specific data file path (relative to DATA_DIR)"
    )

    # Info path (required when using --data-path)
    parser.add_argument(
        "--info-path",
        type=str,
        help="Specific info file path (required with --data-path)",
    )

    # Plotting options
    parser.add_argument(
        "--spectral-components",
        nargs="+",
        default=["abs", "real"],
        choices=["real", "imag", "abs", "phase"],
    )

    args = parser.parse_args()

    # Validate arguments
    if args.data_path and not args.info_path:
        print("❌ Error: --info-path is required when using --data-path")
        sys.exit(1)

    try:
        # =============================
        # LOAD DATA
        # =============================
        if args.data_path:
            print(f"📁 Loading specific files:")
            print(f"  Data: {args.data_path}")
            print(f"  Info: {args.info_path}")
            data_dict = load_data_from_paths(
                data_path=DATA_DIR / args.data_path, info_path=DATA_DIR / args.info_path
            )
            file_path = args.data_path

        elif args.base_dir:
            print(f"📁 Loading newest files from: {args.base_dir}")
            all_data = load_all_data_from_directory(args.base_dir)
            if not all_data:
                print("❌ No data files found!")
                sys.exit(1)
            # Use first dataset
            file_path, data_dict = next(iter(all_data.items()))
            print(f"🎨 Using: {file_path}")

        elif args.latest_from:
            print(f"📁 Loading latest from: {args.latest_from}")
            data_dict = load_latest_data_from_directory(args.latest_from)
            file_path = "latest"

        else:
            # Default: load latest from 1d_spectroscopy
            print("🔍 Auto-mode: Loading latest from 1d_spectroscopy...")
            data_dict = load_latest_data_from_directory(Path("1d_spectroscopy"))
            file_path = "auto_latest"

        # =============================
        # EXTRACT DATA AND PLOT
        # =============================
        print(
            f"✅ Data: {data_dict['data'].shape}, Time: {data_dict['axes']['axs1'][0]:.1f} to {data_dict['axes']['axs1'][-1]:.1f} fs"
        )

        # Simple plot config
        plot_config = {
            "plot_time_domain": True,
            "plot_frequency_domain": True,
            "spectral_components_to_plot": args.spectral_components,
            "extend_for": (1, 2),
        }

        # Output directory
        data_config = data_dict["data_config"]
        system = data_dict["system"]
        sub_dir = generate_base_sub_dir(data_config, system)
        output_dir = FIGURES_PYTHON_DIR / sub_dir

        print(f"📊 Plotting to: {output_dir}")

        # plot_1d_data handles all plotting and saving automatically
        plot_1d_data(
            loaded_data=data_dict,
            plot_config=plot_config,
            output_dir=output_dir,
        )

        print("✅ Plotting completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
