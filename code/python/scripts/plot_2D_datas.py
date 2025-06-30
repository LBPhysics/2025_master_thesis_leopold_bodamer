"""
2D Electronic Spectroscopy Data Plotting Script

This script provides a unified interface for plotting 2D electronic spectroscopy data
using the standardized data structure.

Usage:
    # Auto mode (latest from 2d_spectroscopy)
    python plot_2D_datas.py

    # Load specific files
    python plot_2D_datas.py --data-path "path/to/data.npz" --info-path "path/to/info.pkl"

    # Load from directory
    python plot_2D_datas.py --base-dir DIR
    python plot_2D_datas.py --latest-from DIR
"""

import sys
import argparse
from pathlib import Path

# Modern imports from reorganized package structure
from qspectro2d.data import (
    load_latest_data_from_directory,
    load_data_from_paths,
)
from qspectro2d.data.files import generate_base_sub_dir

from qspectro2d.visualization import plot_2d_data
from config.paths import DATA_DIR, FIGURES_PYTHON_DIR


def main():
    """Main function to run the 2D spectroscopy plotting."""
    # Simple plot config
    plot_config = {
        "plot_time_domain": True,  # TODO CHANGE TO True
        "plot_frequency_domain": False,
        "extend_for": (1, 10),
        "spectral_components_to_plot": ["abs", "real", "imag"],
        "section": (1.5, 1.7, 1.5, 1.7),  # (x_min, x_max, y_min, y_max)
    }

    parser = argparse.ArgumentParser(
        description="Plot 2D electronic spectroscopy data"
    )  # Input options

    # Input options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--data-path", type=str, help="Specific data file path (relative to DATA_DIR)"
    )

    # Info path (required when using --data-path)
    parser.add_argument(
        "--info-path",
        type=str,
        help="Specific info file path (required with --data-path)",
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

        else:
            # Default: load latest from 2d_spectroscopy
            print("🔍 Auto-mode: Loading latest from 2d_spectroscopy...")
            data_dict = load_latest_data_from_directory(Path("2d_spectroscopy"))

        # =============================
        # EXTRACT DATA AND PLOT
        # =============================
        print(
            f"✅ Data: {data_dict['data'].shape}, Time: {data_dict['axes']['axs1'][0]:.1f} to {data_dict['axes']['axs1'][-1]:.1f} fs"
        )

        # Output directory
        data_config = data_dict["data_config"]
        system = data_dict["system"]
        sub_dir = generate_base_sub_dir(data_config, system)
        output_dir = FIGURES_PYTHON_DIR / sub_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📊 Plotting to: {output_dir}")

        # plot_2d_data handles all plotting and saving automatically
        plot_2d_data(
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
