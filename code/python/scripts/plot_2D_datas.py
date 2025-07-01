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
    load_data_from_rel_path,
)
from qspectro2d.visualization import plot_2d_data
from qspectro2d.config.paths import DATA_DIR  # Import the DATA_DIR constant


def main():
    """Main function to run the 2D spectroscopy plotting."""
    print("The data is located in the DATA_DIR directory:", DATA_DIR)
    # Simple plot config
    plot_config = {
        "plot_time_domain": True,  # TODO CHANGE TO True
        "plot_frequency_domain": True,
        "extend_for": (1, 1000),
        "spectral_components_to_plot": ["abs", "real", "imag"],
        "section": (1.5, 1.7, 1.5, 1.7),  # (x_min, x_max, y_min, y_max)
    }

    parser = argparse.ArgumentParser(
        description="Plot 2D electronic spectroscopy data"
    )  # Input options

    # Input options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--rel-path", type=str, help="Specific data file path (relative to DATA_DIR)"
    )

    args = parser.parse_args()

    try:
        # =============================
        # LOAD DATA
        # =============================
        if args.rel_path:
            print(f"📁 Loading specific files:")
            print(f"  Data and Info at: {args.rel_path}")
            data_dict = load_data_from_rel_path(relative_path=args.rel_path)

        else:
            # Default: load latest from 2d_spectroscopy
            print("🔍 Auto-mode: Loading latest from 2d_spectroscopy...")
            data_dict = load_latest_data_from_directory("2d_spectroscopy")

        # =============================
        # EXTRACT DATA AND PLOT
        # =============================
        print(
            f"✅ Data: {data_dict['data'].shape}, Time: {data_dict['axes']['axs1'][0]:.1f} to {data_dict['axes']['axs1'][-1]:.1f} fs"
        )

        # plot_2d_data handles all plotting and saving automatically
        plot_2d_data(
            loaded_data=data_dict,
            plot_config=plot_config,
        )

        print("✅ Plotting completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
