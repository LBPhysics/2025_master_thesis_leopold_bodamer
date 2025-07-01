"""
Unified 1D/2D Electronic Spectroscopy Data Plotting Script

Usage:
    # Auto mode (latest data)
    python plot_spectro_data.py --dim 1
    python plot_spectro_data.py --dim 2

    # Load specific files
    python plot_spectro_data.py --dim 1 --rel-path "relative/path/to/data"

    # Load from directory
    python plot_spectro_data.py --dim 2 --latest-from DIR
"""

import sys
import argparse
from qspectro2d.data import (
    load_latest_data_from_directory,
    load_data_from_rel_path,
)
from qspectro2d.visualization import plot_1d_data, plot_2d_data
from qspectro2d.config.paths import DATA_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Plot 1D or 2D electronic spectroscopy data"
    )

    parser.add_argument(
        "--dim", type=int, choices=[1, 2], required=True, help="Data dimensionality (1 or 2)"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rel-path", type=str, help="Specific file path (relative to DATA_DIR)")
    group.add_argument("--latest-from", type=str, help="Load latest from subdirectory")

    args = parser.parse_args()

    # Plot configs for 1D and 2D
    plot_configs = {
        1: {
            "plot_time_domain": True,
            "plot_frequency_domain": True,
            "extend_for": (1, 100),
            "spectral_components_to_plot": ["abs", "real", "imag"],
        },
        2: {
            "plot_time_domain": True,
            "plot_frequency_domain": True,
            "extend_for": (1, 10),
            "spectral_components_to_plot": ["abs", "real", "imag"],
            "section": (1.5, 1.7, 1.5, 1.7),
        },
    }

    try:
        # =============================
        # LOAD DATA
        # =============================
        if args.rel_path:
            print(f"📁 Loading specific file: {args.rel_path}")
            data_dict = load_data_from_rel_path(relative_path=args.rel_path)
        else:
            subdir = args.latest_from if args.latest_from else f"{args.dim}d_spectroscopy"
            print(f"🔍 Auto-mode: Loading latest from {subdir}...")
            data_dict = load_latest_data_from_directory(subdir)

        print(
            f"✅ Data shape: {data_dict['data'].shape}, Time range: {data_dict['axes']['axs1'][0]:.1f} to {data_dict['axes']['axs1'][-1]:.1f} fs"
        )

        # =============================
        # PLOT
        # =============================
        if args.dim == 1:
            plot_1d_data(data_dict, plot_configs[1])
        else:
            plot_2d_data(data_dict, plot_configs[2])

        print("✅ Plotting completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
