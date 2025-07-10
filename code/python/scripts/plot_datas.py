"""
Unified 1D/2D Electronic Spectroscopy Data Plotting Script

Usage:
    # Auto mode (latest data)
    python plot_datas.py

    # Load specific files
    python plot_datas.py --rel_path "relative/path/to/data/filename(WITHOUT_SUFFIX).npz"

    # Load from directory
    python plot_datas.py --latest_from DIR
"""

import sys
import argparse
from qspectro2d.data import (
    load_latest_data_from_directory,
    load_data_from_rel_path,
)
from qspectro2d.visualization import plot_1d_data, plot_2d_data


def main():
    parser = argparse.ArgumentParser(
        description="Plot 1D or 2D electronic spectroscopy data"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--rel_path", type=str, help="Specific file path (relative to DATA_DIR)"
    )
    group.add_argument("--latest_from", type=str, help="Load latest from subdirectory")

    args = parser.parse_args()

    plot_config = {
        "plot_time_domain": True,
        "plot_frequency_domain": True,
        "extend_for": (1, 15),
        "spectral_components_to_plot": ["abs", "real", "imag"],
        "section": (1, 2, 1, 2),
        # "section": (1.5, 1.7, 1.5, 1.7),
    }

    try:
        # =============================
        # LOAD DATA
        # =============================
        if args.rel_path:
            print(f"📁 Loading specific file: {args.rel_path}")
            data_dict = load_data_from_rel_path(relative_path=args.rel_path)
        else:
            subdir = args.latest_from if args.latest_from else "2d_spectroscopy"
            print(f"🔍 Auto-mode: Loading latest from {subdir}...")
            data_dict = load_latest_data_from_directory(subdir)

        ndim = data_dict["data"].ndim
        print(
            f"✅ Data shape: {data_dict['data'].shape}, Time range: {data_dict['axes']['axis1'][0]:.1f} to {data_dict['axes']['axis1'][-1]:.1f} fs"
        )

        # =============================
        # PLOT
        # =============================
        if ndim == 1:
            plot_1d_data(data_dict, plot_config)
        elif ndim == 2:
            plot_2d_data(data_dict, plot_config)
        else:
            print(f"❌ Unsupported data dimension: {ndim}")
            sys.exit(1)

        print("✅ Plotting completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
