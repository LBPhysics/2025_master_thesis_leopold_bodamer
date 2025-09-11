"""
Unified 1D/2D Electronic Spectroscopy Data Plotting Script

Usage:
    # Load specific files
    python plot_datas.py --abs_path "absolute/path/to/data/filename(WITHOUT_SUFFIX And .npz)"

"""

import sys
import argparse

from qspectro2d.utils import load_data_from_abs_path
from qspectro2d.visualization.plotting_functions import plot_data


def main():
    parser = argparse.ArgumentParser(description="Plot 1D or 2D electronic spectroscopy data")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--abs_path", type=str, help="Specific file path (absolute)")

    args = parser.parse_args()

    plot_config = {
        # "plot_time_domain": True,
        "plot_frequency_domain": True,
        "extend_for": (1, 10),
        "spectral_components_to_plot": ["abs", "real", "img"],
        # "section": [(1, 3), (1, 3)],
        "section": [(1.4, 1.8), (1.4, 1.8)],
    }

    try:
        print(f"📁 Loading specific file: {args.abs_path}")
        loaded = load_data_from_abs_path(abs_path=args.abs_path)
        is_2d = "t_coh" in loaded["axes"]

        if is_2d:
            plot_data(loaded, plot_config, dimension="2d")
        else:
            plot_data(loaded, plot_config, dimension="1d")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
