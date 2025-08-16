"""
Unified 1D/2D Electronic Spectroscopy Data Plotting Script

Usage:
    # Auto mode (latest data)
    python plot_datas.py

    # Load specific files
    python plot_datas.py --abs_path "absolute/path/to/data/filename(WITHOUT_SUFFIX And .npz)"

    # Load from directory
    python plot_datas.py --latest_from DIR
"""

import sys
import argparse
from qspectro2d.utils import (
    load_data_from_abs_path,
)
from qspectro2d.visualization import plot_1d_data, plot_2d_data


def main():
    parser = argparse.ArgumentParser(
        description="Plot 1D or 2D electronic spectroscopy data"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--abs_path", type=str, help="Specific file path (absolute)")
    group.add_argument("--latest_from", type=str, help="Load latest from subdirectory")

    args = parser.parse_args()

    plot_config = {
        # "plot_time_domain": True,
        "plot_frequency_domain": True,
        "extend_for": (1, 10),
        "spectral_components_to_plot": ["abs", "real", "imag"],
        # "section": [(1, 3), (1, 3)],
        # "section": [(1.5, 1.7), (1.5, 1.7)],
    }

    try:
        print(f"📁 Loading specific file: {args.abs_path}")
        data_and_info_dict = load_data_from_abs_path(abs_path=args.abs_path)
        ndim = data_and_info_dict["data"].ndim
        t_det_axis = data_and_info_dict["axes"].get("t_det")
        if t_det_axis is not None:
            print(
                f"✅ Data shape: {data_and_info_dict['data'].shape}, Time range: {t_det_axis[0]:.1f} to {t_det_axis[-1]:.1f} fs"
            )
        # Plot
        if ndim == 1:
            plot_1d_data(data_and_info_dict, plot_config)
        elif ndim == 2:
            plot_2d_data(data_and_info_dict, plot_config)
        else:
            print(f"❌ Unsupported data dimension: {ndim}")
            sys.exit(1)
        print("✅ Plotting completed!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
