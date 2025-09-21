"""
Unified 1D/2D Electronic Spectroscopy Data Plotting Script

Usage:
    # Load specific files
    python plot_datas.py --abs_path "absolute/path/to/data/filename(WITHOUT_SUFFIX And .npz)"

"""

import sys
import argparse
import numpy as np
import warnings
from qspectro2d.utils import load_simulation_data
from qspectro2d.visualization.plotting_functions import plot_data

# Suppress noisy but harmless warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")


def main():
    parser = argparse.ArgumentParser(description="Plot 1D or 2D electronic spectroscopy data")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--abs_path", type=str, help="Specific file path (absolute)")

    args = parser.parse_args()

    plot_config = {
        "extend_for": (1, 10),
        # "section": [(1, 3), (1, 3)],
        "section": [(1.4, 1.8), (1.4, 1.8)],
    }

    try:
        print(f"📁 Loading specific file: {args.abs_path}")
        loaded_data_and_info = load_simulation_data(abs_path=args.abs_path)
        t_det_axis = loaded_data_and_info.get("t_det")
        t_coh_axis = loaded_data_and_info.get("t_coh")
        sim_config = loaded_data_and_info["sim_config"]
        try:
            is_2d = (
                t_coh_axis is not None and hasattr(t_coh_axis, "__len__") and len(t_coh_axis) > 0
            )
        except Exception:
            is_2d = False

        n_t_det = len(t_det_axis)
        try:
            n_t_coh = len(t_coh_axis) if is_2d else 0
        except Exception:
            n_t_coh = 0
        print(f"   Axes: t_det len={n_t_det}; t_coh len={n_t_coh if is_2d else '—'}")
        sigs = [str(s) for s in sim_config.signal_types]
        datas = []
        for s in sigs:
            E_comp = loaded_data_and_info.get(s)
            shape = getattr(E_comp, "shape", None)
            print(f"   Signal '{s}' shape: {shape}")
            datas.append(E_comp)

        if datas and all(
            isinstance(a, np.ndarray) and a.size > 0 and np.allclose(a, 0) for a in datas
        ):
            print("⚠️  All-zero time-domain signals detected.")
        # Remove narrow section cropping for 1D to avoid empty slices; keep for 2D
        if not is_2d and "section" in plot_config:
            print("ℹ️  Removing frequency 'section' for 1D to auto-range the axis.")
            plot_config.pop("section", None)

        if is_2d:
            plot_data(loaded_data_and_info, plot_config, dimension="2d")
        else:
            plot_data(loaded_data_and_info, plot_config, dimension="1d")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
