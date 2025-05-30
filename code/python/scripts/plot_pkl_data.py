#!/usr/bin/env python3
"""Simple plotting script for 2D spectroscopy data from pickle files."""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config.paths import DATA_DIR, FIGURES_DIR
from src.spectroscopy.post_processing import extend_and_plot_results
from src.visualization.plotting import Plot_polarization_2d_spectrum
from src.visualization import mpl_tex_settings
from src.spectroscopy.calculations import get_tau_cohs_and_t_dets_for_T_wait


def find_latest_pkl_file():
    """Find the most recent pickle file in the data directory."""
    data_dir = DATA_DIR / "raw" / "2d_spectroscopy"
    pkl_files = list(data_dir.glob("*.pkl"))

    if not pkl_files:
        print("No pickle files found in", data_dir)
        return None

    # Sort by modification time, newest first
    latest_file = max(pkl_files, key=lambda f: f.stat().st_mtime)
    return latest_file


def load_pkl_data(filepath):
    """Load data from pickle file."""
    print(f"Loading data from: {filepath}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    two_d_datas = data["two_d_datas"]
    times_T = data["times_T"]
    times = data["times"]
    system_data = data["system"]

    print(f"Loaded {len(two_d_datas)} datasets")
    return two_d_datas, times_T, times, system_data


def main():
    # Get file path
    if len(sys.argv) > 1:
        # Use provided file path
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
    else:
        # Find latest file automatically
        file_path = find_latest_pkl_file()
        if file_path is None:
            return

    # Load data
    two_d_datas, times_T, times, system_data = load_pkl_data(file_path)

    # Create output directory
    output_dir = FIGURES_DIR / "2d_spectroscopy"
    os.makedirs(output_dir, exist_ok=True)

    # Plot each type
    types = ["imag", "real", "abs", "phase"]

    for type_ in types:
        print(f"Plotting {type_}...")
        for idx, T_wait in enumerate(times_T):
            print(f"Processing T_wait = {T_wait:.2f} fs")
            ts, taus = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait)
            Plot_polarization_2d_spectrum(
                (ts, taus, two_d_datas[idx]),
                T_wait=T_wait,
                save=True,
                output_dir=output_dir,
                use_custom_colormap=True,
            )
        plot_args_freq = {
            "domain": "freq",
            "type": type_,
            "save": True,
            "output_dir": output_dir,
            "positive": False,
            "use_custom_colormap": True,
            "system": system_data,
        }

        try:
            extend_and_plot_results(
                two_d_datas,
                times_T=times_T,
                times=times,
                extend_for=(1, 1),
                **plot_args_freq,
            )
            plt.close("all")
        except Exception as e:
            print(f"Error plotting {type_}: {e}")

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
