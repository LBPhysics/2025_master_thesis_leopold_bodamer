#!/usr/bin/env python3
"""Simple plotting script for averaged 2D data."""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from plot_settings import *
from functions2DES import *


def main():
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(
        script_dir,
        "papers_with_proteus_output/average_over_freqs/averaged_data.pkl",  # TODO change this to the desired file path!!
    )

    with open(save_path, "rb") as f:
        data = pickle.load(f)

    two_d_datas = data["two_d_datas"]
    times_T = data["times_T"]
    times = data["times"]
    system_data = data["system"]

    # Plot each type
    types = ["imag", "real", "abs", "phase"]

    for type_ in types:
        print(f"Plotting {type_}...")

        plot_args_freq = {
            "domain": "freq",
            "type": type_,
            "save": True,
            "output_dir": "./figures",
            "positive": False,
            "use_custom_colormap": True,
            "system": system_data,
        }

        extend_and_plot_results(
            two_d_datas,
            times_T=times_T,
            times=times,
            extend_for=(1, 1),
            **plot_args_freq,
        )

        plt.close("all")


if __name__ == "__main__":
    main()
