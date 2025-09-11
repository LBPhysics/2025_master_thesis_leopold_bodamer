"""
Visualization subpackage for qspectro2d.

This subpackage provides plotting and visualization functionality
for spectroscopy data.
"""

from .plotting import (
    plot_pulse_envelopes,
    plot_e_pulses,
    plot_epsilon_pulses,
    plot_all_pulse_components,
    plot_example_evo,
    plot_1d_el_field,
    plot_2d_el_field,
    plot_example_polarization,
    crop_nd_data_along_axis,
    add_custom_contour_lines,
)
from .plotting_functions import (
    plot_data,
)

__all__ = [
    # plotting.py
    "plot_pulse_envelopes",
    "plot_e_pulses",
    "plot_epsilon_pulses",
    "plot_all_pulse_components",
    "plot_example_evo",
    "plot_1d_el_field",
    "plot_2d_el_field",
    "plot_example_polarization",
    "crop_nd_data_along_axis",
    "add_custom_contour_lines",
    # plotting_functions.py
    "plot_data",
]
