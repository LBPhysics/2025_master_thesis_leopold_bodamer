"""
Visualization subpackage for qspectro2d.

This subpackage provides plotting and visualization functionality
for spectroscopy data.
"""

from .plotting import (
    plot_pulse_envelope,
    plot_e_pulse,
    plot_epsilon_pulse,
    plot_all_pulse_components,
    plot_example_evo,
    plot_1d_el_field,
    plot_2d_el_field,
    plot_example_polarization,
    crop_2d_data_to_section,
    add_custom_contour_lines,
)
from .data_plots import (
    plot_1d_data,
    plot_2d_data,
)

__all__ = [
    # plotting.py
    "plot_pulse_envelope",
    "plot_e_pulse",
    "plot_epsilon_pulse",
    "plot_all_pulse_components",
    "plot_example_evo",
    "plot_1d_el_field",
    "plot_2d_el_field",
    "plot_example_polarization",
    "crop_2d_data_to_section",
    "add_custom_contour_lines",
    # data_plots.py
    "plot_1d_data",
    "plot_2d_data",
]
