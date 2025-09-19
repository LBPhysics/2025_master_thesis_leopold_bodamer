"""
Spectroscopy package for qspectro2d.

This package provides computational tools for 1D and 2D spectroscopy simulations,
including pulse evolution calculations, polarization computations, and post-processing
routines for Fourier transforms and signal analysis.

Main components:
- one_d_field: 1D field and polarization computation pipeline
- inhomogenity: Tools for handling inhomogeneous broadening
- post_processing: FFT and signal processing utilities
- simulation: High-level simulation runners and utilities
"""

# CORE CALCULATION FUNCTIONS

from .one_d_field import (
    compute_evolution,
    parallel_compute_1d_e_comps,
)
from .one_d_field import phase_cycle_component as extract_ift_signal_component
from .polarization import complex_polarization

from .solver_check import check_the_solver

# INHOMOGENEOUS BROADENING

from .inhomogenity import (
    normalized_gauss,
    sample_from_gaussian,
)


# POST-PROCESSING FUNCTIONS

from .post_processing import (
    extend_time_axes,
    compute_1d_fft_wavenumber,
    compute_2d_fft_wavenumber,
)


# SIMULATION FUNCTIONS

from qspectro2d.utils import (
    rotating_frame_unitary,
    to_rotating_frame_op,
    from_rotating_frame_op,
    to_rotating_frame_list,
    from_rotating_frame_list,
    get_expect_vals_with_RWA,
)


__all__ = [
    # Core calculations
    "complex_polarization",
    "compute_evolution",
    "check_the_solver",
    "parallel_compute_1d_e_comps",
    "extract_ift_signal_component",
    # Inhomogeneous broadening
    "normalized_gauss",
    "sample_from_gaussian",
    # Post-processing
    "extend_time_axes",
    "compute_1d_fft_wavenumber",
    "compute_2d_fft_wavenumber",
    # Functions with RWA
    "rotating_frame_unitary",
    "to_rotating_frame_op",
    "from_rotating_frame_op",
    "to_rotating_frame_list",
    "from_rotating_frame_list",
    "get_expect_vals_with_RWA",
]
