"""
Spectroscopy package for qspectro2d.

This package provides computational tools for 1D and 2D spectroscopy simulations,
including pulse evolution calculations, polarization computations, and post-processing
routines for Fourier transforms and signal analysis.

Main components:
- calculations: Core spectroscopy calculation functions
- inhomogenity: Tools for handling inhomogeneous broadening
- post_processing: FFT and signal processing utilities
- simulation: High-level simulation runners and utilities
"""

# =============================
# CORE CALCULATION FUNCTIONS
# =============================
from .calculations import (
    compute_pulse_evolution,
    compute_1d_polarization,
    check_the_solver,
    parallel_compute_1d_E_with_inhomogenity,
    extract_ift_signal_component,
)
from .polarization import complex_polarization

# =============================
# INHOMOGENEOUS BROADENING
# =============================
from .inhomogenity import (
    normalized_gauss,
    sample_from_gaussian,
)

# =============================
# POST-PROCESSING FUNCTIONS
# =============================
from .post_processing import (
    extend_time_axes,
    compute_1d_fft_wavenumber,
    compute_2d_fft_wavenumber,
)

# =============================
# SIMULATION FUNCTIONS
# =============================
from qspectro2d.utils import (
    apply_RWA_phase_factors,
    get_expect_vals_with_RWA,
    get_max_workers,
    print_simulation_summary,
)


__all__ = [
    # Core calculations
    "complex_polarization",
    "compute_pulse_evolution",
    "compute_1d_polarization",
    "check_the_solver",
    "parallel_compute_1d_E_with_inhomogenity",
    "parallel_compute_2d_E_with_inhomogenity",
    "extract_ift_signal_component",
    # Inhomogeneous broadening
    "normalized_gauss",
    "sample_from_gaussian",
    # Post-processing
    "extend_time_axes",
    "compute_1d_fft_wavenumber",
    "compute_2d_fft_wavenumber",
    # Submodules and simulation functions
    "get_max_workers",
    "print_simulation_summary",
    # Functions with RWA
    "apply_RWA_phase_factors",
    "get_expect_vals_with_RWA",
]
