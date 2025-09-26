"""
QSpectro2D - Quantum 2D Electronic Spectroscopy Package

A comprehensive Python package for simulating 2D electronic spectroscopy
with Open Quantum Systems models. This package provides tools for:

- System parameter configuration and pulse sequence design
- OQS dynamics simulation with various bath models
- 1D and 2D spectroscopy calculations with inhomogeneous broadening (TODO not yet)
- Data visualization tools
- Configuration management and file I/O utilities

Main subpackages:
- config: Configuration settings and constants
- core: Fundamental simulation components (AtomicSystem, LaserPulseSequence, solvers, bath models)
- spectroscopy: 1D/2D spectroscopy calculations and post-processing
- utils: File I/O, units, and helper utilities
- visualization: Plotting and data visualization tools
"""

__version__ = "1.0"
__author__ = "Leopold Bodamer"

# Silence a specific QuTiP FutureWarning about keyword-only args in brmesolve
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*c_ops, e_ops, args and options will be keyword only from qutip 5\.3.*",
    module=r"qutip\.solver\.brmesolve",
)


# Core exports
from .config import (
    create_base_sim_oqs,
)

from .core import (
    # atomic system
    AtomicSystem,
    # laser pulses
    LaserPulse,
    LaserPulseSequence,
    e_pulses,
    epsilon_pulses,
    single_pulse_envelope,
    pulse_envelopes,
    # bath models
    power_spectrum_func_paper,
    power_spectrum_func_drude_lorentz,
    power_spectrum_func_ohmic,
    # simulation functions
    matrix_ODE_paper,
)

from .utils import (
    # file naming
    generate_unique_plot_filename,
    generate_unique_data_filename,
    # data I/O
    save_simulation_data,
    load_simulation_data,
)

# Spectroscopy exports (imported after data I/O to avoid partial init race)
from .spectroscopy import (
    extend_time_domain_data,
    compute_spectra,
    complex_polarization,
    compute_evolution,
    check_the_solver,
    sim_with_only_pulses,
)


# PUBLIC API - MOST COMMONLY USED
__all__ = [
    # Configuration
    "create_base_sim_oqs",
    # Core classes - most important
    "AtomicSystem",
    "LaserPulse",
    "LaserPulseSequence",
    # Essential functions
    "single_pulse_envelope",
    "pulse_envelopes",
    "e_pulses",
    "epsilon_pulses",
    # Bath functions
    "power_spectrum_func_paper",
    "power_spectrum_func_drude_lorentz",
    "power_spectrum_func_ohmic",
    # Simulation functions
    "matrix_ODE_paper",
    # High-level simulation functions
    "complex_polarization",
    "compute_evolution",
    "check_the_solver",
    "sim_with_only_pulses",
    # Post-processing
    "extend_time_domain_data",
    "compute_spectra",
    # Data management
    "save_simulation_data",
    "load_simulation_data",
    # file management
    "generate_unique_plot_filename",
    "generate_unique_data_filename",
]


# PACKAGE INFORMATION
def list_available_functions():
    """
    List all functions available in the main namespace.
    """
    return __all__
