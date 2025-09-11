"""
QSpectro2D - Quantum 2D Electronic Spectroscopy Package

A comprehensive Python package for simulating 2D electronic spectroscopy
with quantum mechanical models. This package provides tools for:

- System parameter configuration and pulse sequence design
- Quantum dynamics simulation with various bath models
- 1D and 2D spectroscopy calculations with inhomogeneous broadening
- Data visualization and analysis tools
- Configuration management and file I/O utilities

Main subpackages:
- core: Fundamental simulation components (AtomicSystem, LaserPulseSequence, solvers)
- spectroscopy: 1D/2D spectroscopy calculations and post-processing
- visualization: Plotting and data visualization tools
- data: File I/O and data management utilities
- config: Configuration settings and constants
"""

__version__ = "0.1.0"  # Keep in sync with pyproject.toml
__author__ = "Leopold"
__email__ = ""


# LAZY CORE IMPORTS (avoid circular import during package init)


def __getattr__(name):  # PEP 562 lazy attribute loading
    # Core exports
    if name in {
        "AtomicSystem",
        "LaserPulse",
        "LaserPulseSequence",
        "e_pulses",
        "pulse_envelopes",
        "matrix_ODE_paper",
    }:
        from . import core as _core

        return getattr(_core, name)

    # Spectroscopy exports (lazy to avoid import cycles)
    if name in {
        "complex_polarization",
        "compute_1d_fft_wavenumber",
        "compute_2d_fft_wavenumber",
    }:
        from . import spectroscopy as _spec

        return getattr(_spec, name)

    # Bath functions (lazy)
    if name in {
        "power_spectrum_func_paper",
        "power_spectrum_func_drude_lorentz",
        "power_spectrum_func_ohmic",
    }:
        from .core import bath_system as _bath

        return getattr(_bath, name)

    # Visualization exports (lazy)
    if name in {
        "plot_1d_data",
        "plot_2d_data",
        "plot_pulse_envelopes",
        "plot_all_pulse_components",
    }:
        from . import visualization as _viz

        return getattr(_viz, name)

    # Utils exports (lazy)
    if name in {
        "save_simulation_data",
        "load_data_from_abs_path",
        "load_latest_data_from_directory",
    }:
        from . import utils as _utils

        return getattr(_utils, name)

    # Config exports (lazy)
    if name in {"DATA_DIR", "FIGURES_DIR"}:
        from . import config as _cfg

        return getattr(_cfg, name)

    raise AttributeError(name)


# Bath models are available under qspectro2d.core.bath_system
# Avoid importing here to prevent circular import at package import time.

# Spectroscopy functions are provided via lazy loader above to avoid cycles at import time.


# VISUALIZATION

try:
    from .visualization import (
        plot_1d_data,
        plot_2d_data,
        plot_pulse_envelopes,
        plot_all_pulse_components,
    )
except ImportError as e:
    print(f"Warning: Could not import visualization module: {e}")


# DATA MANAGEMENT

try:
    from .utils import (
        save_simulation_data,
        load_data_from_abs_path,
        load_latest_data_from_directory,
    )
except ImportError as e:
    print(f"Warning: Could not import data module: {e}")


# CONFIGURATION

try:
    from .config import (
        DATA_DIR,
        FIGURES_DIR,
    )
except ImportError as e:
    print(f"Warning: Could not import config module: {e}")


# PUBLIC API - MOST COMMONLY USED

__all__ = [
    # Core classes - most important for users
    "AtomicSystem",
    "LaserPulse",
    "LaserPulseSequence",
    # Essential functions
    "e_pulses",
    "pulse_envelopes",
    "matrix_ODE_paper",
    # Bath functions
    "power_spectrum_func_paper",
    "power_spectrum_func_drude_lorentz",
    "power_spectrum_func_ohmic",
    # High-level simulation functions
    "complex_polarization",
    # Post-processing
    "compute_1d_fft_wavenumber",
    "compute_2d_fft_wavenumber",
    # Visualization
    "plot_1d_data",
    "plot_2d_data",
    "plot_pulse_envelopes",
    "plot_all_pulse_components",
    # Data management
    "save_simulation_data",
    "load_data_from_abs_path",
    "load_latest_data_from_directory",
    # Configuration
    "DATA_DIR",
    "FIGURES_DIR",
]


# PACKAGE INFORMATION


def get_package_info():
    """
    Display package information and available modules.
    """
    info = f"""
QSpectro2D Package Information
=============================
Version: {__version__}
Author: {__author__}

Available Subpackages:
- core: System parameters, pulses, and solvers
- baths: Bosonic bath models  
- spectroscopy: 1D/2D simulation and post-processing
- visualization: Plotting and data visualization
- data: File I/O and data management
- config: Configuration and constants

Quick Start:
from qspectro2d import AtomicSystem, LaserPulseSequence, run_1d_simulation

For detailed documentation, see individual module docstrings.
"""
    return info


def list_available_functions():
    """
    List all functions available in the main namespace.
    """
    return __all__
