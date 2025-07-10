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
- core: Fundamental simulation components (AtomicSystem, LaserPulseSequence, BathClass, solvers)
- spectroscopy: 1D/2D spectroscopy calculations and post-processing
- visualization: Plotting and data visualization tools
- data: File I/O and data management utilities
- config: Configuration settings and constants
"""

__version__ = "1.0.0"
__author__ = "Leopold"
__email__ = ""

# =============================
# CORE FUNCTIONALITY
# =============================
# Import the most essential classes that users will need
try:
    from .core import AtomicSystem, LaserPulse, LaserPulseSequence
    from .core import E_pulse, pulse_envelope
    from .core import matrix_ODE_paper
    from .core import convert_cm_to_fs, convert_fs_to_cm, HBAR, BOLTZMANN
except ImportError as e:
    print(f"Warning: Could not import core module: {e}")

# =============================
# BATH MODELS
# =============================
try:
    from .core.bath_system import (
        power_spectrum_func_paper,
        power_spectrum_func_drude_lorentz,
        power_spectrum_func_ohmic,
    )
except ImportError as e:
    print(f"Warning: Could not import bath_system module: {e}")

# =============================
# SPECTROSCOPY SIMULATIONS
# =============================
try:
    from .spectroscopy import (
        run_1d_simulation,
        run_2d_simulation,
        complex_polarization,
        compute_1d_fft_wavenumber,
        compute_2d_fft_wavenumber,
    )
except ImportError as e:
    print(f"Warning: Could not import spectroscopy module: {e}")

# =============================
# VISUALIZATION
# =============================
try:
    from .visualization import (
        plot_1d_data,
        plot_2d_data,
        plot_pulse_envelope,
        plot_all_pulse_components,
    )
except ImportError as e:
    print(f"Warning: Could not import visualization module: {e}")

# =============================
# DATA MANAGEMENT
# =============================
try:
    from .data import (
        save_simulation_data,
        load_data_from_rel_path,
        load_latest_data_from_directory,
    )
except ImportError as e:
    print(f"Warning: Could not import data module: {e}")

# =============================
# CONFIGURATION
# =============================
try:
    from .config import (
        DATA_DIR,
        FIGURES_DIR,
        set_size,
        save_fig,
        COLORS,
        LINE_STYLES,
    )
except ImportError as e:
    print(f"Warning: Could not import config module: {e}")

# =============================
# PUBLIC API - MOST COMMONLY USED
# =============================
__all__ = [
    # Core classes - most important for users
    "AtomicSystem",
    "LaserPulse",
    "LaserPulseSequence",
    # Essential functions
    "E_pulse",
    "pulse_envelope",
    "matrix_ODE_paper",
    # Bath functions
    "power_spectrum_func_paper",
    "power_spectrum_func_drude_lorentz",
    "power_spectrum_func_ohmic",
    # High-level simulation functions
    "run_1d_simulation",
    "run_2d_simulation",
    "complex_polarization",
    # Post-processing
    "compute_1d_fft_wavenumber",
    "compute_2d_fft_wavenumber",
    # Visualization
    "plot_1d_data",
    "plot_2d_data",
    "plot_pulse_envelope",
    "plot_all_pulse_components",
    # Data management
    "save_simulation_data",
    "load_data_from_rel_path",
    "load_latest_data_from_directory",
    # Configuration
    "DATA_DIR",
    "FIGURES_DIR",
    "set_size",
    "save_fig",
    "COLORS",
    "LINE_STYLES",
]


# =============================
# PACKAGE INFORMATION
# =============================
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
