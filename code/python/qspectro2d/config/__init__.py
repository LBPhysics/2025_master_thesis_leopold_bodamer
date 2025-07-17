"""
Configuration package for the Master's thesis project.

This package contains all configuration files for the project.
"""

# Make modules importable
from .mpl_tex_settings import (
    # Constants
    LATEX_DOC_WIDTH,
    LATEX_FONT_SIZE,
    FIGSIZE,
    DPI,
    FONT_SIZE,
    FIG_FORMAT,
    TRANSPARENCY,
    COLORS,
    LINE_STYLES,
    MARKERS,
    # Settings
    latex_available,
    # Functions
    set_size,
    format_sci_notation,
    save_fig,
)

from .paths import (
    # Paths
    DATA_DIR,
    FIGURES_DIR,
    FIGURES_PYTHON_DIR,
    SCRIPTS_DIR,
    FIGURES_BATH_DIR,
    FIGURES_PULSES_DIR,
    FIGURES_TESTS_DIR,
)

from .default_simulation_params import (
    # Fundamental constants
    HBAR,
    BOLTZMANN,
    # Atomic system defaults
    N_ATOMS,
    FREQS_CM,
    DELTA_CM,
    J_COUPLING_CM,
    DIP_MOMENTS,
    # Simulation defaults
    ODE_SOLVER,
    RWA_SL,
    N_FREQS,
    N_PHASES,
    # Bath system defaults
    BATH_TYPE,
    BATH_TEMP,
    BATH_CUTOFF,
    BATH_GAMMA_0,
    BATH_GAMMA_PHI,
    # Laser system defaults
    PULSE_FWHM,
    CARRIER_FREQ_CM,
    ENVELOPE_TYPE,
    BASE_AMPLITUDE,
    # Signal processing defaults
    IFT_COMPONENT,
    RELATIVE_E0S,
    # Solver defaults
    SOLVER_OPTIONS,
    NEGATIVE_EIGVAL_THRESHOLD,
    TRACE_TOLERANCE,
    PHASE_CYCLING_PHASES,
    DETECTION_PHASE,
    # 2d simulation defaults
    T_DET_MAX,
    DT,
    BATCHES,
    # Supported options
    SUPPORTED_SOLVERS,
    SUPPORTED_BATHS,
    # Validation function
    validate_defaults,
)

# Export all important symbols for import *
__all__ = [
    # constants
    "LATEX_DOC_WIDTH",
    "LATEX_FONT_SIZE",
    "FIGSIZE",
    "DPI",
    "FONT_SIZE",
    "FIG_FORMAT",
    "TRANSPARENCY",
    "COLORS",
    "LINE_STYLES",
    "MARKERS",
    # settings
    "latex_available",
    # functions
    "set_size",
    "format_sci_notation",
    "save_fig",
    # paths
    "DATA_DIR",
    "FIGURES_DIR",
    "FIGURES_PYTHON_DIR",
    "SCRIPTS_DIR",
    "FIGURES_BATH_DIR",
    "FIGURES_PULSES_DIR",
    "FIGURES_TESTS_DIR",
    # default simulation parameters
    "HBAR",
    "BOLTZMANN",
    "N_ATOMS",
    "DELTA_CM",
    "FREQS_CM",
    "DIP_MOMENTS",
    "J_COUPLING_CM",
    # simulation defaults
    "ODE_SOLVER",
    "RWA_SL",
    "N_FREQS",
    "N_PHASES",
    # bath system defaults
    "BATH_TYPE",
    "BATH_TEMP",
    "BATH_CUTOFF",
    "BATH_GAMMA_0",
    "BATH_GAMMA_PHI",
    # laser system defaults
    "PULSE_FWHM",
    "CARRIER_FREQ_CM",
    "ENVELOPE_TYPE",
    "BASE_AMPLITUDE",
    # signal processing defaults
    "IFT_COMPONENT",
    "RELATIVE_E0S",
    # solver defaults
    "SOLVER_OPTIONS",
    "NEGATIVE_EIGVAL_THRESHOLD",
    "TRACE_TOLERANCE",
    "PHASE_CYCLING_PHASES",
    "DETECTION_PHASE",
    # 2d simulation defaults
    "T_DET_MAX",
    "DT",
    "BATCHES",
    # supported options
    "SUPPORTED_SOLVERS",
    "SUPPORTED_BATHS",
    # validation
    "validate_defaults",
]
