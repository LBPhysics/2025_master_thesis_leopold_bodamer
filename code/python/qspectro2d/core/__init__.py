"""
Core module for qspectro2d package.

This module provides the fundamental building blocks for 2D spectroscopy simulations:
- System parameters and configuration
- Pulse definitions and sequences
- Pulse field functions
- Solver functions for system dynamics
- RWA (Rotating Wave Approximation) utilities

The core module is designed to handle both single-atom and two-atom systems
with various bath models and pulse configurations.
"""

# =============================
# SYSTEM PARAMETERS
# =============================
from .atomic_system.system_class import AtomicSystem

# =============================
# PULSE DEFINITIONS AND SEQUENCES
# =============================
from .laser_system.laser_class import (
    Pulse,
    LaserPulseSystem,
    identify_non_zero_pulse_regions,
    split_by_active_regions,
)

# =============================
# PULSE FIELD FUNCTIONS
# =============================
from .laser_system.laser_fcts import (
    pulse_envelope,
    E_pulse,
    Epsilon_pulse,
)

# =============================
# SOLVER FUNCTIONS
# =============================
from .solver_fcts import (
    matrix_ODE_paper,
    R_paper,
)

# =============================
# RWA UTILITIES
# =============================
from .functions_with_rwa import (
    apply_RWA_phase_factors,
    get_expect_vals_with_RWA,
)

# =============================
# UTILITIES AND CONSTANTS
# =============================
from .utils_and_config import (
    convert_cm_to_fs,
    convert_fs_to_cm,
    HBAR,
    BOLTZMANN,
)

# =============================
# PUBLIC API
# =============================
__all__ = [
    # System configuration
    "AtomicSystem",
    # Pulse definitions
    "Pulse",
    "LaserPulseSystem",
    # Pulse field functions
    "pulse_envelope",
    "E_pulse",
    "Epsilon_pulse",
    "identify_non_zero_pulse_regions",
    "split_by_active_regions",
    # Solver functions
    "matrix_ODE_paper",
    "R_paper",
    # RWA utilities
    "apply_RWA_phase_factors",
    "get_expect_vals_with_RWA",
    # Utilities and constants
    "convert_cm_to_fs",
    "convert_fs_to_cm",
    # constants
    "HBAR",
    "BOLTZMANN",
]

# =============================
# VERSION INFO
# =============================
__version__ = "1.0.0"
__author__ = "Leopold Bodamer"
__email__ = ""
