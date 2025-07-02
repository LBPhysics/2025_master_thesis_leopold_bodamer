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
from .system_parameters import SystemParameters

# =============================
# PULSE DEFINITIONS AND SEQUENCES
# =============================
from .pulse_sequences import Pulse, PulseSequence

# =============================
# PULSE FIELD FUNCTIONS
# =============================
from .pulse_functions import (
    pulse_envelope,
    E_pulse,
    Epsilon_pulse,
    identify_non_zero_pulse_regions,
    split_by_active_regions,
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
    H_int,
    apply_RWA_phase_factors,
    get_expect_vals_with_RWA,
)

# =============================
# PUBLIC API
# =============================
__all__ = [
    # System configuration
    "SystemParameters",
    # Pulse definitions
    "Pulse",
    "PulseSequence",
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
    "H_int",
    "apply_RWA_phase_factors",
    "get_expect_vals_with_RWA",
]

# =============================
# VERSION INFO
# =============================
__version__ = "1.0.0"
__author__ = "Leopold"
__email__ = ""