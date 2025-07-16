"""
Core module for qspectro2d package.

This module provides the fundamental building blocks for 2D spectroscopy simulations:
- System parameters and configuration
- LaserPulse definitions and sequences
- LaserPulse field functions
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
# BATH SYSTEMS
# =============================
from .bath_system.bath_class import BathSystem

# =============================
# PULSE DEFINITIONS AND SEQUENCES
# =============================
from .laser_system.laser_class import (
    LaserPulse,
    LaserPulseSequence,
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
# WHOLE MODULE CLASS AND specific Paper SOLVER FUNCTIONS
# =============================
from .simulation_class import (
    SimulationModuleOQS,
    SimulationConfig,
    SimulationModuleOQS,
    matrix_ODE_paper,
    R_paper,
)

# =============================
# PUBLIC API
# =============================
__all__ = [
    # System configuration
    "AtomicSystem",
    # LaserPulse definitions
    "LaserPulse",
    "LaserPulseSequence",
    # Environment system
    "BathSystem",
    # Simulation module and configuration
    "SimulationModuleOQS",
    "SimulationConfig",
    # LaserPulse field functions
    "pulse_envelope",
    "E_pulse",
    "Epsilon_pulse",
    "identify_non_zero_pulse_regions",
    "split_by_active_regions",
    # Solver functions
    "matrix_ODE_paper",
    "R_paper",
]

# =============================
# VERSION INFO
# =============================
__version__ = "1.0.0"
__author__ = "Leopold Bodamer"
__email__ = ""
