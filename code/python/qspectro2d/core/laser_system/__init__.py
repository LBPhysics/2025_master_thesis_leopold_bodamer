"""
Laser system module for qspectro2d package.

This module provides functions and classes for defining and manipulating laser pulses,
including their electric field profiles and temporal shapes.

"""

# =============================
# LASER SYSTEM FUNCTIONS AND CLASSES
# =============================
from .laser_fcts import E_pulse, Epsilon_pulse, pulse_envelope
from .laser_class import (
    LaserPulseSequence,
    LaserPulse,
    identify_non_zero_pulse_regions,
    split_by_active_regions,
)

# =============================
# PUBLIC API
# =============================
__all__ = [
    # functions
    "E_pulse",
    "Epsilon_pulse",
    "pulse_envelope",
    "identify_non_zero_pulse_regions",
    "split_by_active_regions",
    # classes
    "LaserPulseSequence",
    "LaserPulse",
]

# =============================
# VERSION INFO
# =============================
__version__ = "1.0.0"
__author__ = "Leopold Bodamer"
__email__ = ""
