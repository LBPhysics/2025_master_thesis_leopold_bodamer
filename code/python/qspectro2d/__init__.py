"""
QSpectro2D - Quantum 2D Electronic Spectroscopy Package

This package contains modules for simulating 2D electronic spectroscopy.
"""

__version__ = "0.1.0"
__author__ = "Leopold"

# Import main modules
try:
    from .baths import *
    from .core import *
    from .spectroscopy import *
    from .visualization import *
except ImportError as e:
    # Graceful fallback if modules are not yet available
    print("Error")
    pass
