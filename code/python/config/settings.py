"""TODO implement this module
General configuration settings for the qspectro2d project.

This module contains project-wide configuration constants and settings.
"""

import numpy as np

# =============================
# PHYSICAL CONSTANTS
# =============================

# Time units (in femtoseconds)
FS_TO_PS = 1e-3
PS_TO_FS = 1e3

# Energy units (conversion factors)
EV_TO_CM1 = 8065.54  # eV to cm^-1
CM1_TO_EV = 1 / EV_TO_CM1
HARTREE_TO_EV = 27.2114
EV_TO_HARTREE = 1 / HARTREE_TO_EV

# Fundamental constants
HBAR = 1.0545718e-34  # J⋅s
KB = 1.380649e-23  # J/K
C_LIGHT = 2.99792458e8  # m/s

# =============================
# NUMERICAL SETTINGS
# =============================

# Default numerical precision
DEFAULT_ATOL = 1e-12
DEFAULT_RTOL = 1e-9

# Default integration parameters
DEFAULT_NSTEPS = 1000
DEFAULT_MAX_STEP = 0.1  # fs


# =============================
# COMPUTATIONAL SETTINGS
# =============================

# Default number of parallel processes (None = auto-detect)
DEFAULT_N_JOBS = None

# Memory management
DEFAULT_CHUNK_SIZE = 1000  # For large calculations
