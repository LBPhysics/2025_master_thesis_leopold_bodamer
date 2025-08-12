"""
Default simulation parameters for qspectro2d.

This module contains default values for simulation parameters used across
the project. Centralizing these constants makes them easier to maintain
and reduces code duplication.
"""

import numpy as np
from qspectro2d.constants import (
    BOLTZMANN,
    convert_cm_to_fs,
)


# =============================
# fixed constants: don't change!
# =============================

# === signal processing / phase cycling ===
PHASE_CYCLING_PHASES = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
DETECTION_PHASE = 0  # Fixed phase for detection pulse
# (0, 0, 0) == normal average || (-1, 1, 0) == photon echo signal(phase cycling)
IFT_COMPONENT = (
    1,
    -1,
    1,  # does not matter because DETECTION_PHASE = 0
)
# last pulse is 10% of the first two to ensure probing character
RELATIVE_E0S = [1.0, 1.0, 0.1]


# =============================
# solver defaults  # very rough estimate, not optimized
# =============================
SOLVER_OPTIONS = {"nsteps": 200000, "atol": 1e-6, "rtol": 1e-4}

# Validation thresholds for physics checks
NEGATIVE_EIGVAL_THRESHOLD = -1e-3
TRACE_TOLERANCE = 1e-6

# physical constants imported from qspectro2d.constants

# supported solvers and bath models
SUPPORTED_SOLVERS = ["ME", "BR", "Paper_eqs", "Paper_BR"]
SUPPORTED_BATHS = ["ohmic"]  # , "dl"


# === ATOMIC SYSTEM DEFAULTS ===
N_ATOMS = 2
N_RINGS = (
    None  # If N_ATOMS>2 and None -> defaults to linear chain (single chain layout)
)
FREQS_CM = [15900.0, 16100.0]  # Number of frequency components in the system
DIP_MOMENTS = [1.0, 1.0]  # Dipole moments for each atom
AT_COUPLING_CM = 0.0  # Coupling strength [cm⁻¹]
DELTA_CM = 0.0  # Inhomogeneous broadening [cm⁻¹]
MAX_EXCITATION = 2  # 1 -> ground+single manifold, 2 -> add double-excitation manifold

# === LASER SYSTEM DEFAULTS ===
PULSE_FWHM = 15.0 if N_ATOMS == 1 else 5.0  # Pulse FWHM in fs
BASE_AMPLITUDE = 0.5  # should be such that only one interaction at a time, here that |excitation|² < 1%
ENVELOPE_TYPE = "gaussian"  # Type of pulse envelope # gaussian or cos2
CARRIER_FREQ_CM = 16000.0  # np.mean(FREQS_CM)  # Carrier frequency of the laser

# === SIMULATION DEFAULTS ===
ODE_SOLVER = "BR"  # ODE solver to use
RWA_SL = True
N_FREQS = 1  # 1 == no inhomogeneous broadening
N_PHASES = 4  # Number of phase cycles for the simulation


# === BATH SYSTEM DEFAULTS ===
at_freqs_fs = [convert_cm_to_fs(freq_cm) for freq_cm in FREQS_CM]
BATH_TYPE = "ohmic"  # TODO at the moment only ohmic baths are supported
BATH_CUTOFF = 1e2 * at_freqs_fs[0]  # Cutoff frequency in cm⁻¹
BATH_TEMP = 1e0 * at_freqs_fs[0] / BOLTZMANN
BATH_COUPLING = 1e-4 * at_freqs_fs[0]


# === 2D SIMULATION DEFAULTS ===
BATCHES = 10  # You can increase/decrease this
T_DET_MAX = 200.0  # Maximum detection time in fs
DT = 0.1  # Spacing between t_coh, and of also t_det values in fs


# =============================
# VALIDATION AND SANITY CHECKS
# =============================
def validate_defaults():
    """Validate that all default values are consistent and sensible."""

    # Validate solver
    if ODE_SOLVER not in SUPPORTED_SOLVERS:
        raise ValueError(f"ODE_SOLVER '{ODE_SOLVER}' not in {SUPPORTED_SOLVERS}")

    # Validate bath type
    if BATH_TYPE not in SUPPORTED_BATHS:
        raise ValueError(f"BATH_TYPE '{BATH_TYPE}' not in {SUPPORTED_BATHS}")

    # Validate atomic system consistency
    if len(FREQS_CM) != N_ATOMS:
        raise ValueError(f"FREQS_CM length ({len(FREQS_CM)}) != N_ATOMS ({N_ATOMS})")

    if len(DIP_MOMENTS) != N_ATOMS:
        raise ValueError(
            f"DIP_MOMENTS length ({len(DIP_MOMENTS)}) != N_ATOMS ({N_ATOMS})"
        )

    # Validate positive values
    if BATH_TEMP <= 0:
        raise ValueError("BATH_TEMP must be positive")

    if BATH_CUTOFF <= 0:
        raise ValueError("BATH_CUTOFF must be positive")

    if BATH_COUPLING <= 0:
        raise ValueError("BATH_COUPLING must be positive")

    # Validate phases
    if N_PHASES <= 0:
        raise ValueError("N_PHASES must be positive")

    # Validate excitation truncation
    if MAX_EXCITATION not in (1, 2):
        raise ValueError("MAX_EXCITATION must be 1 or 2")

    # Validate n_rings divisibility if provided and relevant
    if N_RINGS is not None and N_ATOMS > 2:
        if N_RINGS < 1:
            raise ValueError("N_RINGS must be >=1 when specified")
        if N_ATOMS % N_RINGS != 0:
            raise ValueError(
                f"N_RINGS ({N_RINGS}) does not divide N_ATOMS ({N_ATOMS}) for cylindrical geometry"
            )

    # Validate relative amplitudes
    if len(RELATIVE_E0S) != 3:
        raise ValueError("RELATIVE_E0S must have exactly 3 elements")

    if RWA_SL:
        freqs_array = np.array(FREQS_CM)
        max_detuning = np.max(np.abs(freqs_array - CARRIER_FREQ_CM))
        rel_detuning = (
            max_detuning / CARRIER_FREQ_CM if CARRIER_FREQ_CM != 0 else np.inf
        )
        if rel_detuning > 1e-2:
            print(
                f"WARNING: RWA probably not valid, since relative detuning: {rel_detuning} is too large"
            )


# Run validation when module is imported
validate_defaults()
