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
N_ATOMS = 1
N_CHAINS = 1  # defaults to linear chain (single chain layout)

FREQUENCIES_CM = [15900.0]  # Number of frequency components in the system
DIP_MOMENTS = [1.0]  # Dipole moments for each atom
COUPLING_CM = 0.0  # Coupling strength [cm⁻¹]
DELTA_CM = 0.0  # Inhomogeneous broadening [cm⁻¹]
MAX_EXCITATION = 2  # 1 -> ground+single manifold, 2 -> add double-excitation manifold

# === LASER SYSTEM DEFAULTS ===
PULSE_FWHM = 15.0 if N_ATOMS == 1 else 5.0  # Pulse FWHM in fs
BASE_AMPLITUDE = 0.5  # should be such that only one interaction at a time, here that |excitation|² < 1%
ENVELOPE_TYPE = "gaussian"  # Type of pulse envelope # gaussian or cos2
CARRIER_FREQ_CM = 16000.0  # np.mean(FREQUENCIES_CM)  # Carrier frequency of the laser

# === SIMULATION DEFAULTS ===
ODE_SOLVER = "BR"  # ODE solver to use
RWA_SL = True
N_FREQS = 1  # 1 == no inhomogeneous broadening
N_PHASES = 4  # Number of phase cycles for the simulation


# === BATH SYSTEM DEFAULTS ===
frequencies = [convert_cm_to_fs(freq_cm) for freq_cm in FREQUENCIES_CM]
BATH_TYPE = "ohmic"  # TODO at the moment only ohmic baths are supported
BATH_CUTOFF = 1e2 * frequencies[0]  # Cutoff frequency in cm⁻¹
BATH_TEMP = 1e3 * frequencies[0] / BOLTZMANN
BATH_COUPLING = 1e-4 * frequencies[0]


# === 2D SIMULATION DEFAULTS ===
BATCHES = 10  # You can increase/decrease this
T_DET_MAX = 200.0  # Maximum detection time in fs
DT = 0.1  # Spacing between t_coh, and of also t_det values in fs


# =============================
# VALIDATION AND SANITY CHECKS
# =============================
def validate(params: dict):
    """Validate that a parameter dictionary is consistent and sensible."""
    # Extract parameters with defaults fallback
    ode_solver = params.get("solver", ODE_SOLVER)
    bath_type = params.get("bath_type", BATH_TYPE)
    frequencies_cm = params.get("frequencies_cm", FREQUENCIES_CM)
    n_atoms = params.get("n_atoms", N_ATOMS)
    dip_moments = params.get("dip_moments", DIP_MOMENTS)
    bath_temp = params.get("temperature", BATH_TEMP)
    bath_cutoff = params.get("cutoff", BATH_CUTOFF)
    bath_coupling = params.get("coupling", BATH_COUPLING)
    n_phases = params.get("n_phases", N_PHASES)
    max_excitation = params.get("max_excitation", MAX_EXCITATION)
    n_chains = params.get("n_chains", N_CHAINS)
    relative_e0s = params.get("relative_e0s", RELATIVE_E0S)
    rwa_sl = params.get("rwa_sl", RWA_SL)
    carrier_freq_cm = params.get("carrier_freq_cm", CARRIER_FREQ_CM)

    # Validate solver
    if ode_solver not in SUPPORTED_SOLVERS:
        raise ValueError(f"ODE_SOLVER '{ode_solver}' not in {SUPPORTED_SOLVERS}")

    # Validate bath type
    if bath_type not in SUPPORTED_BATHS:
        raise ValueError(f"BATH_TYPE '{bath_type}' not in {SUPPORTED_BATHS}")

    # Validate atomic system consistency
    if len(frequencies_cm) != n_atoms:
        raise ValueError(
            f"FREQUENCIES_CM length ({len(frequencies_cm)}) != N_ATOMS ({n_atoms})"
        )

    if len(dip_moments) != n_atoms:
        raise ValueError(
            f"DIP_MOMENTS length ({len(dip_moments)}) != N_ATOMS ({n_atoms})"
        )

    # Validate positive values
    if bath_temp <= 0:
        raise ValueError("BATH_TEMP must be positive")

    if bath_cutoff <= 0:
        raise ValueError("BATH_CUTOFF must be positive")

    if bath_coupling <= 0:
        raise ValueError("BATH_COUPLING must be positive")

    # Validate phases
    if n_phases <= 0:
        raise ValueError("N_PHASES must be positive")

    # Validate excitation truncation
    if max_excitation not in (1, 2):
        raise ValueError("MAX_EXCITATION must be 1 or 2")

    # Validate n_chains divisibility if provided and relevant
    if n_chains is not None and n_atoms > 2:
        if n_chains < 1:
            raise ValueError("N_CHAINS must be >=1 when specified")
        if n_atoms % n_chains != 0:
            raise ValueError(
                f"N_CHAINS ({n_chains}) does not divide N_ATOMS ({n_atoms}) for cylindrical geometry"
            )

    # Validate relative amplitudes
    if len(relative_e0s) != 3:
        raise ValueError("RELATIVE_E0S must have exactly 3 elements")

    if rwa_sl:
        freqs_array = np.array(frequencies_cm)
        max_detuning = np.max(np.abs(freqs_array - carrier_freq_cm))
        rel_detuning = (
            max_detuning / carrier_freq_cm if carrier_freq_cm != 0 else np.inf
        )
        if rel_detuning > 1e-2:
            print(
                f"WARNING: RWA probably not valid, since relative detuning: {rel_detuning} is too large",
                flush=True,
            )


def validate_defaults():
    """Validate that all default values are consistent and sensible."""
    params = {
        "solver": ODE_SOLVER,
        "bath_type": BATH_TYPE,
        "frequencies_cm": FREQUENCIES_CM,
        "n_atoms": N_ATOMS,
        "dip_moments": DIP_MOMENTS,
        "temperature": BATH_TEMP,
        "cutoff": BATH_CUTOFF,
        "coupling": BATH_COUPLING,
        "n_phases": N_PHASES,
        "max_excitation": MAX_EXCITATION,
        "n_chains": N_CHAINS,
        "relative_e0s": RELATIVE_E0S,
        "rwa_sl": RWA_SL,
        "carrier_freq_cm": CARRIER_FREQ_CM,
    }
    validate(params)
