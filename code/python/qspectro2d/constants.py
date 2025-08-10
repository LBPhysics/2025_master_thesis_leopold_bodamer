"""Core physical constants and unit conversion helpers.

Lightweight module: safe to import from any layer without triggering
expensive or circular imports. Keep ONLY primitive constants and pure
functions here.
"""

from __future__ import annotations
import numpy as np

# Fundamental constants (natural units convention inside project)
HBAR: float = 1.0  # Reduced Planck constant
BOLTZMANN: float = 1.0  # Boltzmann constant

_C_CM_PER_FS = 2.998  # speed of light factor in (1e-5 * cm/fs)
_TWOPI = 2 * np.pi


def convert_cm_to_fs(value: float) -> float:
    """
    Convert the wavenumber-frequencies from cm^-1 to angular frequency fs^-1

    Parameters:
        value (float): Value in cm^-1

    Returns:
        float: Value in fs^-1
    """
    return value * _C_CM_PER_FS * _TWOPI * 1e-5


def convert_fs_to_cm(value):
    """
    Convert angular frequency fs^-1 to wavenumber-frequencies cm^-1

    Parameters:
        value (float): Value in fs^-1

    Returns:
        float: Value in cm^-1
    """
    return value / (_C_CM_PER_FS * _TWOPI * 1e-5)


__all__ = ["HBAR", "BOLTZMANN", "convert_cm_to_fs", "convert_fs_to_cm"]
