import numpy as np


def convert_cm_to_fs(value):
    """
    Convert the wavenumber-frequencies from cm^-1 to angular frequency fs^-1

    Parameters:
        value (float): Value in cm^-1

    Returns:
        float: Value in fs^-1
    """
    return value * 2.998 * 2 * np.pi * 10**-5


def convert_fs_to_cm(value):
    """
    Convert angular frequency fs^-1 to wavenumber-frequencies cm^-1

    Parameters:
        value (float): Value in fs^-1

    Returns:
        float: Value in cm^-1
    """
    return value / (2.998 * 2 * np.pi * 10**-5)


# =============================
# Fundamental constants
# =============================
HBAR = 1.0
BOLTZMANN = 1.0
