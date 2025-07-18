import numpy as np
from qutip.utilities import n_thermal

"""

This module contains functions for calculating spectral density and power spectrum
for various types of bosonic baths, including Drude-Lorentz and ohmic baths.

"""


def spectral_density_func_drude_lorentz(w, args):
    """
    Spectral density function for a Drude-Lorentz bath.
    Compatible with scalar and array inputs.
    """
    alpha = args["alpha"]
    cutoff = args["cutoff"]
    lambda_ = alpha * cutoff / 2  # Reorganization energy (coupling strength)
    gamma = cutoff  # Drude decay rate (cutoff frequency)
    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)
    result = (2 * lambda_ * gamma * w) / (w**2 + gamma**2)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result


def power_spectrum_func_drude_lorentz(w, args):
    """
    power spectrum function in the frequency domain for an drude lorentzian bath.
    Handles both positive and negative frequencies, compatible with arrays.
    """
    temp = args["temp"]

    Boltzmann = args["Boltzmann"] if "Boltzmann" in args else 1.0
    hbar = args["hbar"] if "hbar" in args else 1.0

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)

    # Avoid division by zero in tanh
    w_safe = np.where(w == 0, 1e-10, w)
    w_th = Boltzmann * temp / hbar  # Thermal energy in frequency units
    coth_term = 1 / np.tanh(w_safe / (2 * w_th))

    result = (
        np.sign(w)
        * spectral_density_func_drude_lorentz(np.abs(w), args)
        * (coth_term + 1)
    )

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result


def spectral_density_func_ohmic(w, args):
    """
    Spectral density function for an ohmic bath.
    Compatible with scalar and array inputs.
    Qutip handles w as angular frequency!
    """
    wc = args["cutoff"]
    alpha = args["alpha"]
    s = args["s"] if "s" in args else 1.0  # Default to ohmic (s=1)

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)
    result = alpha * w**s / wc ** (s - 1) * np.exp(-w / wc) * (w > 0)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result


def power_spectrum_func_ohmic(w, args):
    """
    power spectrum function in the frequency domain for an ohmic bath.
    Handles both positive and negative frequencies, compatible with arrays.
    """
    temp = args["temp"]

    Boltzmann = args["Boltzmann"] if "Boltzmann" in args else 1.0
    hbar = args["hbar"] if "hbar" in args else 1.0

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)

    # Avoid division by zero in tanh
    w_safe = np.where(w == 0, 1e-10, w)
    w_th = Boltzmann * temp / hbar  # Thermal energy in frequency units
    coth_term = 1 / np.tanh(w_safe / (2 * w_th))

    result = np.sign(w) * spectral_density_func_ohmic(np.abs(w), args) * (coth_term + 1)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result


# =============================
# BATH FUNCTIONS as defined in the paper
# =============================
def spectral_density_func_paper(w, args):
    """
    Spectral density function for a bath as given in the paper.
    Compatible with scalar and array inputs.
    """
    alpha = args["alpha"]
    cutoff = args["cutoff"]

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)

    result = alpha * (w / cutoff) * np.exp(-w / cutoff) * (w > 0)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result


def power_spectrum_func_paper(w, args):
    """
    power spectrum function in the frequency domain as given in the paper.
    Compatible with scalar and array inputs.
    """
    # Extract constants from args
    Boltzmann = args["Boltzmann"] if "Boltzmann" in args else 1.0
    hbar = args["hbar"] if "hbar" in args else 1.0

    temp = args["temp"]
    alpha = args["alpha"]
    cutoff = args["cutoff"]

    w_th = Boltzmann * temp / hbar  # Thermal energy in frequency units

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)
    result = np.zeros_like(w)

    # Positive frequency
    pos_mask = w > 0
    neg_mask = w < 0
    result[pos_mask] = (1 + n_thermal(w[pos_mask], w_th)) * spectral_density_func_paper(
        w[pos_mask], args
    )

    # Negative frequency
    result[neg_mask] = n_thermal(-w[neg_mask], w_th) * spectral_density_func_paper(
        -w[neg_mask], args
    )

    # Zero frequency C(0)
    zero_mask = w == 0
    result[zero_mask] = alpha * w_th / cutoff
    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result
