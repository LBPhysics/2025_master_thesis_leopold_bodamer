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
    lambda_ = args["lambda"]  # Reorganization energy (coupling strength)
    gamma = args["cutoff"]  # Drude decay rate (cutoff frequency)

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)
    result = (2 * lambda_ * gamma * w) / (w**2 + gamma**2)

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
    s = args["s"]

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)
    result = alpha * w**s / wc ** (s - 1) * np.exp(-w / wc) * (w > 0)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result


def Power_spectrum_func_ohmic(w, args):
    """
    Power spectrum function in the frequency domain for an ohmic bath.
    Handles both positive and negative frequencies, compatible with arrays.
    """
    Boltzmann = args["Boltzmann"]
    Temp = args["Temp"]
    hbar = args["hbar"]

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)

    # Avoid division by zero in tanh
    w_safe = np.where(w == 0, 1e-10, w)
    w_th = Boltzmann * Temp / hbar  # Thermal energy in frequency units
    coth_term = 1 / np.tanh(w_safe / (2 * w_th))

    result = np.sign(w) * spectral_density_func_ohmic(np.abs(w), args) * (coth_term + 1)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result


# =============================
# BATH FUNCTIONS defined in the paper
# =============================
def spectral_density_func_paper(w, args):
    """
    Spectral density function for a bath as given in the paper.
    Compatible with scalar and array inputs.
    """
    g = args["g"]
    cutoff = args["cutoff"]

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)

    result = g**2 * (w / cutoff) * np.exp(-w / cutoff) * (w > 0)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result


def Power_spectrum_func_paper(w, args):
    """
    Power spectrum function in the frequency domain as given in the paper.
    Compatible with scalar and array inputs.
    """
    # Extract constants from args
    Boltzmann = args["Boltzmann"]
    hbar = args["hbar"]
    Temp = args["Temp"]
    g = args["g"]
    cutoff = args["cutoff"]

    w_th = Boltzmann * Temp / hbar  # Thermal energy in frequency units

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)
    result = np.zeros_like(w)

    # Positive frequency
    pos_mask = w > 0
    neg_mask = w < 0
    result[pos_mask] = spectral_density_func_paper(w[pos_mask], args) * (
        1 + n_thermal(w[pos_mask], w_th)
    )

    # Negative frequency
    result[neg_mask] = spectral_density_func_paper(-w[neg_mask], args) * n_thermal(
        -w[neg_mask], w_th
    )

    # Zero frequency C(0)
    zero_mask = w == 0
    result[zero_mask] = g**2 * Boltzmann * Temp / (cutoff * hbar)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return 2 * result
