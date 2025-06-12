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


def power_spectrum_func_drude_lorentz(w, args):  # TODO CHECK, its just a guess
    """
    power spectrum (symmetrized correlation function) for a Drude-Lorentz bath.

    Parameters:
        w : float or ndarray
            Frequency (can be scalar or array).
        args : dict
            Dictionary containing:
                - "alpha": Reorganization energy (coupling strength)
                - "cutoff": Drude decay rate (cutoff frequency)
                - "beta": Inverse temperature (1 / (k_B * T))

    Returns:
        float or ndarray: power spectrum value(s)
    """
    alpha = args["alpha"]
    cutoff = args["cutoff"]
    Boltzmann = args["Boltzmann"]
    Temp = args["Temp"]
    hbar = args["hbar"]
    beta = 1 / (Boltzmann * Temp)  # Inverse temperature

    w_input = w
    w = np.asarray(w, dtype=float)

    J = (2 * alpha * cutoff * w) / (w**2 + cutoff**2)
    S = (1 / np.pi) * J * np.tanh(0.5 * beta * w)

    # Handle zero frequency (avoid division by zero in coth)
    S = np.where(w == 0, (1 / np.pi) * (2 * alpha * cutoff / cutoff) * (2 / beta), S)

    return float(S) if np.isscalar(w_input) else S


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


def power_spectrum_func_ohmic(w, args):
    """
    power spectrum function in the frequency domain for an ohmic bath.
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
    alpha = args["alpha"]
    cutoff = args["cutoff"]

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)

    result = alpha**2 * (w / cutoff) * np.exp(-w / cutoff) * (w > 0)

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
    Boltzmann = args["Boltzmann"]
    hbar = args["hbar"]
    Temp = args["Temp"]
    alpha = args["alpha"]
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
    result[zero_mask] = alpha**2 * Boltzmann * Temp / (cutoff * hbar)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return 2 * result
