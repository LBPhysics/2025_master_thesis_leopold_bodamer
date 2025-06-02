import numpy as np

"""

This module contains functions for calculating spectral density and power spectrum
for various types of bosonic baths, including Drude-Lorentz and ohmic baths.

"""


def spectral_density_func_drude_lorentz(w, args):
    """
    Spectral density function for a Drude-Lorentz bath.
    """
    lambda_ = args["lambda"]  # Reorganization energy (coupling strength)
    gamma = args["cutoff"]  # Drude decay rate (cutoff frequency)

    return (2 * lambda_ * gamma * w) / (w**2 + gamma**2)


def spectral_density_func_ohmic(w, args):
    """
    Spectral density function for an ohmic bath.
    """
    wc = args["cutoff"]
    eta = args["eta"]
    s = args["s"]
    return eta * w**s / wc ** (s - 1) * np.exp(-w / wc) * (w > 0)


def Power_spectrum_func_ohmic(w, args):
    """
    Power spectrum function in the frequency domain for an ohmic bath.
    Handles both positive and negative frequencies.
    """
    Boltzmann = args["Boltzmann"]
    Temp = args["Temp"]
    coth_term = 1 / np.tanh(w / (2 * Boltzmann * Temp))
    return np.sign(w) * spectral_density_func_ohmic(np.abs(w), args) * (coth_term + 1)


# =============================
# BATH FUNCTIONS
# =============================


def n(w, Boltzmann, hbar, Temp):
    """
    Bose-Einstein distribution function for scalar inputs.
    """
    if w == 0:
        return 0  # Avoid division by zero for w == 0

    # Avoid overflow for large values of hbar * w / (Boltzmann * Temp)
    if (hbar * w / (Boltzmann * Temp)) > 700:
        return 0  # Approximate the result as 0 for large values

    # Compute the Bose-Einstein distribution
    exp_term = np.exp(hbar * w / (Boltzmann * Temp))
    return 1 / (exp_term - 1)


def spectral_density_func_paper(w, args):
    """
    Spectral density function for a bath as given in the paper.
    """
    g = args["g"]
    cutoff = args["cutoff"]
    return g**2 * (w / cutoff) * np.exp(-w / cutoff) * (w > 0)


def Power_spectrum_func_paper(w, args):
    """
    Power spectrum function in the frequency domain as given in the paper.
    Handles only float inputs for w.
    """
    # Extract constants from args
    Boltzmann = args["Boltzmann"]
    hbar = args["hbar"]
    Temp = args["Temp"]
    g = args["g"]
    cutoff = args["cutoff"]

    if w > 0:
        # Positive frequency
        return spectral_density_func_paper(w, args) * n(w, Boltzmann, hbar, Temp)
    elif w < 0:
        # Negative frequency
        return spectral_density_func_paper(-w, args) * (
            1 + n(-w, Boltzmann, hbar, Temp)
        )
    else:
        # Zero frequency
        return g**2 * Boltzmann * Temp / cutoff


def Power_spectrum_func_paper_array(w_array, args):
    """
    Wrapper for Power_spectrum_func_paper to handle array-like inputs.
    Calls Power_spectrum_func_paper for each element in the array.

    Parameters:
        w_array (array-like): Array of frequency values.
        args (dict): Arguments for the power spectrum function.

    Returns:
        np.ndarray: Array of power spectrum values.
    """
    w_array = np.asarray(w_array, dtype=float)  # Ensure input is a NumPy array
    return np.array([Power_spectrum_func_paper(w, args) for w in w_array])
