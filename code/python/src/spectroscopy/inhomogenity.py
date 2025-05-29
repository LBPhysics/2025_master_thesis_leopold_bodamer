import numpy as np
from qutip import Qobj, Result
from src.core.system_parameters import SystemParameters
from src.core.pulse_sequences import PulseSequence
from src.core.functions_with_rwa import apply_RWA_phase_factors


def normalized_gauss(x_vals: np.ndarray, FWHM: float, x_0: float = 0.0) -> np.ndarray:
    """
    Compute the normalized Gaussian function σ(x_vals - x_0) with given FWHM.

    Parameters
    ----------
    x_vals : np.ndarray
        Energy value(s) at which to evaluate σ(x_vals - x_0).
    FWHM : float
        Full Width at Half Maximum (FWHM) of the Gaussian.
    x_0 : float, optional
        Center energy (default: 0.0).

    Returns
    -------
    np.ndarray
        The value(s) of σ(x_vals - x_0) at x_vals.

    Notes
    -----
    The function is normalized such that
        ∫σ(x_vals - x_0) dE = 1
    for all FWHM.
    """
    # =============================
    # Compute normalized Gaussian
    # =============================
    sigma_val = FWHM / (2 * np.sqrt(2 * np.log(2)))  # standard deviation from FWHM
    norm = 1.0 / (sigma_val * np.sqrt(2 * np.pi))  # normalization factor
    exponent = -0.5 * ((x_vals - x_0) / sigma_val) ** 2  # Gaussian exponent

    return norm * np.exp(exponent)


def sample_from_sigma(
    n_samples: int, FWHM: float, x_0: float, E_range: float = 10.0
) -> np.ndarray:
    """
    Sample n_samples values from the normalized σ(x_vals) distribution using rejection sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    FWHM : float
        Full Width at Half Maximum (FWHM) of the Gaussian.
    x_0 : float
        Center energy of the distribution.
    E_range : float, optional
        Range (in units of FWHM) to sample from around x_0 (default: 10).

    Returns
    -------
    np.ndarray
        Array of sampled energy values.
    """
    # =============================
    # Special case: FWHM = 0 (no inhomogeneity)
    # =============================
    if FWHM == 0 or np.isclose(FWHM, 0):
        # Return an array of just x_0 since this represents a FWHM function at x_0
        return np.array([x_0])

    # =============================
    # Define the sampling range and maximum
    # =============================
    E_min = x_0 - E_range * FWHM
    E_max = x_0 + E_range * FWHM
    E_vals = np.linspace(E_min, E_max, 10000)
    sigma_vals = normalized_gauss(E_vals, FWHM, x_0)
    sigma_max = np.max(sigma_vals)

    # =============================
    # Rejection sampling
    # =============================
    samples = []
    while len(samples) < n_samples:
        E_try = np.random.uniform(E_min, E_max)
        y_try = np.random.uniform(0, sigma_max)
        if y_try < normalized_gauss(E_try, FWHM, x_0):
            samples.append(E_try)
    return np.array(samples)
