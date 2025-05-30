import numpy as np
from qutip import Qobj, Result
from src.core.system_parameters import SystemParameters
from src.core.pulse_sequences import PulseSequence
from src.core.functions_with_rwa import apply_RWA_phase_factors


def normalized_gauss(x_vals: np.ndarray, FWHM: float, mu: float = 0.0) -> np.ndarray:
    """
    Compute the normalized Gaussian function σ(x_vals - mu) with given FWHM.

    Parameters
    ----------
    x_vals : np.ndarray
        Energy value(s) at which to evaluate σ(x_vals - mu).
    FWHM : float
        Full Width at Half Maximum (FWHM) of the Gaussian.
    mu : float, optional
        Center energy (default: 0.0).

    Returns
    -------
    np.ndarray
        The value(s) of σ(x_vals - mu) at x_vals.

    Notes
    -----
    The function is normalized such that
        ∫σ(x_vals - mu) dE = 1
    for all FWHM.
    """
    # =============================
    # Compute normalized Gaussian
    # =============================
    sigma_val = FWHM / (2 * np.sqrt(2 * np.log(2)))  # standard deviation from FWHM
    norm = 1.0 / (sigma_val * np.sqrt(2 * np.pi))  # normalization factor
    exponent = -0.5 * ((x_vals - mu) / sigma_val) ** 2  # Gaussian exponent

    return norm * np.exp(exponent)


def sample_from_sigma(
    n_samples: int, FWHM: float, mu: float, max_detuning: float = 10.0
) -> np.ndarray:
    """
    Sample n_samples values from the normalized σ(x_vals) distribution using rejection sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    FWHM : float
        Full Width at Half Maximum (FWHM) of the Gaussian.
    mu : float
        Center energy of the distribution.
    max_detuning : float, optional
        Range (in units of FWHM) to sample from around mu (default: 10).

    Returns
    -------
    np.ndarray
        Array of sampled energy values.
    """
    # =============================
    # Special case: FWHM = 0 (no inhomogeneity)
    # =============================
    if FWHM == 0 or np.isclose(FWHM, 0):
        # Return an array of just mu since this represents a FWHM function at mu
        return np.array([mu])

    # =============================
    # Define the sampling range and maximum
    # =============================
    E_min = mu - max_detuning * FWHM
    E_max = mu + max_detuning * FWHM
    E_vals = np.linspace(E_min, E_max, 10000)
    sigma_vals = normalized_gauss(E_vals, FWHM, mu)
    sigma_max = np.max(sigma_vals)

    # =============================
    # Rejection sampling
    # =============================
    samples = []
    while len(samples) < n_samples:
        E_try = np.random.uniform(E_min, E_max)
        y_try = np.random.uniform(0, sigma_max)
        if y_try < normalized_gauss(E_try, FWHM, mu):
            samples.append(E_try)
    return np.array(samples)
