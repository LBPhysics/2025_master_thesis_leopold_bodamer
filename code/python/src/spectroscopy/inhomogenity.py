import numpy as np
from qutip import Qobj, Result
from src.core.system_parameters import SystemParameters
from src.core.pulse_sequences import PulseSequence
from src.core.functions_with_rwa import apply_RWA_phase_factors


def sigma(E: np.ndarray, Delta: float, E0: float = 0.0) -> np.ndarray:
    """
    Compute the normalized Gaussian function σ(E - E0) with given FWHM.

    Parameters
    ----------
    E : np.ndarray
        Energy value(s) at which to evaluate σ(E - E0).
    Delta : float
        Full Width at Half Maximum (FWHM) of the Gaussian.
    E0 : float, optional
        Center energy (default: 0.0).

    Returns
    -------
    np.ndarray
        The value(s) of σ(E - E0) at E.

    Notes
    -----
    The function is normalized such that
        ∫σ(E - E0) dE = 1
    for all Delta.
    """
    # =============================
    # Compute normalized Gaussian
    # =============================
    ln2 = np.log(2)  # natural logarithm of 2
    sigma_val = Delta / (2 * np.sqrt(2 * ln2))  # standard deviation from FWHM
    norm = 1.0 / (sigma_val * np.sqrt(2 * np.pi))  # normalization factor
    exponent = -0.5 * ((E - E0) / sigma_val) ** 2  # Gaussian exponent

    return norm * np.exp(exponent)


def sample_from_sigma(
    N: int, Delta: float, E0: float, E_range: float = 10.0
) -> np.ndarray:
    """
    Sample N values from the normalized σ(E) distribution using rejection sampling.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    Delta : float
        Full Width at Half Maximum (FWHM) of the Gaussian.
    E0 : float
        Center energy of the distribution.
    E_range : float, optional
        Range (in units of Delta) to sample from around E0 (default: 10).

    Returns
    -------
    np.ndarray
        Array of sampled energy values.
    """
    # =============================
    # Special case: Delta = 0 (no inhomogeneity)
    # =============================
    if Delta == 0 or np.isclose(Delta, 0):
        # Return an array of just E0 since this represents a delta function at E0
        return np.array([E0])

    # =============================
    # Define the sampling range and maximum
    # =============================
    E_min = E0 - E_range * Delta
    E_max = E0 + E_range * Delta
    E_vals = np.linspace(E_min, E_max, 10000)
    sigma_vals = sigma(E_vals, Delta, E0)
    sigma_max = np.max(sigma_vals)

    # =============================
    # Rejection sampling
    # =============================
    samples = []
    while len(samples) < N:
        E_try = np.random.uniform(E_min, E_max)
        y_try = np.random.uniform(0, sigma_max)
        if y_try < sigma(E_try, Delta, E0):
            samples.append(E_try)
    return np.array(samples)
