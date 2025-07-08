import numpy as np


def normalized_gauss(x_vals: np.ndarray, fwhm: float, mu: float = 0.0) -> np.ndarray:
    """
    Compute the normalized Gaussian function σ(x_vals - mu) with given fwhm.

    Parameters
    ----------
    x_vals : np.ndarray
        Energy value(s) at which to evaluate σ(x_vals - mu).
    fwhm : float
        Full Width at Half Maximum (fwhm) of the Gaussian.
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
    for all fwhm.
    """
    # =============================
    # Compute normalized Gaussian
    # =============================
    sigma_val = fwhm / (2 * np.sqrt(2 * np.log(2)))  # standard deviation from fwhm
    norm = 1.0 / (sigma_val * np.sqrt(2 * np.pi))  # normalization factor
    exponent = -0.5 * ((x_vals - mu) / sigma_val) ** 2  # Gaussian exponent

    return norm * np.exp(exponent)


def sample_from_gaussian(
    n_samples: int, fwhm: float, mu: float, max_detuning: float = 10.0
) -> np.ndarray:
    """
    Sample n_samples values from the normalized σ(x_vals) distribution using rejection sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    fwhm : float
        Full Width at Half Maximum (fwhm) of the Gaussian.
    mu : float
        Center energy of the distribution.
    max_detuning : float, optional
        Range (in units of fwhm) to sample from around mu (default: 10).

    Returns
    -------
    np.ndarray
        Array of sampled energy values.
    """
    # =============================
    # Special case: fwhm = 0 (no inhomogeneity)
    # =============================
    if fwhm == 0 or np.isclose(fwhm, 0):
        # Return an array of just mu since this represents a fwhm function at mu
        return np.array([mu])

    # =============================
    # Define the sampling range and maximum
    # =============================
    E_min = mu - max_detuning * fwhm
    E_max = mu + max_detuning * fwhm
    E_vals = np.linspace(E_min, E_max, 10000)
    sigma_vals = normalized_gauss(E_vals, fwhm, mu)
    sigma_max = np.max(sigma_vals)

    # =============================
    # Rejection sampling
    # =============================
    samples = []
    while len(samples) < n_samples:
        E_try = np.random.uniform(E_min, E_max)
        y_try = np.random.uniform(0, sigma_max)
        if y_try < normalized_gauss(E_try, fwhm, mu):
            samples.append(E_try)
    return np.array(samples)
