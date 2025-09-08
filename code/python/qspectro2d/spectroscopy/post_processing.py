from itertools import count
from qspectro2d.core.laser_system.laser_fcts import *
import numpy as np


# ##########################
# functions for post-processing and plotting
# ##########################
def extend_time_axes(
    data: np.ndarray,
    t_det: np.ndarray,
    t_coh: np.ndarray = None,
    pad_t_det: tuple[float, float] = (1, 1),
    pad_t_coh: tuple[float, float] = (1, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]:
    """
    Extend time axes by padding for both 1D and 2D spectroscopy data.

    This function handles both 1D and 2D cases automatically based on input dimensions.
    For 1D: extends t_det axis only. For 2D: extends both t_coh and t_det axes.
    The padding is specified as multipliers, where values >= 1 indicate how much
    to extend each axis. Data types are preserved during the operation.

    Parameters
    ----------
    data : np.ndarray
        Spectroscopy data array.
        - 1D case: Shape (N_t_det,) - data along t_det axis
        - 2D case: Shape (N_t_coh, N_t_det) - data along (t_coh, t_det) axes
    t_det : np.ndarray
        Time axis for detection. Shape: (N_t_det,)
    t_coh : np.ndarray, optional
        Time axis for coherence. Shape: (N_t_coh,). Required for 2D data.
    pad_t_det : tuple[float, float], default=(1, 1)
        Padding multipliers for t_det axis as (before, after).
        Values >= 1, where 1 means no padding, 2 means add original length, etc.
    pad_t_coh : tuple[float, float], default=(1, 1)
        Padding multipliers for t_coh axis as (before, after).
        Only used for 2D data. Values >= 1.

    Returns
    -------
    For 1D data:
        tuple[np.ndarray, np.ndarray]
            extended_t_det : np.ndarray
                Extended time axis for detection.
            padded_data : np.ndarray
                Zero-padded data array.

    For 2D data:
        tuple[np.ndarray, np.ndarray, np.ndarray]
            extended_t_det : np.ndarray
                Extended time axis for detection.
            extended_t_coh : np.ndarray
                Extended time axis for coherence.
            padded_data : np.ndarray
                Zero-padded data array.

    Raises
    ------
    ValueError
        If any padding multiplier is < 1, or if 2D data provided without t_coh.

    Examples
    --------
    # 1D case
    >>> t_det = np.linspace(0, 100, 51)  # 0 to 100 fs, 51 points
    >>> data_1d = np.random.rand(51)
    >>> ext_t_det, ext_data = extend_time_axes(data_1d, t_det, pad_t_det=(1, 2))
    >>> # Result: t_det axis doubled after

    # 2D case
    >>> t_det = np.linspace(0, 100, 51)  # 0 to 100 fs, 51 points
    >>> t_coh = np.linspace(0, 50, 26)  # 0 to 50 fs, 26 points
    >>> data_2d = np.random.rand(26, 51)
    >>> ext_t_det, ext_t_coh, ext_data = extend_time_axes(
    ...     data_2d, t_det, t_coh, pad_t_det=(1, 3), pad_t_coh=(2, 2)
    ... )
    >>> # Result: t_coh axis doubled on both sides, t_det axis tripled after
    """
    # Validate input multipliers
    if any(val < 1 for val in pad_t_det + pad_t_coh):
        raise ValueError("All padding multipliers must be >= 1")

    # Determine if we're dealing with 1D or 2D data
    is_2d = data.ndim == 2

    if is_2d:
        if t_coh is None:
            raise ValueError("t_coh must be provided for 2D data")

        # 2D case: data shape is (N_t_coh, N_t_det)
        original_t_coh_points, original_t_points = data.shape

        # Convert multipliers to actual padding values
        pad_t_coh_actual = (
            int((pad_t_coh[0] - 1) * original_t_coh_points),
            int((pad_t_coh[1] - 1) * original_t_coh_points),
        )
        pad_t_actual = (
            int((pad_t_det[0] - 1) * original_t_points),
            int((pad_t_det[1] - 1) * original_t_points),
        )

        # Pad the data array (rows=t_coh, cols=t_det)
        padded_data = np.pad(
            data, (pad_t_coh_actual, pad_t_actual), mode="constant", constant_values=0
        )

        # Compute steps
        dt_det = t_det[1] - t_det[0]
        dt_coh = t_coh[1] - t_coh[0]

        # Extend axes
        extended_t_det = np.linspace(
            t_det[0] - pad_t_actual[0] * dt_det,
            t_det[-1] + pad_t_actual[1] * dt_det,
            padded_data.shape[1],
        )
        extended_t_coh = np.linspace(
            t_coh[0] - pad_t_coh_actual[0] * dt_coh,
            t_coh[-1] + pad_t_coh_actual[1] * dt_coh,
            padded_data.shape[0],
        )

        return extended_t_det, extended_t_coh, padded_data

    else:
        # 1D case: data shape is (N_t_det,)
        original_t_points = data.shape[0]

        # Convert multipliers to actual padding values
        pad_t_actual = (
            int((pad_t_det[0] - 1) * original_t_points),
            int((pad_t_det[1] - 1) * original_t_points),
        )

        # Pad the data array
        padded_data = np.pad(data, pad_t_actual, mode="constant", constant_values=0)

        # Compute step
        dt_det = t_det[1] - t_det[0]

        # Extend axis
        extended_t_det = np.linspace(
            t_det[0] - pad_t_actual[0] * dt_det,
            t_det[-1] + pad_t_actual[1] * dt_det,
            padded_data.shape[0],
        )

        return extended_t_det, padded_data


def compute_1d_fft_wavenumber(
    t_dets: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D real FFT of spectroscopy data and convert frequency axis to wavenumber units.

    This function performs a 1D real-valued FFT on the input data and converts the
    resulting frequency axis from cycles/fs to wavenumber units (10^4 cm⁻¹). The
    output spectrum is multiplied by 1j to account for the relationship E ~ i*P
    between electric field and polarization.

    Parameters
    ----------
    t_dets : np.ndarray
        Time axis for detection (t_det) in femtoseconds. Shape: (N_t_det,)
        Must be evenly spaced for accurate FFT.
    data : np.ndarray
        1D spectroscopy data array, typically real-valued polarization.
        Shape: (N_t_det,). Data should be real for rfft to be appropriate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        nu_dets : np.ndarray
            Wavenumber axis for detection in units of 10^4 cm⁻¹.
            Shape: (N_t_det//2 + 1,) due to rfft.
        s1d : np.ndarray
            1D FFT spectrum with dtype np.complex64.
            Shape: (N_t_det//2 + 1,).
            Includes factor of 1j to represent E ~ i*P relationship.

    Notes
    -----
    - Uses np.fft.rfft() which assumes real input data and returns only positive frequencies
    - Conversion factor 2.998 * 10 converts from cycles/fs to 10^4 cm⁻¹:
      * Speed of light c ≈ 2.998 × 10^8 m/s = 2.998 × 10^-4 cm/fs
      * Wavenumber = frequency / c, scaled by 10^4
    - The 1j factor accounts for the physical relationship between electric field and polarization

    Examples
    --------
    >>> t_dets = np.linspace(0, 100, 101)  # 0-100 fs, dt_det = 1 fs
    >>> data = np.random.rand(101)  # Real polarization data
    >>> nu_dets, spectrum = compute_1d_fft_wavenumber(t_dets, data)
    >>> # nu_dets in 10^4 cm⁻¹, spectrum is complex
    """
    # Calculate sampling rates and perform FFT
    dt_det = t_dets[1] - t_dets[0]  # Sampling interval in fs
    N_t_det = len(t_dets)

    # Full FFT with shift (similar to 2D implementation)
    s1d = np.fft.fft(data)
    s1d = np.fft.fftshift(s1d)
    freq_t = np.fft.fftshift(np.fft.fftfreq(N_t_det, d=dt_det))

    # Convert to wavenumber (10^4 cm^-1)
    # Speed of light: c ≈ 2.998 × 10^8 m/s = 2.998 × 10^-5 cm/fs
    # Wavenumber = frequency / c, scaled by 10^4
    nu_dets = freq_t / 2.998 * 10

    return nu_dets, s1d


def compute_2d_fft_wavenumber(
    t_dets: np.ndarray,
    t_cohs: np.ndarray,
    data: np.ndarray | tuple[np.ndarray, np.ndarray],
    signal: str = "rephasing",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D FFT and return a single spectrum (always one ndarray output).

    Inputs:
        data :
            - single 2D array (N_t_coh, N_t_det)
            - OR tuple (time_domain_rephasing, time_domain_nonrephasing)
        signal : {'rephasing','non-rephasing','absorptive'}
            Meaning when tuple provided:
                rephasing      -> return only rephasing spectrum
                non-rephasing  -> return only non-rephasing spectrum
                absorptive     -> return real((S_re + S_nr)/2) (average of the two)
            For single array input behaves as before (absorptive combines internally).

    Returns:
        (nu_dets, nu_cohs, S) with S shape (N_t_coh, N_t_det).
    """
    dt_coh = t_cohs[1] - t_cohs[0]
    dt_det = t_dets[1] - t_dets[0]
    N_coh = len(t_cohs)
    N_det = len(t_dets)

    def _validate(arr: np.ndarray, label: str):
        if arr.ndim != 2:
            raise ValueError(f"{label} must be 2D (N_t_coh, N_t_det)")
        if arr.shape != (N_coh, N_det):
            raise ValueError(
                f"{label} shape {arr.shape} != ({N_coh}, {N_det}) from provided axes"
            )

    def _fft2(arr: np.ndarray, kind: str) -> np.ndarray:
        tmp_local = np.fft.fft(arr, axis=0)
        tmp_local = np.fft.fft(tmp_local, axis=1)
        if kind == "rephasing":
            spec = np.flip(tmp_local, axis=0)
        elif kind == "non-rephasing":
            spec = np.flip(np.flip(tmp_local, axis=0), axis=1)
        else:
            raise ValueError(f"Internal kind '{kind}' unsupported")
        return spec * (dt_coh * dt_det)

    allowed = {"rephasing", "non-rephasing", "absorptive"}
    if signal not in allowed:
        raise ValueError(f"Invalid signal '{signal}'. Allowed: {sorted(allowed)}")

    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError("Tuple data must have exactly two elements")
        re_t, nr_t = data
        _validate(re_t, "Tuple element 0 (rephasing)")
        _validate(nr_t, "Tuple element 1 (non-rephasing)")
        S_re = _fft2(re_t, "rephasing")
        S_nr = _fft2(nr_t, "non-rephasing")
        if signal == "rephasing":
            S = S_re
        elif signal == "non-rephasing":
            S = S_nr
        else:  # absorptive average
            S = np.real((S_re + S_nr) / 2.0)
    else:
        _validate(data, "data")
        tmp = np.fft.fft(data, axis=0)
        tmp = np.fft.fft(tmp, axis=1)
        if signal == "rephasing":
            S = np.flip(tmp, axis=0)
        elif signal == "non-rephasing":
            S = np.flip(np.flip(tmp, axis=0), axis=1)

    # Frequency axes & shift
    freq_cohs = np.fft.fftfreq(N_coh, d=dt_coh)
    freq_dets = np.fft.fftfreq(N_det, d=dt_det)
    S = np.fft.fftshift(S, axes=(0, 1))
    freq_cohs = np.fft.fftshift(freq_cohs)
    freq_dets = np.fft.fftshift(freq_dets)

    nu_cohs = freq_cohs / 2.998 * 10
    nu_dets = freq_dets / 2.998 * 10
    return nu_dets, nu_cohs, S
