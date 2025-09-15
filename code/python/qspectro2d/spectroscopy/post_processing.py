from typing import List
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
    t_dets: np.ndarray,
    datas: List[np.ndarray],
    signal_types: List[str] = ["rephasing"],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D FFT of spectroscopy data and convert frequency axis to wavenumber units.

    This function performs a 1D FFT on the input data and converts the
    resulting frequency axis from cycles/fs to wavenumber units (10^4 cm⁻¹). The
    implementation follows the same convention as the 2D case using full FFT with fftshift.

    Parameters
    ----------
    t_dets : np.ndarray
        Time axis for detection (t_det) in femtoseconds. Shape: (N_t_det,)
        Must be evenly spaced for accurate FFT.
    datas : List[np.ndarray]
        One or two 1D arrays with shape (N_t_det,), typically time-domain
        data for the requested signal_types. For absorptive spectra, pass both components
        [rephasing, nonrephasing].
    signal_types : List[str], default=["rephasing"]
        List of signal labels corresponding one-to-one with `datas`.
        Allowed normalized values per entry: "rephasing" or "nonrephasing".
        For absorptive: provide signal_types=["rephasing", "nonrephasing"].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        nu_dets : np.ndarray
            Wavenumber axis for detection in units of 10^4 cm⁻¹.
            Shape: (N_t_det,) - full spectrum including negative frequencies.
        s1d : np.ndarray
            1D FFT spectrum with complex values.
            Shape: (N_t_det,).
            Scaled by dt_det for consistency with 2D implementation.

    Notes
    -----
    - Uses np.fft.fft() for full complex FFT, consistent with 2D implementation
    - Conversion factor 2.998 * 10 converts from cycles/fs to 10^4 cm⁻¹:
      * Speed of light c ≈ 2.998 × 10^8 m/s = 2.998 × 10^-4 cm/fs
      * Wavenumber = frequency / c, scaled by 10^4
    - Results are fftshifted to center zero frequency
    - Absorptive spectrum is computed as real((S_re + S_nr)/2) when both components
      are provided.

    Examples
    --------
    >>> t_dets = np.linspace(0, 100, 101)  # 0-100 fs, dt_det = 1 fs
    >>> data = [np.random.rand(101) + 1j*np.random.rand(101)]  # Single component
    >>> nu_dets, spectrum = compute_1d_fft_wavenumber(t_dets, data, ["rephasing"])

    >>> # Absorptive spectrum with both components
    >>> data_re = np.random.rand(101) + 1j*np.random.rand(101)
    >>> data_nr = np.random.rand(101) + 1j*np.random.rand(101)
    >>> nu_dets, spectrum = compute_1d_fft_wavenumber(
    ...     t_dets, [data_re, data_nr], ["rephasing", "nonrephasing"]
    ... )
    """
    # Calculate sampling parameters
    dt_det = t_dets[1] - t_dets[0]  # Sampling interval in fs
    N_t_det = len(t_dets)

    def _validate(arr: np.ndarray, label: str):
        if arr.ndim != 1:
            raise ValueError(f"{label} must be 1D (N_t_det,)")
        if arr.shape != (N_t_det,):
            raise ValueError(f"{label} shape {arr.shape} != ({N_t_det},) from provided t_dets axis")

    def _fft1d(arr: np.ndarray) -> np.ndarray:
        """Compute 1D FFT with scaling and shift."""
        spec = np.fft.fft(arr)
        spec = np.fft.fftshift(spec)
        return spec * dt_det  # Scale by time step for consistency with 2D case

    if not datas:
        raise ValueError("'datas' must contain at least one 1D array")

    # Validate all arrays and compute individual spectra
    S_list: list[np.ndarray] = []
    for idx, data in enumerate(datas):
        _validate(data, f"datas[{idx}]")
        S_list.append(_fft1d(data))

    uniq = set(signal_types)
    if uniq == {"rephasing"}:
        s1d_out = S_list[0]
    elif uniq == {"nonrephasing"}:
        s1d_out = S_list[0]
    elif uniq == {"rephasing", "nonrephasing"}:
        # Find first occurrence of each component
        try:
            S_re = next(S for S, s in zip(S_list, signal_types) if s == "rephasing")
            S_nr = next(S for S, s in zip(S_list, signal_types) if s == "nonrephasing")
        except StopIteration:
            raise ValueError("Both rephasing and nonrephasing must be provided for absorptive")
        s1d_out = np.real((S_re + S_nr) / 2.0)
    else:
        raise ValueError(f"Unsupported combination of signal_types: {sorted(list(uniq))}")

    # Generate frequency axis and shift
    freq_dets = np.fft.fftfreq(N_t_det, d=dt_det)
    freq_dets = np.fft.fftshift(freq_dets)

    # Convert to wavenumber (10^4 cm^-1)
    # Speed of light: c ≈ 2.998 × 10^8 m/s = 2.998 × 10^-4 cm/fs
    # Wavenumber = frequency / c, scaled by 10^4
    nu_dets = freq_dets / 2.998 * 10

    return nu_dets, s1d_out


def compute_2d_fft_wavenumber(
    t_dets: np.ndarray,
    t_cohs: np.ndarray,
    datas: List[np.ndarray],
    signal_types: List[str] = ["rephasing"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D FFT and return a single spectrum (always one ndarray output).

    Parameters
    ----------
    t_dets : np.ndarray
        Detection time axis (N_t_det,)
    t_cohs : np.ndarray
        Coherence time axis (N_t_coh,)
    datas : List[np.ndarray]
        One or two 2D arrays with shape (N_t_coh, N_t_det), typically time-domain
        data for the requested signal_types. For absorptive spectra, pass both components
        [rephasing, nonrephasing].
    signal_types : List[str], default=["rephasing"]
        List of signal labels corresponding one-to-one with `datas`.
        Allowed normalized values per entry: "rephasing" or "nonrephasing".
        For absorptive: provide signal_types=["rephasing", "nonrephasing"].

    Returns
    -------
    (nu_dets, nu_cohs, S)
        nu_dets, nu_cohs : np.ndarray (wavenumber axis, 10^4 cm^-1)
        S       : np.ndarray complex spectrum with shape (N_t_coh, N_t_det)

    Notes
    -----
    - The FFT convention follows two successive FFTs (axis=0 then axis=1) with flips
      depending on the signal_type. Results are scaled by dt_coh*dt_det and fftshifted.
    - Absorptive spectrum is computed as real((S_re + S_nr)/2) when both components
      are provided.
    """
    dt_coh = t_cohs[1] - t_cohs[0]
    dt_det = t_dets[1] - t_dets[0]
    N_coh = len(t_cohs)
    N_det = len(t_dets)

    def _validate(arr: np.ndarray, label: str):
        if arr.ndim != 2:
            raise ValueError(f"{label} must be 2D (N_t_coh, N_t_det)")
        if arr.shape != (N_coh, N_det):
            raise ValueError(f"{label} shape {arr.shape} != ({N_coh}, {N_det}) from provided axes")

    def _fft2(arr: np.ndarray, signal_type: str) -> np.ndarray:
        tmp_local = np.fft.fft(arr, axis=0)
        tmp_local = np.fft.fft(tmp_local, axis=1)
        # TODO for now just fix this
        spec = np.flip(np.flip(tmp_local, axis=0), axis=1)
        """
        if signal_type == "rephasing":  # flip coh -> +
            spec = np.flip(tmp_local, axis=0)
        elif signal_type == "nonrephasing":  # flip both -> +
            spec = np.flip(np.flip(tmp_local, axis=0), axis=1)
        elif signal_type == "average":  # no change
            spec = tmp_local
        else:
            raise ValueError(f"Internal signal_type '{signal_type}' unsupported")
        """
        return spec * (dt_coh * dt_det)

    if not datas:
        raise ValueError("'datas' must contain at least one 2D array")

    # Validate all arrays and compute individual spectra
    S_list: list[np.ndarray] = []
    for idx, (data, sig) in enumerate(zip(datas, signal_types)):
        _validate(data, f"datas[{idx}]")
        S_list.append(_fft2(data, sig))

    uniq = set(signal_types)
    if uniq == {"rephasing"} or uniq == {"average"} or uniq == {"nonrephasing"}:
        S_out = S_list[0]
    elif uniq == {"rephasing", "nonrephasing"}:
        # Find first occurrence of each component
        try:
            S_re = next(S for S, s in zip(S_list, signal_types) if s == "rephasing")
            S_nr = next(S for S, s in zip(S_list, signal_types) if s == "nonrephasing")
        except StopIteration:
            raise ValueError("Both rephasing and nonrephasing must be provided for absorptive")
        S_out = np.real((S_re + S_nr) / 2.0)
    else:
        raise ValueError(f"Unsupported combination of signal_types: {sorted(list(uniq))}")

    # Frequency axes & shift
    freq_cohs = np.fft.fftfreq(N_coh, d=dt_coh)
    freq_dets = np.fft.fftfreq(N_det, d=dt_det)
    S_out = np.fft.fftshift(S_out, axes=(0, 1))
    freq_cohs = np.fft.fftshift(freq_cohs)
    freq_dets = np.fft.fftshift(freq_dets)

    nu_cohs = freq_cohs / 2.998 * 10
    nu_dets = freq_dets / 2.998 * 10
    return nu_dets, nu_cohs, S_out
