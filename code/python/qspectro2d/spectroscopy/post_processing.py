from qspectro2d.core.pulse_functions import *
import numpy as np


# ##########################
# functions for post-processing and plotting
# ##########################
def extend_time_axes(
    data: np.ndarray,
    t_det: np.ndarray,
    tau_coh: np.ndarray = None,
    pad_t_det: tuple[float, float] = (1, 1),
    pad_tau_coh: tuple[float, float] = (1, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]:
    """
    Extend time axes by padding for both 1D and 2D spectroscopy data.

    This function handles both 1D and 2D cases automatically based on input dimensions.
    For 1D: extends t_det axis only. For 2D: extends both tau_coh and t_det axes.
    The padding is specified as multipliers, where values >= 1 indicate how much
    to extend each axis. Data types are preserved during the operation.

    Parameters
    ----------
    data : np.ndarray
        Spectroscopy data array.
        - 1D case: Shape (N_t,) - data along t_det axis
        - 2D case: Shape (N_tau, N_t) - data along (tau_coh, t_det) axes
    t_det : np.ndarray
        Time axis for detection. Shape: (N_t,)
    tau_coh : np.ndarray, optional
        Time axis for coherence. Shape: (N_tau,). Required for 2D data.
    pad_t_det : tuple[float, float], default=(1, 1)
        Padding multipliers for t_det axis as (before, after).
        Values >= 1, where 1 means no padding, 2 means add original length, etc.
    pad_tau_coh : tuple[float, float], default=(1, 1)
        Padding multipliers for tau_coh axis as (before, after).
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
            extended_tau_coh : np.ndarray
                Extended time axis for coherence.
            padded_data : np.ndarray
                Zero-padded data array.

    Raises
    ------
    ValueError
        If any padding multiplier is < 1, or if 2D data provided without tau_coh.

    Examples
    --------
    # 1D case
    >>> t_det = np.linspace(0, 100, 51)  # 0 to 100 fs, 51 points
    >>> data_1d = np.random.rand(51)
    >>> ext_t_det, ext_data = extend_time_axes(data_1d, t_det, pad_t_det=(1, 2))
    >>> # Result: t_det axis doubled after

    # 2D case
    >>> t_det = np.linspace(0, 100, 51)  # 0 to 100 fs, 51 points
    >>> tau_coh = np.linspace(0, 50, 26)  # 0 to 50 fs, 26 points
    >>> data_2d = np.random.rand(26, 51)
    >>> ext_t_det, ext_tau_coh, ext_data = extend_time_axes(
    ...     data_2d, t_det, tau_coh, pad_t_det=(1, 3), pad_tau_coh=(2, 2)
    ... )
    >>> # Result: tau_coh axis doubled on both sides, t_det axis tripled after
    """
    # Validate input multipliers
    if any(val < 1 for val in pad_t_det + pad_tau_coh):
        raise ValueError("All padding multipliers must be >= 1")

    # Determine if we're dealing with 1D or 2D data
    is_2d = data.ndim == 2

    if is_2d:
        if tau_coh is None:
            raise ValueError("tau_coh must be provided for 2D data")

        # 2D case: data shape is (N_tau, N_t)
        original_tau_points, original_t_points = data.shape

        # Convert multipliers to actual padding values
        pad_tau_actual = (
            int((pad_tau_coh[0] - 1) * original_tau_points),
            int((pad_tau_coh[1] - 1) * original_tau_points),
        )
        pad_t_actual = (
            int((pad_t_det[0] - 1) * original_t_points),
            int((pad_t_det[1] - 1) * original_t_points),
        )

        # Pad the data array (rows=tau_coh, cols=t_det)
        padded_data = np.pad(
            data, (pad_tau_actual, pad_t_actual), mode="constant", constant_values=0
        )

        # Compute steps
        dt = t_det[1] - t_det[0]
        dtau = tau_coh[1] - tau_coh[0]

        # Extend axes
        extended_t_det = np.linspace(
            t_det[0] - pad_t_actual[0] * dt,
            t_det[-1] + pad_t_actual[1] * dt,
            padded_data.shape[1],
        )
        extended_tau_coh = np.linspace(
            tau_coh[0] - pad_tau_actual[0] * dtau,
            tau_coh[-1] + pad_tau_actual[1] * dtau,
            padded_data.shape[0],
        )

        return extended_t_det, extended_tau_coh, padded_data

    else:
        # 1D case: data shape is (N_t,)
        original_t_points = data.shape[0]

        # Convert multipliers to actual padding values
        pad_t_actual = (
            int((pad_t_det[0] - 1) * original_t_points),
            int((pad_t_det[1] - 1) * original_t_points),
        )

        # Pad the data array
        padded_data = np.pad(data, pad_t_actual, mode="constant", constant_values=0)

        # Compute step
        dt = t_det[1] - t_det[0]

        # Extend axis
        extended_t_det = np.linspace(
            t_det[0] - pad_t_actual[0] * dt,
            t_det[-1] + pad_t_actual[1] * dt,
            padded_data.shape[0],
        )

        return extended_t_det, padded_data


def compute_1d_fft_wavenumber(
    ts: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D real FFT of spectroscopy data and convert frequency axis to wavenumber units.

    This function performs a 1D real-valued FFT on the input data and converts the
    resulting frequency axis from cycles/fs to wavenumber units (10^4 cm⁻¹). The
    output spectrum is multiplied by 1j to account for the relationship E ~ i*P
    between electric field and polarization.

    Parameters
    ----------
    ts : np.ndarray
        Time axis for detection (t_det) in femtoseconds. Shape: (N_t,)
        Must be evenly spaced for accurate FFT.
    data : np.ndarray
        1D spectroscopy data array, typically real-valued polarization.
        Shape: (N_t,). Data should be real for rfft to be appropriate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        nu_ts : np.ndarray
            Wavenumber axis for detection in units of 10^4 cm⁻¹.
            Shape: (N_t//2 + 1,) due to rfft.
        s1d : np.ndarray
            1D FFT spectrum with dtype np.complex64.
            Shape: (N_t//2 + 1,).
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
    >>> ts = np.linspace(0, 100, 101)  # 0-100 fs, dt = 1 fs
    >>> data = np.random.rand(101)  # Real polarization data
    >>> nu_ts, spectrum = compute_1d_fft_wavenumber(ts, data)
    >>> # nu_ts in 10^4 cm⁻¹, spectrum is complex
    """
    # Calculate sampling rates and perform FFT
    dt = ts[1] - ts[0]  # Sampling interval in fs
    N_t = len(ts)

    # Full FFT with shift (similar to 2D implementation)
    s1d = np.fft.fft(data)
    s1d = np.fft.fftshift(s1d)
    freq_t = np.fft.fftshift(np.fft.fftfreq(N_t, d=dt))

    # Convert to wavenumber (10^4 cm^-1)
    # Speed of light: c ≈ 2.998 × 10^8 m/s = 2.998 × 10^-5 cm/fs
    # Wavenumber = frequency / c, scaled by 10^4
    nu_ts = freq_t / 2.998 * 10

    return nu_ts, s1d


def compute_2d_fft_wavenumber(
    ts: np.ndarray, taus: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D real FFT of spectroscopy data and convert frequency axes to wavenumber units.

    This function performs a 2D real-valued FFT on the input data and converts the
    resulting frequency axes from cycles/fs to wavenumber units (10^4 cm⁻¹). The
    output spectrum is multiplied by 1j to account for the relationship E ~ i*P
    between electric field and polarization.

    Parameters
    ----------
    ts : np.ndarray
        Time axis for detection (t_det) in femtoseconds. Shape: (N_t,)
        Must be evenly spaced for accurate FFT.
    taus : np.ndarray
        Time axis for coherence (tau_coh) in femtoseconds. Shape: (N_tau,)
        Must be evenly spaced for accurate FFT.
    data : np.ndarray
        2D spectroscopy data array, typically real-valued polarization.
        Shape: (N_tau, N_t). Data should be real for rfft2 to be appropriate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        nu_ts : np.ndarray
            Wavenumber axis for detection in units of 10^4 cm⁻¹.
            Shape: (N_t//2 + 1,) due to rfft.
        nu_taus : np.ndarray
            Wavenumber axis for coherence in units of 10^4 cm⁻¹.
            Shape: (N_tau//2 + 1,) due to rfft.
        s2d : np.ndarray
            2D FFT spectrum with dtype np.complex64.
            Shape: (N_tau//2 + 1, N_t//2 + 1).
            Includes factor of 1j to represent E ~ i*P relationship.

    Notes
    -----
    - Uses np.fft.rfft2() which assumes real input data and returns only positive frequencies
    - Conversion factor 2.998 * 10 converts from cycles/fs to 10^4 cm⁻¹:
      * Speed of light c ≈ 2.998 × 10^8 m/s = 2.998 × 10^-4 cm/fs
      * Wavenumber = frequency / c, scaled by 10^4
    - The 1j factor accounts for the physical relationship between electric field and polarization

    Examples
    --------
    >>> ts = np.linspace(0, 100, 101)  # 0-100 fs, dt = 1 fs
    >>> taus = np.linspace(0, 50, 51)  # 0-50 fs, dtau = 1 fs
    >>> data = np.random.rand(51, 101)  # Real polarization data
    >>> nu_ts, nu_taus, spectrum = compute_2d_fft_wavenumber(ts, taus, data)
    >>> # nu_ts and nu_taus in 10^4 cm⁻¹, spectrum is complex
    """

    # Calculate only the frequency axes (cycle/fs)
    dτ = taus[1] - taus[0]
    dt = ts[1] - ts[0]
    tfreqs = np.fft.fftfreq(len(ts), d=dt)
    taufreqs = -np.fft.fftfreq(len(taus), d=dτ)

    # Optional: Shift zero-frequency component to center (only along taus)
    # s2d = np.fft.rfft2(data)  # axis 0 (tau) → fft, axis 1 (t) → rfft
    s2d = np.fft.fft(data, axis=1)  # axis 1 (t) → rfft
    s2d = np.fft.ifft(s2d, axis=0) * len(taus)  # axis 0 (tau) → ifft
    s2d = np.fft.fftshift(s2d, axes=1)
    s2d = np.fft.ifftshift(s2d, axes=0)
    taufreqs = np.fft.ifftshift(taufreqs)
    tfreqs = np.fft.ifftshift(tfreqs)
    data_freq = s2d

    # Convert to wavenumber units [10^4 cm⁻¹]
    nu_taus = taufreqs / 2.998 * 10
    nu_ts = tfreqs / 2.998 * 10

    return (
        nu_ts,
        nu_taus,
        data_freq,
    )
