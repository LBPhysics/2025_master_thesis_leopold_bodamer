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
    t_dets: np.ndarray, t_cohs: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D FFT of spectroscopy data and convert frequency axes to wavenumber units.

    This function performs a 2D FFT on the input data following the formula:
    S2D(ω_coh,T, ω_det) ~ ∫dt exp(-iω_det*t) × ∫dcoh exp(iω_coh*coh) E_ks(coh,T,t)

    The t-axis uses forward FFT (exp(-iω_det*t)) and coh-axis uses inverse FFT (exp(+iω_coh*coh)).

    Parameters
    ----------
    t_dets : np.ndarray
        Time axis for detection (t_det) in femtoseconds. Shape: (N_t_det,)
        Must be evenly spaced for accurate FFT.
    t_cohs : np.ndarray
        Time axis for coherence (t_coh/coh) in femtoseconds. Shape: (N_t_coh,)
        Must be evenly spaced for accurate FFT.
    data : np.ndarray
        2D spectroscopy data array E_ks(coh,T,t).
        Shape: (N_t_coh, N_t_det).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        nu_dets : np.ndarray
            Wavenumber axis for detection in units of 10^4 cm⁻¹.
            Shape: (N_t_det,).
        nu_cohs : np.ndarray
            Wavenumber axis for coherence in units of 10^4 cm⁻¹.
            Shape: (N_t_coh,).
        s2d : np.ndarray
            2D FFT spectrum.
            Shape: (N_t_coh, N_t_det).
    """
    ### Calculate time steps
    dt_coh = t_cohs[1] - t_cohs[0]  # coh step
    dt_det = t_dets[1] - t_dets[0]  # t step

    ### Perform 2D FFT according to the formula
    # Forward FFT along t-axis (axis=1): exp(-iω_det*t)
    # Inverse FFT along coh-axis (axis=0): exp(+iω_coh*coh)
    # s2d = np.fft.fft(data, axis=0)  # * len(t_cohs)
    # s2d = np.fft.fft(s2d, axis=1)

    s2d = np.fft.fft2(data)

    ### Generate frequency axes
    freq_dets = np.fft.fftfreq(len(t_dets), d=dt_det)
    freq_cohs = np.fft.fftfreq(len(t_cohs), d=dt_coh)

    ### Apply frequency shifts for centered display
    # s2d = np.fft.fftshift(s2d, axes=(0))
    # s2d = np.fft.fftshift(s2d, axes=(1))
    # freq_cohs = np.fft.fftshift(freq_cohs)
    # freq_dets = np.fft.fftshift(freq_dets)

    ### Convert to wavenumber units [10^4 cm⁻¹]
    # Speed of light: c ≈ 2.998 × 10^-4 cm/fs
    nu_dets = freq_dets / 2.998 * 10
    nu_cohs = freq_cohs / 2.998 * 10

    return nu_dets, nu_cohs, s2d
