from qspectro2d.core.pulse_functions import *
from qspectro2d.visualization.plotting import Plot_2d_El_field
from qspectro2d.spectroscopy.calculations import (
    get_tau_cohs_and_t_dets_for_T_wait,
)
from qutip import basis, ket2dm, tensor, Qobj, mesolve, brmesolve, expect
import numpy as np


# ##########################
# functions for post-processing and plotting
# ##########################
def extend_time_tau_axes(
    ts: np.ndarray,
    taus: np.ndarray,
    data: np.ndarray,
    pad_rows: tuple[float, float] = (1, 1),
    pad_cols: tuple[float, float] = (1, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extend the time and tau axes by padding and correspondingly extend the data array.

    This function takes 2D spectroscopy data and extends both time axes by padding
    with zeros. The padding is specified as multipliers, where values >= 1 indicate
    how much to extend each axis. Data types are preserved during the operation.

    Parameters
    ----------
    ts : np.ndarray
        Time axis for detection (t_det). Shape: (N_t,)
    taus : np.ndarray
        Time axis for coherence (tau_coh). Shape: (N_tau,)
    data : np.ndarray
        2D spectroscopy data array. Shape: (N_tau, N_t)
        Data type is preserved in the output.
    pad_rows : tuple[float, float], default=(1, 1)
        Padding multipliers for rows (taus axis) as (before, after).
        Values >= 1, where 1 means no padding, 2 means add original length, etc.
    pad_cols : tuple[float, float], default=(1, 1)
        Padding multipliers for columns (ts axis) as (before, after).
        Values >= 1, where 1 means no padding, 2 means add original length, etc.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        extended_ts : np.ndarray
            Extended time axis for detection. Same dtype as input ts.
        extended_taus : np.ndarray
            Extended time axis for coherence. Same dtype as input taus.
        padded_data : np.ndarray
            Zero-padded data array. Same dtype as input data.

    Raises
    ------
    ValueError
        If any padding multiplier is < 1.

    Examples
    --------
    >>> ts = np.linspace(0, 100, 51)  # 0 to 100 fs, 51 points
    >>> taus = np.linspace(0, 50, 26)  # 0 to 50 fs, 26 points
    >>> data = np.random.rand(26, 51)
    >>> ext_ts, ext_taus, ext_data = extend_time_tau_axes(ts, taus, data,
    ...                                                   pad_rows=(2, 2),
    ...                                                   pad_cols=(1, 3))
    >>> # Result: taus axis doubled on both sides, ts axis tripled after
    """
    # Validate input multipliers
    if any(val < 1 for val in pad_rows + pad_cols):
        raise ValueError("All padding multipliers must be >= 1")

    # Convert multipliers to actual padding values
    original_rows, original_cols = data.shape
    pad_rows_actual = (
        int((pad_rows[0] - 1) * original_rows),  # before - convert to int for np.pad
        int((pad_rows[1] - 1) * original_rows),  # after - convert to int for np.pad
    )
    pad_cols_actual = (
        int((pad_cols[0] - 1) * original_cols),  # before - convert to int for np.pad
        int((pad_cols[1] - 1) * original_cols),  # after - convert to int for np.pad
    )

    # Pad the data array
    padded_data = np.pad(
        data, (pad_rows_actual, pad_cols_actual), mode="constant", constant_values=0
    )

    # Compute steps
    dt = ts[1] - ts[0]
    dtau = taus[1] - taus[0]

    # Extend axes
    extended_ts = np.linspace(
        ts[0] - pad_cols_actual[0] * dt,
        ts[-1] + pad_cols_actual[1] * dt,
        padded_data.shape[1],
    )
    extended_taus = np.linspace(
        taus[0] - pad_rows_actual[0] * dtau,
        taus[-1] + pad_rows_actual[1] * dtau,
        padded_data.shape[0],
    )

    return extended_ts, extended_taus, padded_data


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


def extend_and_plot_results(
    averaged_results: list[np.ndarray],
    times_T: np.ndarray,
    times: np.ndarray,
    extend_for: tuple[float, float] = None,
    **plot_args_freq: dict,
) -> None:
    """
    Extend and plot the results for a set of 2D spectra averaged over phase/inhomogeneous broadening.

    Parameters
    ----------
    averaged_results : list of np.ndarray
        List of 2D arrays (each shape: [len(taus), len(ts)]) for each T_wait.
        Can contain None values for invalid T_wait values.
    times_T : np.ndarray
        Array of T_wait values.
    times : np.ndarray
        Time grid used for simulation.
    extend_for : tuple[float, float], optional
        Padding for (rows, columns) as (before, after) for both axes.
    **plot_args_freq : dict
        Additional keyword arguments for frequency-domain plotting.

    Returns
    -------
    None
    """
    # Filter out None values from averaged_results
    valid_results = [res for res in averaged_results if res is not None]
    valid_T_waits = [
        times_T[i] for i, res in enumerate(averaged_results) if res is not None
    ]

    if not valid_results:
        print("No valid results to plot")
        return

    # Find the first valid T_wait index to initialize global arrays
    first_valid_idx = next(
        (i for i, res in enumerate(averaged_results) if res is not None), None
    )
    if first_valid_idx is None:
        print("No valid results found in averaged_results")
        return

    # =============================
    # Combine all data arrays into global arrays for time and frequency domains
    # =============================
    # Initialize global arrays with zeros
    # global_ts and global_taus are the largest axes (from the first valid T_wait)

    global_ts, global_taus = get_tau_cohs_and_t_dets_for_T_wait(
        times, times_T[first_valid_idx]
    )
    global_data_time = np.zeros((len(global_taus), len(global_ts)), dtype=np.complex64)

    if extend_for is not None:
        global_ts, global_taus, global_data_time = extend_time_tau_axes(
            global_ts,
            global_taus,
            global_data_time,
            pad_rows=extend_for,
            pad_cols=extend_for,
        )

    global_nu_ts, global_nu_taus, global_data_freq = compute_2d_fft_wavenumber(
        global_ts, global_taus, global_data_time
    )

    for i, data in enumerate(valid_results):
        T_wait = valid_T_waits[i]
        ts, taus = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait)

        if extend_for is not None:
            ts, taus, data = extend_time_tau_axes(
                ts, taus, data, pad_rows=extend_for, pad_cols=extend_for
            )

        nu_ts, nu_taus, data_freq = compute_2d_fft_wavenumber(ts, taus, data)

        # Map local data into the global arrays with safe index mapping
        tau_indices = np.array([np.argmin(np.abs(global_taus - v)) for v in taus])
        t_indices = np.array([np.argmin(np.abs(global_ts - v)) for v in ts])
        nu_tau_indices = np.array(
            [np.argmin(np.abs(global_nu_taus - v)) for v in nu_taus]
        )
        nu_t_indices = np.array([np.argmin(np.abs(global_nu_ts - v)) for v in nu_ts])

        # Map time domain data using time indices
        for local_tau_idx, global_tau_idx in enumerate(tau_indices):
            for local_t_idx, global_t_idx in enumerate(t_indices):
                # Safe assignment with verified bounds for time domain
                global_data_time[global_tau_idx, global_t_idx] += data[
                    local_tau_idx, local_t_idx
                ]

        # Map frequency domain data using frequency indices
        for local_nu_tau_idx, global_nu_tau_idx in enumerate(nu_tau_indices):
            for local_nu_t_idx, global_nu_t_idx in enumerate(nu_t_indices):
                # Safe assignment with verified bounds for frequency domain
                global_data_freq[global_nu_tau_idx, global_nu_t_idx] += data_freq[
                    local_nu_tau_idx, local_nu_t_idx
                ]

        """if len(times_T) > 1:
            Plot_2d_El_field(
                (ts, taus, data),
                times_T[i],
                use_custom_colormap=True,
                save=True,  # CHANGE TO False for no plotting the Time domain
                output_dir=plot_args_freq.get("output_dir", None),
                system=plot_args_freq.get("system", None),
            )

            Plot_2d_El_field(
                (nu_ts, nu_taus, data_freq), times_T[i], **plot_args_freq
            )"""

    # Normalize by number of valid results
    if len(valid_results) > 0:
        global_data_time /= len(valid_results)
        global_data_freq /= len(valid_results)

    # Plot the global results
    """
    Plot_2d_El_field(
        (global_ts, global_taus, global_data_time),
        save=True,  # CHANGE TO False for no plotting the Time domain
        output_dir=plot_args_freq.get("output_dir", None),
        system=plot_args_freq.get("system", None),
        use_custom_colormap=True,
    )
    """

    Plot_2d_El_field(
        (global_nu_ts, global_nu_taus, global_data_freq),
        **plot_args_freq,
    )
