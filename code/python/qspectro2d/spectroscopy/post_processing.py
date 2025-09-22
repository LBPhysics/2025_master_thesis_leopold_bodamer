# TODO redo this module
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
) -> tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    S(T_wait, w_det) = ∫ E(T_wait, t_det) e^{-i w_det t_det} dt_det
    Compute 1D FFT of spectroscopy data and convert frequency axis to wavenumber units.

    This function performs a 1D FFT on the input data and converts the
    resulting frequency axis from cycles/fs to wavenumber units (10^4 cm⁻¹). The
    implementation follows the same convention as the 2D case using full FFT with fftshift.

    Parameters
    ----------
    t_dets : np.ndarray
        Time axis for detection (t_det) in femtoseconds. Shape: (N_t_det,)
    datas : List[np.ndarray] (N_t_det,) time-domain E-field signal components
        data for the requested signal_types. For absorptive spectra, pass both components
        [rephasing, nonrephasing].
    signal_types : List[str], default=["rephasing"]
        List of signal labels corresponding one-to-one with `datas`.
        Allowed normalized values per entry: "rephasing" or "nonrephasing".
        For absorptive: provide signal_types=["rephasing", "nonrephasing"].

    Returns
    -------
    tuple[np.ndarray, List[np.ndarray], List[str]]
        nu_dets : np.ndarray
        spectra : List[np.ndarray]
            List of 1D FFT spectra in frequency space (fftshifted), scaled by dt_det.
            Contains entries for requested components in deterministic order:
            [rephasing?, nonrephasing?, absorptive?, average?].
        labels : List[str]
            Labels corresponding to each spectrum entry (e.g., 'rephasing', 'absorptive').

    Notes
    -----
    - Conversion factor 2.998 * 10 converts from cycles/fs to 10^4 cm⁻¹:
      * Speed of light c ≈ 2.998 × 10^8 m/s = 2.998 × 10^-4 cm/fs
      * Wavenumber = frequency / c, scaled by 10^4
    - Results are fftshifted to center zero frequency
    - Absorptive spectrum is computed as real((S_re + S_nr)/2) when both components
      are provided.
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
        spec = np.fft.ifft(arr)
        spec = np.fft.ifftshift(spec)
        return spec * dt_det  # Scale by time step for consistency with 2D case

    if not datas:
        raise ValueError("'datas' must contain at least one 1D array")
    if len(datas) != len(signal_types):
        raise ValueError(
            f"Length mismatch: len(datas)={len(datas)} vs len(signal_types)={len(signal_types)}"
        )

    # Validate all arrays and compute individual spectra
    # Compute spectra dictionary by type for first occurrence
    S_by_type: dict[str, np.ndarray] = {}
    for idx, (data, sig) in enumerate(zip(datas, signal_types)):
        _validate(data, f"datas[{idx}]")
        sig_norm = sig.lower()
        if sig_norm not in ("rephasing", "nonrephasing", "average"):
            # Allow unknown tags but still compute their FFT and store under the raw name
            S_by_type.setdefault(sig_norm, _fft1d(data))
            continue
        if sig_norm not in S_by_type:
            S_by_type[sig_norm] = _fft1d(data)

    uniq = set(map(str.lower, signal_types))

    # Assemble outputs in deterministic order
    spectra: List[np.ndarray] = []
    labels: List[str] = []
    if "rephasing" in uniq and "rephasing" in S_by_type:
        spectra.append(S_by_type["rephasing"])
        labels.append("rephasing")
    if "nonrephasing" in uniq and "nonrephasing" in S_by_type:
        spectra.append(S_by_type["nonrephasing"])
        labels.append("nonrephasing")
    if ("rephasing" in uniq and "nonrephasing" in uniq) and (
        "rephasing" in S_by_type and "nonrephasing" in S_by_type
    ):
        spectra.append(np.real((S_by_type["rephasing"] + S_by_type["nonrephasing"]) / 2.0))
        labels.append("absorptive")
    if "average" in uniq and "average" in S_by_type:
        spectra.append(S_by_type["average"])
        labels.append("average")

    # Include any additional custom tags (not one of the above) if present
    for k, v in S_by_type.items():
        if k not in ("rephasing", "nonrephasing", "average"):
            spectra.append(v)
            labels.append(k)

    # Generate frequency axis and shift
    freq_dets = np.fft.fftfreq(N_t_det, d=dt_det)
    freq_dets = np.fft.ifftshift(freq_dets)

    nu_dets = freq_dets / 2.998 * 10

    return nu_dets, spectra, labels


def compute_2d_fft_wavenumber(
    t_dets: np.ndarray,
    t_cohs: np.ndarray,
    datas: List[np.ndarray],
    signal_types: List[str] = ["rephasing"],
) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[str]]:
    """Compute 2D FFT S(w_coh, T_wait, w_det) = ∫ E(t_coh, T_wait, t_det) e^{SIGN_DET i w_det t_det + SIGN_COH i w_coh t_coh} dt_det dt_coh
    The sign depends on how each pulse interacts with the system / which freq component couples
    - Rephasing: SIGN_DET = -1, SIGN_COH = +1
    - otherwise I put: SIGN_DET = -1, SIGN_COH = -1

    Parameters
    ----------
    t_dets : Detection time (N_t_det,)
    t_cohs : Coherence time (N_t_coh,)
    datas : List[np.ndarray]
        shape (N_t_coh, N_t_det)
        datas for the requested signal_types.
        For absorptive: real((S_re + S_nr)/2)
    signal_types : List[str], default=["rephasing"]
        List of signal labels corresponding one-to-one with `datas`.
        Allowed normalized values per entry: "rephasing" or "nonrephasing".

    Returns
    -------
    (nu_dets, nu_cohs, spectra, labels)
        nu_dets, nu_cohs : np.ndarray (wavenumber axis, 10^4 cm^-1)
        spectra : List[np.ndarray]
            List of complex spectra with shape (N_t_coh, N_t_det) in deterministic order:
            [rephasing?, nonrephasing?, absorptive?, average?] (fftshifted, scaled).
        labels : List[str]
            Labels corresponding to each spectrum entry.
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

    def _fft2(arr: np.ndarray, signal_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D transform and provide matching frequency axes.

        Returns
        -------
        spec : complex ndarray (N_coh, N_det)
            2D spectrum scaled by (dt_coh * dt_det).
        freq_cohs_ax : ndarray (N_coh,)
            Frequency axis for coherence dimension (cycles/fs), centered (fftshifted).
        freq_dets_ax : ndarray (N_det,)
            Frequency axis for detection dimension (cycles/fs), centered (fftshifted).
        """

        sig = signal_type.lower()

        if sig == "rephasing":
            # coh: fft (- sign), det: ifft (+ sign)
            tmp_local = np.fft.fft(arr, axis=0)
            spec = np.fft.ifft(tmp_local, axis=1) * N_det

        else:
            # nonrephasing (+, +): ifft both
            tmp_local = np.fft.ifft(arr, axis=0) * N_coh
            spec = np.fft.ifft(tmp_local, axis=1) * N_det

        # Always fftshift for plotting
        spec = np.fft.fftshift(spec, axes=(0, 1))
        freq_cohs_ax = np.fft.fftshift(np.fft.fftfreq(N_coh, d=dt_coh))
        freq_dets_ax = np.fft.fftshift(np.fft.fftfreq(N_det, d=dt_det))

        return spec * (dt_coh * dt_det), freq_cohs_ax, freq_dets_ax

    if not datas:
        raise ValueError("'datas' must contain at least one 2D array")
    if len(datas) != len(signal_types):
        raise ValueError(
            f"Length mismatch: len(datas)={len(datas)} vs len(signal_types)={len(signal_types)}"
        )

    # Validate all arrays and compute individual spectra
    # Compute spectra dictionary by type for first occurrence
    S_by_type: dict[str, np.ndarray] = {}
    axes_by_type: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for idx, (data, sig) in enumerate(zip(datas, signal_types)):
        _validate(data, f"datas[{idx}]")
        sig_norm = sig.lower()
        if sig_norm not in S_by_type:
            spec, f_coh_ax, f_det_ax = _fft2(data, sig_norm)
            S_by_type[sig_norm] = spec
            axes_by_type[sig_norm] = (f_coh_ax, f_det_ax)

    uniq = set(map(str.lower, signal_types))

    # Frequency axes & shift (legacy behavior retained for public API)
    freq_cohs = np.fft.fftshift(np.fft.fftfreq(N_coh, d=dt_coh))
    freq_dets = np.fft.fftshift(np.fft.fftfreq(N_det, d=dt_det))

    nu_cohs = freq_cohs / 2.998 * 10
    nu_dets = freq_dets / 2.998 * 10

    # Assemble outputs in deterministic order and fftshift each
    spectra_2d: List[np.ndarray] = []
    labels_2d: List[str] = []

    if "rephasing" in uniq and "rephasing" in S_by_type:
        spectra_2d.append(np.fft.fftshift(S_by_type["rephasing"], axes=(0, 1)))
        labels_2d.append("rephasing")
    if "nonrephasing" in uniq and "nonrephasing" in S_by_type:
        spectra_2d.append(np.fft.fftshift(S_by_type["nonrephasing"], axes=(0, 1)))
        labels_2d.append("nonrephasing")
    if ("rephasing" in uniq and "nonrephasing" in uniq) and (
        "rephasing" in S_by_type and "nonrephasing" in S_by_type
    ):
        absorptive = np.real((S_by_type["rephasing"] + S_by_type["nonrephasing"]) / 2.0)
        spectra_2d.append(np.fft.fftshift(absorptive, axes=(0, 1)))
        labels_2d.append("absorptive")
    if "average" in uniq and "average" in S_by_type:
        spectra_2d.append(np.fft.fftshift(S_by_type["average"], axes=(0, 1)))
        labels_2d.append("average")

    # Include any additional custom tags (not one of the above) if present
    for k, v in S_by_type.items():
        if k not in ("rephasing", "nonrephasing", "average"):
            spectra_2d.append(np.fft.fftshift(v, axes=(0, 1)))
            labels_2d.append(k)

    return nu_dets, nu_cohs, spectra_2d, labels_2d
