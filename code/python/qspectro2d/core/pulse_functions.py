from qspectro2d.core.pulse_sequences import PulseSequence
from typing import List, Tuple, Union
import numpy as np


def pulse_envelope(
    t: Union[float, np.ndarray], pulse_seq: PulseSequence
) -> Union[float, np.ndarray]:
    """
    Calculate the combined envelope of multiple pulses at time t using PulseSequence.
    Works with both scalar and array time inputs.

    Now uses pulse_peak_time as t_peak (peak time) where cos²/gaussian is maximal.
    Pulse is zero outside [t_peak - fwhm, t_peak + fwhm] == outside of 2 fwhm.

    Uses the pulse_type from each pulse to determine which envelope function to use:
    - 'cos2': cosine squared envelope
    - 'gaussian': Gaussian envelope, shifted so that:
      - The Gaussian is zero at t_peak ± fwhm boundaries: (actually about <= 1%)

    Args:
        t (Union[float, np.ndarray]): Time value or array of time values
        pulse_seq (PulseSequence): The pulse sequence

    Returns:
        Union[float, np.ndarray]: Combined envelope value(s)
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    # Handle array input
    if isinstance(t, np.ndarray):
        result = np.zeros_like(t, dtype=float)
        for i in range(len(t)):
            result[i] = pulse_envelope(float(t[i]), pulse_seq)
        return result

    # Handle scalar input (original functionality)
    envelope = 0.0
    for pulse in pulse_seq.pulses:
        t_peak = pulse.pulse_peak_time
        fwhm = pulse.pulse_fwhm
        pulse_type = getattr(
            pulse, "pulse_type", "cos2"
        )  # Default to cos2 for backward compatibility

        if fwhm is None or fwhm <= 0:
            continue
        if t_peak is None:
            continue

        # Pulse exists only in [t_peak - fwhm, t_peak + fwhm]
        if not (t_peak - fwhm <= t <= t_peak + fwhm):
            continue

        if pulse_type == "cos2":
            ### Cosine squared envelope
            arg = np.pi * (t - t_peak) / (2 * fwhm)
            envelope += np.cos(arg) ** 2

        elif pulse_type == "gaussian":
            ### Gaussian envelope
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            gaussian_val = np.exp(-((t - t_peak) ** 2) / (2 * sigma**2))
            boundary_distance_sq = fwhm**2
            boundary_val = np.exp(-boundary_distance_sq / (2 * sigma**2))
            envelope += (
                max(0.0, gaussian_val) - boundary_val
            )  # effect of this (dis)continuity -> minimal

        else:
            raise ValueError(
                f"Unknown pulse_type: {pulse_type}. Use 'cos2' or 'gaussian'."
            )

    return envelope


def E_pulse(
    t: Union[float, np.ndarray], pulse_seq: PulseSequence
) -> Union[complex, np.ndarray]:
    """
    Calculate the total electric field at time t for a set of pulses (envelope only, no carrier), using PulseSequence.
    Works with both scalar and array time inputs.

    Args:
        t (Union[float, np.ndarray]): Time value or array of time values
        pulse_seq (PulseSequence): The pulse sequence

    Returns:
        Union[complex, np.ndarray]: Electric field value(s)
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    # Handle array input
    if isinstance(t, np.ndarray):
        result = np.zeros_like(t, dtype=complex)
        for i in range(len(t)):
            result[i] = E_pulse(float(t[i]), pulse_seq)
        return result

    # Handle scalar input (original functionality)
    E_total = 0.0 + 0.0j
    for pulse in pulse_seq.pulses:
        phi = pulse.pulse_phase
        E0 = pulse.pulse_amplitude
        if phi is None or E0 is None:
            continue
        envelope = pulse_envelope(
            t, PulseSequence([pulse])
        )  # use pulse_envelope for each pulse
        E_total += E0 * envelope * np.exp(-1j * phi)
    return E_total


def Epsilon_pulse(
    t: Union[float, np.ndarray], pulse_seq: PulseSequence
) -> Union[complex, np.ndarray]:
    """
    Calculate the total electric field at time t for a set of pulses, including carrier oscillation, using PulseSequence.
    Works with both scalar and array time inputs.

    Args:
        t (Union[float, np.ndarray]): Time value or array of time values
        pulse_seq (PulseSequence): The pulse sequence

    Returns:
        Union[complex, np.ndarray]: Electric field with carrier value(s)
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    # Handle array input
    if isinstance(t, np.ndarray):
        result = np.zeros_like(t, dtype=complex)
        for i in range(len(t)):
            result[i] = Epsilon_pulse(float(t[i]), pulse_seq)
        return result

    # Handle scalar input (original functionality)
    E_total = 0.0 + 0.0j
    for pulse in pulse_seq.pulses:
        omega = pulse.pulse_freq
        if omega is None:
            continue
        E_field = E_pulse(t, PulseSequence([pulse]))  # use E_pulse for each pulse
        E_total += E_field * np.exp(-1j * (omega * t))
    return E_total


def identify_non_zero_pulse_regions(
    times: np.ndarray, pulse_seq: PulseSequence
) -> np.ndarray:
    """
    Identify regions where the pulse envelope is non-zero across an array of time values.

    Args:
        times (np.ndarray): Array of time values to evaluate
        pulse_seq (PulseSequence): The pulse sequence to evaluate

    Returns:
        np.ndarray: Boolean array where True indicates times where envelope is non-zero
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    # Initialize an array of all False values
    active_regions = np.zeros_like(times, dtype=bool)

    # For each time point, check if it's in the active region of any pulse
    for i, t in enumerate(times):
        # A time is in an active region if any pulse contributes to the envelope
        for pulse in pulse_seq.pulses:
            t_peak = pulse.pulse_peak_time
            fwhm = pulse.pulse_fwhm

            # Skip pulses with invalid parameters
            if fwhm is None or fwhm <= 0 or t_peak is None:
                continue

            # Check if time point falls within the pulse's active region
            if t_peak - fwhm <= t <= t_peak + fwhm:
                active_regions[i] = True
                break  # Once we know this time point is active, we can move to the next

    return active_regions


def split_by_active_regions(times, active_regions):
    # Find where the active_regions changes value
    change_indices = np.where(np.diff(active_regions.astype(int)) != 0)[0] + 1

    # Split the times at those change points
    split_times = np.split(times, change_indices)
    assert (
        np.concatenate(split_times).size == times.size
    ), "Split times do not match original times size."
    # Return list of (times, state) tuples
    return [times for times in split_times]
