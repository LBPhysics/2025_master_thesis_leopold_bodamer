from src.core.pulse_sequences import PulseSequence
import numpy as np


def pulse_envelope(t: float, pulse_seq: PulseSequence) -> float:
    """
    Calculate the combined envelope of multiple pulses at time t using PulseSequence.

    Now uses pulse_peak_time as t_peak (peak time) where cos² is maximal.
    Pulse is zero outside [t_peak - Delta, t_peak + Delta].
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    envelope = 0.0
    for pulse in pulse_seq.pulses:
        t_peak = pulse.pulse_peak_time  # Now interpreted as peak time
        Delta_width = pulse.pulse_half_width
        if Delta_width is None or Delta_width <= 0:
            continue
        if t_peak is None:
            continue
        # Pulse exists only in [t_peak - Delta, t_peak + Delta]
        if t_peak - Delta_width <= t <= t_peak + Delta_width:
            arg = np.pi * (t - t_peak) / (2 * Delta_width)
            envelope += np.cos(arg) ** 2
    return envelope


def pulse_envelope_gaussian(t: float, pulse_seq: PulseSequence) -> float:
    """
    Calculate the combined Gaussian envelope of multiple pulses at time t using PulseSequence.

    Gaussian envelope centered at t_peak with Delta as FWHM, shifted so that:
    - The Gaussian is zero at t_peak ± 2*Delta
    - The pulse is zero outside [t_peak - 2*Delta, t_peak + 2*Delta]

    The envelope is: exp(-(t-t_peak)^2/(2*sigma^2)) - exp(-2) for |t-t_peak| <= 2*Delta
    where sigma = Delta / (2*sqrt(2*ln(2))) to make Delta the FWHM.
    This ensures the envelope is exactly zero at the boundaries.

    Args:
        t (float): Time value
        pulse_seq (PulseSequence): The pulse sequence

    Returns:
        float: Gaussian envelope value (0 if outside pulse range)
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    envelope = 0.0
    for pulse in pulse_seq.pulses:
        t_peak = pulse.pulse_peak_time
        Delta = pulse.pulse_half_width  # Now interpreted as FWHM
        if Delta is None or Delta <= 0:
            continue
        if t_peak is None:
            continue

        # Convert FWHM to standard deviation: sigma = FWHM / (2*sqrt(2*ln(2)))
        sigma = Delta / (2 * np.sqrt(2 * np.log(2)))

        # Pulse exists only in [t_peak - 2*Delta, t_peak + 2*Delta]
        if t_peak - 2 * Delta <= t <= t_peak + 2 * Delta:
            # Gaussian with sigma derived from FWHM, shifted to be zero at boundaries
            gaussian_val = np.exp(-((t - t_peak) ** 2) / (2 * sigma**2))
            boundary_val = np.exp(-2)  # Value at t_peak ± 2*Delta

            # Shift so that envelope is zero at boundaries
            envelope += max(0, gaussian_val - boundary_val)

    return envelope


def E_pulse(t: float, pulse_seq: PulseSequence) -> complex:
    """
    Calculate the total electric field at time t for a set of pulses (envelope only, no carrier), using PulseSequence.
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

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
    return E_total  # TO make it into a cos: / 2.0


def Epsilon_pulse(t: float, pulse_seq: PulseSequence) -> complex:
    """
    Calculate the total electric field at time t for a set of pulses, including carrier oscillation, using PulseSequence.
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    E_total = 0.0 + 0.0j
    for pulse in pulse_seq.pulses:
        omega = pulse.pulse_freq
        if omega is None:
            continue
        E_field = E_pulse(t, PulseSequence([pulse]))  # use E_pulse for each pulse
        E_total += E_field * np.exp(-1j * (omega * t))
    return E_total


def E_pulse_gaussian(t: float, pulse_seq: PulseSequence) -> complex:
    """
    Calculate the total electric field at time t using Gaussian envelopes (no carrier), using PulseSequence.
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    E_total = 0.0 + 0.0j
    for pulse in pulse_seq.pulses:
        phi = pulse.pulse_phase
        E0 = pulse.pulse_amplitude
        if phi is None or E0 is None:
            continue
        envelope = pulse_envelope_gaussian(
            t, PulseSequence([pulse])
        )  # use Gaussian envelope for each pulse
        E_total += E0 * envelope * np.exp(-1j * phi)
    return E_total


def Epsilon_pulse_gaussian(t: float, pulse_seq: PulseSequence) -> complex:
    """
    Calculate the total electric field at time t using Gaussian envelopes, including carrier oscillation, using PulseSequence.
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    E_total = 0.0 + 0.0j
    for pulse in pulse_seq.pulses:
        omega = pulse.pulse_freq
        if omega is None:
            continue
        E_field = E_pulse_gaussian(
            t, PulseSequence([pulse])
        )  # use Gaussian E_pulse for each pulse
        E_total += E_field * np.exp(-1j * (omega * t))
    return E_total


def El_field_3_pulses(times: np.ndarray, pulse_seq: PulseSequence, f=pulse_envelope):
    """
    Calculate the combined electric field for a PulseSequence.

    Parameters:
        times (np.ndarray): Time range for the pulses.
        pulse_seq (PulseSequence): PulseSequence object.
        f (function): Function to compute field (pulse_envelope, E_pulse, or Epsilon_pulse).

    Returns:
        np.ndarray: Electric field values.
    """
    # Calculate the electric field for each time
    E = np.array([f(t, pulse_seq) for t in times])
    # Normalize if not envelope
    if f != pulse_envelope and len(pulse_seq.pulses) > 0:
        E0 = pulse_seq.pulses[0].pulse_amplitude
        if E0 != 0:
            E *= 0.5 * E0
    return E
