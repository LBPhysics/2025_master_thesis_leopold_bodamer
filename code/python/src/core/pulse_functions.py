from src.core.pulse_sequences import PulseSequence
import numpy as np


def pulse_envelope(t: float, pulse_seq: PulseSequence) -> float:
    """
    Calculate the combined envelope of multiple pulses at time t using PulseSequence.

    Now uses pulse_peak_time as t_peak (peak time) where cos²/gaussian is maximal.
    Pulse is zero outside [t_peak - FWHM, t_peak + FWHM] == outside of 2 FWHM.

    Uses the envelope_type from each pulse to determine which envelope function to use:
    - 'cos2': cosine squared envelope
    - 'gaussian': Gaussian envelope, shifted so that:
      - The Gaussian is zero at t_peak ± FWHM boundaries

    Args:
        t (float): Time value
        pulse_seq (PulseSequence): The pulse sequence

    Returns:
        float: Combined envelope value
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    envelope = 0.0
    for pulse in pulse_seq.pulses:
        t_peak = pulse.pulse_peak_time
        FWHM = pulse.pulse_FWHM
        envelope_type = getattr(
            pulse, "envelope_type", "cos2"
        )  # Default to cos2 for backward compatibility

        if FWHM is None or FWHM <= 0:
            continue
        if t_peak is None:
            continue

        # Pulse exists only in [t_peak - FWHM, t_peak + FWHM]
        if not (t_peak - FWHM <= t <= t_peak + FWHM):
            continue

        if envelope_type == "cos2":
            ### Cosine squared envelope
            arg = np.pi * (t - t_peak) / (2 * FWHM)
            envelope += np.cos(arg) ** 2

        elif envelope_type == "gaussian":
            ### Gaussian envelope
            sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
            gaussian_val = np.exp(-((t - t_peak) ** 2) / (2 * sigma**2))
            boundary_distance_sq = FWHM**2
            boundary_val = np.exp(-boundary_distance_sq / (2 * sigma**2))
            envelope += (
                max(0.0, gaussian_val) - boundary_val
            )  # for continuity if needed  # TODO Check the effect of this discontinuity

        else:
            raise ValueError(
                f"Unknown envelope_type: {envelope_type}. Use 'cos2' or 'gaussian'."
            )

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
