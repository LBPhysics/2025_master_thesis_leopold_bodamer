# =============================
# Pulse and PulseSequence classes for structured pulse handling
# =============================
from src.core.system_parameters import SystemParameters
from dataclasses import dataclass, field  # for the class definiton


@dataclass
class Pulse:
    pulse_peak_time: float
    pulse_FWHM: float
    pulse_phase: float
    pulse_amplitude: float
    pulse_freq: float
    envelope_type: str = "cos2"  # 'cos2' or 'gaussian'


@dataclass
class PulseSequence:
    pulses: list = field(default_factory=list)  # List of Pulse objects

    @staticmethod
    def from_args(
        system: SystemParameters,
        curr: tuple,
        prev: tuple = None,
        preprev: tuple = None,
    ) -> "PulseSequence":
        """
        Factory method to create a PulseSequence from argument tuples and lists,
        using a single global pulse_freq and Delta_t for all pulses.

        Parameters:
            curr (tuple): (start_time, phase) for the current pulse
            prev (tuple, optional): (start_time, phase) for the previous pulse
            preprev (tuple, optional): (start_time, phase) for the earliest pulse
            pulse_freq (float): Frequency for all pulses
            Delta_t (float): Half-width for all pulses
            E_amps (list): List of amplitudes for each pulse

        Returns:
            PulseSequence: An instance containing up to three pulses
        """
        pulse_freq = system.omega_laser
        # TODO a bit bat coding style to get FWHMs and E_amps (which describe the pulse shape) from the system parameters
        FWHMs = system.FWHMs
        E_amps = system.E_amps
        envelope_type = system.envelope_type

        pulses = []

        # Add the earliest pulse if provided (preprev)
        if preprev is not None:
            t0_preprev, phi_preprev = preprev
            pulses.append(
                Pulse(
                    pulse_peak_time=t0_preprev,
                    pulse_phase=phi_preprev,
                    pulse_FWHM=FWHMs[0],
                    pulse_amplitude=E_amps[0],
                    pulse_freq=pulse_freq,
                    envelope_type=envelope_type,
                )
            )

        # Add the previous pulse if provided (prev)
        if prev is not None:
            t0_prev, phi_prev = prev
            idx = 1 if preprev is not None else 0
            pulses.append(
                Pulse(
                    pulse_peak_time=t0_prev,
                    pulse_phase=phi_prev,
                    pulse_FWHM=FWHMs[1],
                    pulse_amplitude=E_amps[idx],
                    pulse_freq=pulse_freq,
                    envelope_type=envelope_type,
                )
            )

        # Always add the current pulse (curr)
        t0_curr, phi_curr = curr
        if preprev is not None and prev is not None:
            idx = 2
        elif preprev is not None or prev is not None:
            idx = 1
        else:
            idx = 0
        pulses.append(
            Pulse(
                pulse_peak_time=t0_curr,
                pulse_phase=phi_curr,
                pulse_FWHM=FWHMs[idx],
                pulse_amplitude=E_amps[idx],
                pulse_freq=pulse_freq,
                envelope_type=envelope_type,
            )
        )

        return PulseSequence(pulses=pulses)

    def as_dict(self) -> dict:
        """
        Convert to dictionary format compatible with legacy code.

        Returns:
            dict: Dictionary with key "pulses" and a list of pulse parameter dicts
        """
        return {"pulses": [pulse.__dict__ for pulse in self.pulses]}
