# =============================
# Pulse and PulseSequence classes for structured pulse handling
# =============================
from src.core.system_parameters import SystemParameters
from dataclasses import dataclass, field  # for the class definiton
from typing import List, Tuple
import numpy as np


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


"""
Utility function for identifying pulse regions in time arrays.
"""


def identify_pulse_regions(
    times: np.ndarray, pulse_seq: PulseSequence, system: SystemParameters = None
) -> List[Tuple[int, int, int]]:
    """
    Identify time regions where pulses are active in a time array.

    This function locates the time indices corresponding to the active regions of each
    pulse in a pulse sequence. For each pulse, it calculates the start and end times
    based on the pulse peak time and width (FWHM), finds the corresponding indices
    in the time array, and returns the regions sorted by start time.

    Parameters
    ----------
    times : np.ndarray
        Time array for the evolution.
    pulse_seq : PulseSequence
        PulseSequence object containing pulse information.
    system : SystemParameters, optional
        System parameters containing FWHM information. If None, FWHM must be
        available directly from pulse_seq.

    Returns
    -------
    List[Tuple[int, int, int]]
        List of tuples (start_idx, end_idx, pulse_idx) representing:
        - start_idx: Index in times array where the pulse region starts
        - end_idx: Index in times array where the pulse region ends
        - pulse_idx: Index of the pulse in the pulse sequence

    Notes
    -----
    The function defines pulse regions as t ∈ [peak_time - width, peak_time + width],
    where width is the FWHM (Full Width at Half Maximum) of the pulse.
    Regions are sorted chronologically by start time.
    """
    # Input validation
    if not isinstance(times, np.ndarray) or len(times) == 0:
        raise ValueError("Times must be a non-empty numpy array")
    if not hasattr(pulse_seq, "pulses") or len(pulse_seq.pulses) == 0:
        raise ValueError("PulseSequence must have at least one pulse")

    # Find pulse regions in the time array
    pulse_regions = []
    for i, pulse in enumerate(pulse_seq.pulses):
        pulse_peak_time = pulse.pulse_peak_time

        # Determine pulse width from system or fallback
        if system is not None and hasattr(system, "FWHMs") and i < len(system.FWHMs):
            pulse_width = system.FWHMs[i]
        elif system is not None and hasattr(system, "FWHM"):
            pulse_width = system.pulse_FWHMs[i]
        elif hasattr(pulse, "pulse_FWHM"):
            pulse_width = pulse.pulse_FWHM
        else:
            raise ValueError(
                f"Could not determine width for pulse {i}. "
                "Either system must provide FWHM/FWHMs or "
                "pulse must have a 'width' attribute."
            )

        # Find indices for pulse region: t ∈ [peak_time - width, peak_time + width]
        start_time = pulse_peak_time - pulse_width
        end_time = pulse_peak_time + pulse_width

        start_idx = np.abs(times - start_time).argmin()
        end_idx = np.abs(times - end_time).argmin()

        if start_idx < end_idx:  # Valid region
            pulse_regions.append((start_idx, end_idx, i))

    # Sort pulse regions by start time
    pulse_regions.sort(key=lambda x: x[0])

    return pulse_regions
