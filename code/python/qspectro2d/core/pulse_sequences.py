# TODO - Add docstrings to all methods and classes
#      - check for unnecessary code

# =============================
# Pulse and PulseSequence classes for structured pulse handling
# =============================
from qspectro2d.core.system_parameters import SystemParameters
from dataclasses import dataclass, field  # for the class definiton
from typing import List, Tuple
import numpy as np


@dataclass
class Pulse:
    """
    Represents a single optical pulse with its temporal and spectral properties.

    This class defines a pulse with configurable envelope shape, timing, and spectral
    characteristics for use in optical spectroscopy simulations.
    """

    pulse_peak_time: float  # Time when pulse reaches maximum intensity [fs]
    pulse_FWHM: float  # Full width at half maximum duration [fs]
    pulse_phase: float  # Phase offset of the pulse [rad]
    pulse_amplitude: float  # Peak amplitude of the electric field [V/m]
    pulse_freq: float  # Central frequency of the pulse [rad/fs]
    pulse_index: int  # Index indicating which pulse this is in a sequence (0-based)
    envelope_type: str = "cos2"  # Envelope shape: 'cos2' or 'gaussian'

    @property
    def active_time_range(self) -> Tuple[float, float]:
        """
        Calculate the time range where the pulse is active.

        The active range is defined as the time interval where the pulse envelope
        has significant amplitude, extending one FWHM before and after the peak time.

        Returns:
            Tuple[float, float]: (start_time, end_time) where the pulse is active
                                start_time = t_peak - FWHM [fs]
                                end_time = t_peak + FWHM [fs]
        """
        start_time = self.pulse_peak_time - self.pulse_FWHM
        end_time = self.pulse_peak_time + self.pulse_FWHM
        return (start_time, end_time)


@dataclass
class PulseSequence:
    pulses: list = field(default_factory=list)  # List of Pulse objects

    @staticmethod
    def from_pulse_specs(
        system: SystemParameters, pulse_specs: List[Tuple[int, float, float]]
    ) -> "PulseSequence":
        """
        Factory method to create a PulseSequence from pulse specifications.

        This method directly specifies pulse indices, times, and phases
        providing a clean and intuitive interface for pulse sequence creation.

        Parameters:
            system (SystemParameters): System configuration containing pulse parameters
            pulse_specs (List[Tuple[int, float, float]]): List of pulse specifications where
                each tuple contains (pulse_index, peak_time, phase)
                - pulse_index: Index of this pulse (used for amplitude/FWHM lookup)
                - peak_time: Time when pulse reaches maximum intensity [fs]
                - phase: Phase offset of the pulse [rad]

        Returns:
            PulseSequence: An instance containing the specified pulses

        Example:
            # Create a 3-pulse sequence
            pulse_specs = [
                (0, 100.0, 0.0),    # Pulse 0 at t=100fs, phase=0
                (1, 200.0, np.pi),  # Pulse 1 at t=200fs, phase=π
                (2, 300.0, 0.5)     # Pulse 2 at t=300fs, phase=0.5
            ]
            seq = PulseSequence.from_pulse_specs(system, pulse_specs)
        """
        # Extract system parameters
        pulse_freq = system.omega_laser
        FWHMs = system.FWHMs
        E_amps = system.E_amps
        envelope_type = system.envelope_type

        # Validate inputs
        if not pulse_specs:
            raise ValueError("pulse_specs cannot be empty")

        for spec in pulse_specs:
            if len(spec) != 3:
                raise ValueError(
                    "Each pulse_spec must be a tuple of (index, time, phase)"
                )
            pulse_idx, _, _ = spec
            if pulse_idx >= len(FWHMs) or pulse_idx >= len(E_amps):
                raise ValueError(
                    f"Pulse index {pulse_idx} exceeds available parameters"
                )

        # Create pulses from specifications
        pulses = []
        for pulse_index, peak_time, phase in pulse_specs:
            pulses.append(
                Pulse(
                    pulse_peak_time=peak_time,
                    pulse_phase=phase,
                    pulse_FWHM=FWHMs[pulse_index],
                    pulse_amplitude=E_amps[pulse_index],
                    pulse_freq=pulse_freq,
                    pulse_index=pulse_index,
                    envelope_type=envelope_type,
                )
            )

        return PulseSequence(pulses=pulses)

    @staticmethod
    def create_sequence(
        system: SystemParameters,
        times: List[float],
        phases: List[float] = None,
        indices: List[int] = None,
    ) -> "PulseSequence":
        """
        Convenience method to create a PulseSequence with automatic indexing.

        This is the simplest way to create a pulse sequence - just specify the times
        and optionally the phases. Pulse indices are assigned automatically (0, 1, 2, ...).

        Parameters:
            system (SystemParameters): System configuration containing pulse parameters
            times (List[float]): Peak times for each pulse [fs]
            phases (List[float], optional): Phase for each pulse [rad]. Defaults to all zeros.
            indices (List[int], optional): Pulse indices. Defaults to [0, 1, 2, ...]

        Returns:
            PulseSequence: An instance containing the specified pulses

        Example:
            # Simple 3-pulse sequence with default phases (all zero)
            seq = PulseSequence.create_sequence(system, [100.0, 200.0, 300.0])

            # With custom phases
            seq = PulseSequence.create_sequence(
                system,
                times=[100.0, 200.0, 300.0],
                phases=[0.0, np.pi, 0.5]
            )
        """
        if not times:
            raise ValueError("times cannot be empty")

        n_pulses = len(times)

        # Set default phases if not provided
        if phases is None:
            phases = [0.0] * n_pulses
        elif len(phases) != n_pulses:
            raise ValueError("Length of phases must match length of times")

        # Set default indices if not provided
        if indices is None:
            indices = list(range(n_pulses))
        elif len(indices) != n_pulses:
            raise ValueError("Length of indices must match length of times")

        # Create pulse specifications
        pulse_specs = list(zip(indices, times, phases))

        return PulseSequence.from_pulse_specs(system, pulse_specs)

    def as_dict(self) -> dict:
        """
        Convert to dictionary format compatible with legacy code.

        Returns:
            dict: Dictionary with key "pulses" and a list of pulse parameter dicts
        """
        return {"pulses": [pulse.__dict__ for pulse in self.pulses]}

    def get_all_active_time_ranges(self) -> List[Tuple[float, float]]:
        """
        Get the active time ranges for all pulses in the sequence.

        Returns:
            List[Tuple[float, float]]: List of (start_time, end_time) tuples for each pulse,
                                     where each tuple represents the time range where
                                     the corresponding pulse is active
        """
        return [pulse.active_time_range for pulse in self.pulses]

    def get_total_active_time_range(self) -> Tuple[float, float]:
        """
        Get the total time range covering all active pulses in the sequence.

        Returns:
            Tuple[float, float]: (earliest_start_time, latest_end_time) covering
                               all pulses in the sequence
        """
        if not self.pulses:
            raise ValueError("Cannot determine time range for empty pulse sequence")

        all_ranges = self.get_all_active_time_ranges()
        earliest_start = min(start for start, _ in all_ranges)
        latest_end = max(end for _, end in all_ranges)

        return (earliest_start, latest_end)

    def get_active_pulses_at_time(self, time: float) -> List[Tuple[int, Pulse]]:
        """
        Get all pulses that are active at a given time.

        Parameters:
            time (float): The time at which to check for active pulses

        Returns:
            List[Tuple[int, Pulse]]: List of (pulse_index, pulse) tuples for pulses
                                   that are active at the given time
        """
        active_pulses = []

        for i, pulse in enumerate(self.pulses):
            start_time, end_time = pulse.active_time_range
            if start_time <= time <= end_time:
                active_pulses.append((i, pulse))

        return active_pulses

    def get_total_amplitude_at_time(self, time: float) -> float:
        """
        Calculate the total electric field amplitude (E0) at a given time.
        This is the sum of all active pulse amplitudes at that time.

        Parameters:
            time (float): The time at which to calculate the total amplitude

        Returns:
            float: Total electric field amplitude E0 = sum of all active pulse_amplitudes
        """
        active_pulses = self.get_active_pulses_at_time(time)
        total_amplitude = sum(pulse.pulse_amplitude for _, pulse in active_pulses)

        return total_amplitude

    def get_field_info_at_time(self, time: float) -> dict:
        """
        Get comprehensive information about the electric field at a given time.

        Parameters:
            time (float): The time at which to analyze the field

        Returns:
            dict: Dictionary containing:
                - 'active_pulses': List of (pulse_index, pulse) tuples
                - 'num_active_pulses': Number of active pulses
                - 'total_amplitude': Total E0 = sum of active pulse amplitudes
                - 'individual_amplitudes': List of individual pulse amplitudes
                - 'pulse_indices': List of indices of active pulses
        """
        active_pulses = self.get_active_pulses_at_time(time)

        return {
            "active_pulses": active_pulses,
            "num_active_pulses": len(active_pulses),
            "total_amplitude": sum(pulse.pulse_amplitude for _, pulse in active_pulses),
            "individual_amplitudes": [
                pulse.pulse_amplitude for _, pulse in active_pulses
            ],
            "pulse_indices": [i for i, _ in active_pulses],
        }


"""
Utility function for identifying pulse regions in time arrays.
"""


def identify_pulse_regions(  # TODO even used??
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
