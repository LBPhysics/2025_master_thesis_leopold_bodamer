# Pulse and LaserPulseSequence classes for structured pulse handling

from dataclasses import dataclass, field  # for the class definiton
from typing import List, Tuple, Optional, Union

import numpy as np
import json
from qspectro2d.constants import convert_cm_to_fs, convert_fs_to_cm


@dataclass
class LaserPulse:
    """
    Represents a single optical pulse with its temporal and spectral properties.

    External (public) frequency unit:  cm^-1  (pulse_freq_cm)
    Internal (private) frequency unit: fs^-1  (_pulse_freq_fs)

    Access patterns:
        - Provide frequency in cm^-1 via pulse_freq_cm at construction
        - Use .pulse_freq_cm for human readable output / serialization
        - Use .pulse_freq_fs internally for dynamics
    """

    pulse_index: int
    pulse_peak_time: float  # [fs]
    pulse_phase: float  # [rad]
    pulse_fwhm_fs: float  # [fs]
    pulse_amplitude: float
    pulse_freq_cm: float  # user supplied central frequency [cm^-1]
    envelope_type: str = "cos2"

    # internal cache (not part of __init__ signature)
    _pulse_freq_fs: float = field(init=False, repr=False)

    def __post_init__(self):

        # VALIDATION

        if self.pulse_fwhm_fs <= 0:
            raise ValueError("Pulse FWHM must be positive")
        if not np.isfinite(self.pulse_amplitude):
            raise ValueError("Pulse amplitude must be finite")
        if self.pulse_freq_cm <= 0:
            raise ValueError("Pulse frequency (cm^-1) must be positive.")

        # UNIT CONVERSION (single source of truth)

        self._pulse_freq_fs = float(convert_cm_to_fs(self.pulse_freq_cm))

        # PRECOMPUTE ENVELOPE SUPPORT
        self._recompute_envelope_support()

    def _recompute_envelope_support(self) -> None:
        """Recompute cached envelope window and Gaussian parameters.

        Must be called whenever attributes affecting timing change, e.g.,
        `pulse_peak_time`, `pulse_fwhm_fs`, or `envelope_type`.
        """
        if self.envelope_type == "gaussian":
            # Use extended active window (≈1% cutoff) defined by active_time_range (± n_fwhm * FWHM)
            self._t_start, self._t_end = self.active_time_range  # uses default n_fwhm (1.094)
            self._sigma = self.pulse_fwhm_fs / (2 * np.sqrt(2 * np.log(2)))
            # Baseline value chosen at EXTENDED window edge (edge_span), not at FWHM, so
            # envelope retains smooth tails between ±FWHM and ±edge_span.
            edge_span = self._t_end - self.pulse_peak_time
            self._boundary_val = float(np.exp(-(edge_span**2) / (2 * self._sigma**2)))
        else:
            self._t_start = self.pulse_peak_time - self.pulse_fwhm_fs
            self._t_end = self.pulse_peak_time + self.pulse_fwhm_fs
            self._sigma = None
            self._boundary_val = None

    @property
    def pulse_freq_fs(self) -> float:
        """Internal frequency in fs^-1 (read-only)."""
        return self._pulse_freq_fs

    def update_frequency_cm(self, new_freq_cm: float) -> None:
        """Update the carrier frequency (cm^-1) and keep internal cache synchronized."""
        if new_freq_cm <= 0:
            raise ValueError("new_freq_cm must be positive")
        self.pulse_freq_cm = float(new_freq_cm)
        self._pulse_freq_fs = float(convert_cm_to_fs(self.pulse_freq_cm))

    @property
    def active_time_range(self, n_fwhm: float = 1.094) -> Tuple[float, float]:
        """Return (t_min, t_max) where gaussian envelope ≳1% (n_fwhm=1.094)."""
        duration = n_fwhm * self.pulse_fwhm_fs
        return (self.pulse_peak_time - duration, self.pulse_peak_time + duration)

    def summary_line(self) -> str:
        return (
            f"Pulse {self.pulse_index:>2}: "
            f"t = {self.pulse_peak_time:6.2f} fs | "
            f"E₀ = {self.pulse_amplitude:.3e} | "
            f"FWHM = {self.pulse_fwhm_fs:4.1f} fs | "
            f"ω = {self.pulse_freq_cm:8.2f} cm^-1 | "
            f"ϕ = {self.pulse_phase:6.3f} rad | "
            f"type = {self.envelope_type:<7}"
        )

    def to_dict(self):
        return {
            "pulse_index": self.pulse_index,
            "pulse_peak_time": self.pulse_peak_time,
            "pulse_phase": self.pulse_phase,
            "pulse_fwhm_fs": self.pulse_fwhm_fs,
            "pulse_amplitude": self.pulse_amplitude,
            "pulse_freq_cm": self.pulse_freq_cm,
            "envelope_type": self.envelope_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LaserPulse":
        if "pulse_freq_cm" not in data:
            raise KeyError("Missing required key 'pulse_freq_cm' (cm^-1).")
        return cls(**data)


@dataclass
class LaserPulseSequence:
    pulses: List[LaserPulse] = field(default_factory=list)

    def __post_init__(self):
        self.pulses.sort(key=lambda p: p.pulse_peak_time)
        if not self.pulses:
            self.E0 = 0.0
            self._carrier_freq_fs: Optional[float] = None
            return
        self.E0 = self.pulses[0].pulse_amplitude
        first_fs = self.pulses[0].pulse_freq_fs
        if all(np.isclose(p.pulse_freq_fs, first_fs) for p in self.pulses):
            self._carrier_freq_fs = first_fs
        else:
            self._carrier_freq_fs = None

    # --- Dynamic Properties ---
    @property
    def pulse_indices(self) -> List[int]:
        return [p.pulse_index for p in self.pulses]

    @property
    def pulse_peak_times(self) -> List[float]:
        return [p.pulse_peak_time for p in self.pulses]

    @property
    def pulse_phases(self) -> List[float]:
        return [p.pulse_phase for p in self.pulses]

    @property
    def pulse_fwhms(self) -> List[float]:
        return [p.pulse_fwhm_fs for p in self.pulses]

    @property
    def pulse_freqs_cm(self) -> List[float]:
        return [p.pulse_freq_cm for p in self.pulses]

    @property
    def pulse_freqs_fs(self) -> List[float]:
        return [p.pulse_freq_fs for p in self.pulses]

    @property
    def envelope_types(self) -> List[str]:
        return [p.envelope_type for p in self.pulses]

    @property
    def pulse_amplitudes(self) -> List[float]:
        return [p.pulse_amplitude for p in self.pulses]

    @property
    def carrier_freq_fs(self) -> Optional[float]:
        return self._carrier_freq_fs

    @property
    def carrier_freq_cm(self) -> Optional[float]:
        if self._carrier_freq_fs is None:
            return None
        return float(convert_fs_to_cm(self._carrier_freq_fs))

    # --- Factory Methods ---
    @staticmethod
    def _create_pulse(index, t, phi, fwhm, freq_cm, amp, env) -> LaserPulse:
        return LaserPulse(
            pulse_index=index,
            pulse_peak_time=t,
            pulse_phase=phi,
            pulse_freq_cm=freq_cm,
            pulse_fwhm_fs=fwhm,
            pulse_amplitude=amp,
            envelope_type=env,
        )

    @staticmethod
    def from_delays(
        delays: List[float],
        base_amplitude: float = 1,
        pulse_fwhm_fs: float = 15.0,
        carrier_freq_cm: float = 16000.0,
        envelope_type: str = "gaussian",
        relative_E0s: Optional[List[float]] = None,
        phases: Optional[List[float]] = None,
    ) -> "LaserPulseSequence":
        n_pulses = len(delays) + 1
        if relative_E0s is None:
            relative_E0s = [1.0] * n_pulses
            relative_E0s[-1] = 0.1
        if phases is None:
            phases = [0.0] * n_pulses

        if not (len(relative_E0s) == len(phases) == n_pulses):
            raise ValueError("Lengths of delays, relative_E0s, and phases must match")

        peak_times = np.insert(np.cumsum(delays), 0, 0.0)

        pulses = [
            LaserPulseSequence._create_pulse(
                i,
                peak_times[i],
                phases[i],
                pulse_fwhm_fs,
                carrier_freq_cm,
                base_amplitude * relative_E0s[i],
                envelope_type,
            )
            for i in range(n_pulses)
        ]

        return LaserPulseSequence(pulses=pulses)

    @staticmethod
    def from_general_specs(
        pulse_peak_times: Union[float, List[float]],
        pulse_phases: Union[float, List[float]],
        pulse_amplitudes: Union[float, List[float]],
        pulse_fwhms: Union[float, List[float]],
        pulse_freqs_cm: Union[float, List[float]],  # cm^-1 interface preserved
        envelope_types: Union[str, List[str]],
        pulse_indices: Optional[List[int]] = None,
    ) -> "LaserPulseSequence":
        if isinstance(pulse_peak_times, (float, int)):
            pulse_peak_times = [pulse_peak_times]

        n = len(pulse_peak_times)

        def expand(param, name):
            if isinstance(param, (float, int, str)):
                return [param] * n
            if isinstance(param, list):
                if len(param) != n:
                    raise ValueError(f"{name} must have length {n}")
                return param
            raise TypeError(f"{name} must be float, str, or list")

        pulse_phases = expand(pulse_phases, "pulse_phases")
        amps = expand(pulse_amplitudes, "pulse_amplitudes")
        fwhms = expand(pulse_fwhms, "pulse_fwhms")
        freqs_cm = expand(pulse_freqs_cm, "pulse_freqs_cm (cm^-1)")
        envs = expand(envelope_types, "envelope_types")

        if pulse_indices is None:
            pulse_indices = list(range(n))
        elif len(pulse_indices) != n:
            raise ValueError("pulse_indices must match number of pulses")

        pulses = [
            LaserPulseSequence._create_pulse(
                pulse_indices[i],
                pulse_peak_times[i],
                pulse_phases[i],
                fwhms[i],
                freqs_cm[i],
                amps[i],
                envs[i],
            )
            for i in range(n)
        ]

        return LaserPulseSequence(pulses=pulses)

    def get_active_pulses_at_time(self, time: float) -> List[LaserPulse]:
        """Return list of pulses active at given time (within their active_time_range)."""
        active_pulses: List[LaserPulse] = []
        for pulse in self.pulses:
            start_time, end_time = pulse.active_time_range
            if start_time <= time <= end_time:
                active_pulses.append(pulse)

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
        total_amplitude = sum(pulse.pulse_amplitude for pulse in active_pulses)

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
        active = self.get_active_pulses_at_time(time)

        return {
            "active_pulses": active,
            "num_active_pulses": len(active),
            "total_amplitude": sum(p.pulse_amplitude for p in active),
            "individual_amplitudes": [p.pulse_amplitude for p in active],
            "pulse_indices": [p.pulse_index for p in active],
        }

    def update_phases(self, phases: List[float]) -> None:
        """
        Update the pulse_phases of the pulses in the LaserPulseSequence.
        Raises:
            ValueError: If the number of pulses doesnt match the phases.
        """
        if len(phases) - 1 > len(self.pulses):
            raise ValueError(
                f"Number of pulses ({len(self.pulses)}) must be at least number of phases ({len(phases)})"
            )
        for i, phase in enumerate(phases):
            if i < len(self.pulses):
                self.pulses[i].pulse_phase = phase

    def update_delays(self, delays: List[float]) -> None:
        """
        Update the pulse_peak_time of each pulse in the sequence based on inter-pulse delays.

        Args:
            delays (List[float]): Delays between pulses. Length must be one less than number of pulses.

        Raises:
            ValueError: If the number of delays is not one less than the number of pulses.
        """
        if len(delays) != len(self.pulses) - 1:
            raise ValueError(
                f"Number of delays ({len(delays)}) must be one less than number of pulses ({len(self.pulses)})"
            )

        peak_times = np.insert(np.cumsum(delays), 0, 0.0)

        # Update each pulse's peak time and recompute envelope-derived caches
        for pulse, new_peak in zip(self.pulses, peak_times):
            pulse.pulse_peak_time = new_peak
            # Keep envelope support in sync with new peak time
            if hasattr(pulse, "_recompute_envelope_support"):
                pulse._recompute_envelope_support()

    # --- Convenience ---
    def __len__(self):
        return len(self.pulses)

    def __getitem__(self, index):
        return self.pulses[index]

    def __iter__(self):
        return iter(self.pulses)

    def summary(self):
        print(str(self))

    def __str__(self) -> str:
        header = f"LaserPulseSequence Summary\n{'-' * 80}\n"
        return header + "\n".join(p.summary_line() for p in self.pulses)

    def to_dict(self) -> dict:
        return {
            "pulses": [p.to_dict() for p in self.pulses],
            "E0": self.E0,
            "carrier_freq_cm": self.carrier_freq_cm,
        }

    @staticmethod
    def from_dict(data: dict) -> "LaserPulseSequence":
        pulses = [LaserPulse.from_dict(d) for d in data["pulses"]]
        return LaserPulseSequence(pulses=pulses)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(json_str: str) -> "LaserPulseSequence":
        return LaserPulseSequence.from_dict(json.loads(json_str))


# How to recreate a LaserPulseSequence and serialize it to JSON
"""
seq = LaserPulseSequence.from_delays(delays=[100.0, 200.0], pulse_fwhm_fs=10.0)
json_data = seq.to_json()
print("Serialized LaserPulseSequence:\n", json_data)

reconstructed = LaserPulseSequence.from_json(json_data)
reconstructed.summary()
"""


def identify_non_zero_pulse_regions(times: np.ndarray, pulse_seq: LaserPulseSequence) -> np.ndarray:
    """
    Identify regions where the pulse envelope is non-zero across an array of time values.

    Args:
        times (np.ndarray): Array of time values to evaluate
        pulse_seq (LaserPulseSequence): The pulse sequence to evaluate

    Returns:
        np.ndarray: Boolean array where True indicates times where envelope is non-zero
    """

    if not isinstance(pulse_seq, LaserPulseSequence):
        raise TypeError("pulse_seq must be a LaserPulseSequence instance.")

    # Initialize an array of all False values
    active_regions = np.zeros_like(times, dtype=bool)

    # For each time point, check if it's in the active region of any pulse
    for i, t in enumerate(times):
        # A time is in an active region if any pulse contributes to the envelope
        for pulse in pulse_seq.pulses:
            start_time, end_time = pulse.active_time_range

            # Check if time point falls within the pulse's active region
            if start_time <= t <= end_time:
                active_regions[i] = True
                break  # Once we know this time point is active, we can move to the next

    return active_regions


def split_by_active_regions(times: np.ndarray, active_regions: np.ndarray) -> List[np.ndarray]:
    """
    Split the time array into segments based on active regions.

    Args:
        times (np.ndarray): Array of time values.
        active_regions (np.ndarray): Boolean array indicating active regions.

    Returns:
        List[np.ndarray]: List of time segments split by active regions.
    """
    # Find where the active_regions changes value
    change_indices = np.where(np.diff(active_regions.astype(int)) != 0)[0] + 1

    # Split the times at those change points
    split_times = np.split(times, change_indices)

    # Return list of time segments
    return split_times
