# =============================
# Pulse and LaserPulseSequence classes for structured pulse handling
# =============================
from dataclasses import dataclass, field, asdict  # for the class definiton
from typing import List, Tuple, Optional, Union

from matplotlib.pylab import f
from qspectro2d.core.utils_and_config import convert_cm_to_fs
import numpy as np
import json


@dataclass
class LaserPulse:
    """
    Represents a single optical pulse with its temporal and spectral properties.
    """

    pulse_index: int
    pulse_peak_time: float  # [fs]
    pulse_phase: float  # [rad]
    pulse_fwhm: float  # [fs]
    pulse_amplitude: float
    pulse_freq: float  # [cm^-1], converted to rad/fs in __post_init__
    envelope_type: str = "cos2"
    _freq_converted: bool = field(default=False)  # Track conversion state

    def __post_init__(self):
        if self.pulse_fwhm <= 0:
            raise ValueError("Pulse FWHM must be positive")
        if not np.isfinite(self.pulse_amplitude):
            raise ValueError("Pulse amplitude must be finite")
        if self.pulse_freq <= 0:
            raise ValueError("Pulse frequency must be positive.")

        if not self._freq_converted:
            self.pulse_freq = convert_cm_to_fs(self.pulse_freq)
            self._freq_converted = True

    @property
    def omega_laser(self) -> float:
        return self.pulse_freq

    @property
    def active_time_range(self, n_fwhm: float = 1.094) -> Tuple[float, float]:
        duration = n_fwhm * self.pulse_fwhm
        return (self.pulse_peak_time - duration, self.pulse_peak_time + duration)

    def summary_line(self) -> str:
        return (
            f"Pulse {self.pulse_index:>2}: "
            f"t = {self.pulse_peak_time:6.2f} fs | "
            f"E₀ = {self.pulse_amplitude:.3e} | "
            f"FWHM = {self.pulse_fwhm:4.1f} fs | "
            f"ω = {self.pulse_freq:8.2f} rad/fs | "
            f"ϕ = {self.pulse_phase:6.3f} rad | "
            f"type = {self.envelope_type:<7}"
        )

    def to_dict(self):
        return {
            "pulse_index": self.pulse_index,
            "pulse_peak_time": self.pulse_peak_time,
            "pulse_phase": self.pulse_phase,
            "pulse_fwhm": self.pulse_fwhm,
            "pulse_amplitude": self.pulse_amplitude,
            "pulse_freq": self.pulse_freq,
            "envelope_type": self.envelope_type,
            "_freq_converted": self._freq_converted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LaserPulse":
        instance = cls(**data)
        return instance


@dataclass
class LaserPulseSequence:
    pulses: List[LaserPulse] = field(default_factory=list)

    def __post_init__(self):
        self.pulses.sort(key=lambda p: p.pulse_peak_time)

        # Set base E0 and omega_laser if consistent
        if self.pulses:
            self.E0 = self.pulses[0].pulse_amplitude
            if all(p.pulse_freq == self.pulses[0].pulse_freq for p in self.pulses):
                self.omega_laser = self.pulses[0].pulse_freq
            else:
                self.omega_laser = None
        else:
            self.E0 = 0.0
            self.omega_laser = None

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
        return [p.pulse_fwhm for p in self.pulses]

    @property
    def pulse_freqs(self) -> List[float]:
        return [p.pulse_freq for p in self.pulses]

    @property
    def envelope_types(self) -> List[str]:
        return [p.envelope_type for p in self.pulses]

    @property
    def pulse_amplitudes(self) -> List[float]:
        return [p.pulse_amplitude for p in self.pulses]

    # --- Factory Methods ---
    @staticmethod
    def _create_pulse(index, t, phi, fwhm, freq, amp, env) -> LaserPulse:
        return LaserPulse(
            pulse_index=index,
            pulse_peak_time=t,
            pulse_phase=phi,
            pulse_freq=freq,
            pulse_fwhm=fwhm,
            pulse_amplitude=amp,
            envelope_type=env,
        )

    @staticmethod
    def from_delays(
        delays: List[float],
        base_amplitude: float = 1,
        pulse_fwhm: float = 15.0,
        carrier_freq_cm: float = 16000.0,
        envelope_type: str = "gaussian",
        relative_E0s: Optional[List[float]] = None,
        phases: Optional[List[float]] = None,
    ) -> "LaserPulseSequence":
        n_pulses = len(delays)
        relative_E0s = relative_E0s or [1.0] * n_pulses
        phases = phases or [0.0] * n_pulses

        if not (len(relative_E0s) == len(phases) == n_pulses):
            raise ValueError("Lengths of delays, relative_E0s, and phases must match")

        sorted_indices = sorted(range(n_pulses), key=lambda i: delays[i])
        pulses = [
            LaserPulseSequence._create_pulse(
                i,
                delays[idx],
                phases[idx],
                pulse_fwhm,
                carrier_freq_cm,
                base_amplitude * relative_E0s[idx],
                envelope_type,
            )
            for i, idx in enumerate(sorted_indices)
        ]

        return LaserPulseSequence(pulses=pulses)

    @staticmethod
    def from_general_specs(
        pulse_peak_times: Union[float, List[float]],
        pulse_phases: Union[float, List[float]],
        pulse_amplitudes: Union[float, List[float]],
        pulse_fwhms: Union[float, List[float]],
        pulse_freqs: Union[float, List[float]],
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
        freqs = expand(pulse_freqs, "pulse_freqs")
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
                freqs[i],
                amps[i],
                envs[i],
            )
            for i in range(n)
        ]

        return LaserPulseSequence(pulses=pulses)

    def get_active_pulses_at_time(self, time: float) -> List[Tuple[int, LaserPulse]]:
        """
        Get all pulses that are active at a given time.

        Parameters:
            time (float): The time at which to check for active pulses

        Returns:
            List[Tuple[int, LaserPulse]]: List of (pulse_index, pulse) tuples for pulses
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
        active = self.get_active_pulses_at_time(time)

        return {
            "active_pulses": active,
            "num_active_pulses": len(active),
            "total_amplitude": sum(p.pulse_amplitude for _, p in active),
            "individual_amplitudes": [p.pulse_amplitude for _, p in active],
            "pulse_indices": [i for i, _ in active],
        }

    def update_phases(self, phases: List[float]) -> None:
        """
        Update the pulse_phases of the first two pulses in the LaserPulseSequence.

        Args:
            phase1 (float): Phase to set for the first pulse.
            phase2 (float): Phase to set for the second pulse.

        Raises:
            ValueError: If there are fewer than two pulses in the system.
        """
        if len(phases) != len(self.pulses):
            raise ValueError(
                f"Number of phases ({len(phases)}) must match number of pulses ({len(self.pulses)})"
            )
        for i, phase in enumerate(phases):
            self.pulses[i].pulse_phase = phase

    def update_delays(self, delays: List[float]) -> None:
        """
        Update the pulse_peak_time of each pulse in the sequence.

        Args:
            delays (List[float]): New peak times for each pulse. Must match the number of pulses.

        Raises:
            ValueError: If the number of delays doesn't match the number of pulses.
        """
        if len(delays) != len(self.pulses):
            raise ValueError(
                f"Number of delays ({len(delays)}) must match number of pulses ({len(self.pulses)})"
            )

        # Sort delays and pulses by new peak times
        sorted_indices = sorted(range(len(delays)), key=lambda i: delays[i])
        delays = [delays[i] for i in sorted_indices]
        # Update each pulse's peak time
        for pulse, new_delay in zip(self.pulses, delays):
            pulse.pulse_peak_time = new_delay

    # --- Convenience ---
    # turn the LaserPulseSequence into a list-like object
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
        return {"pulses": [p.to_dict() for p in self.pulses]}

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
seq = LaserPulseSequence.from_delays(delays=[100.0, 200.0], pulse_fwhm=10.0)
json_data = seq.to_json()
print("Serialized LaserPulseSequence:\n", json_data)

reconstructed = LaserPulseSequence.from_json(json_data)
reconstructed.summary()
"""


def identify_non_zero_pulse_regions(
    times: np.ndarray, pulse_seq: LaserPulseSequence
) -> np.ndarray:
    """
    Identify regions where the pulse envelope is non-zero across an array of time values.

    Args:
        times (np.ndarray): Array of time values to evaluate
        pulse_seq (LaserPulseSequence): The pulse sequence to evaluate

    Returns:
        np.ndarray: Boolean array where True indicates times where envelope is non-zero
    """

    if type(pulse_seq) is not LaserPulseSequence:
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


def split_by_active_regions(
    times: np.ndarray, active_regions: np.ndarray
) -> List[np.ndarray]:
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
