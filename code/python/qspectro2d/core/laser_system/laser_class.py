# =============================
# Pulse and LaserPulseSystem classes for structured pulse handling
# =============================
from dataclasses import dataclass, field  # for the class definiton
from typing import List, Tuple, Optional, Union
from qspectro2d.core.utils_and_config import convert_cm_to_fs
import numpy as np
import json


@dataclass
class Pulse:
    """
    Represents a single optical pulse with its temporal and spectral properties.
    """

    pulse_index: int
    pulse_peak_time: float
    pulse_phase: float
    pulse_fwhm: float
    pulse_amplitude: float
    pulse_freq: float
    pulse_type: str = "cos2"

    def __post_init__(self):
        if self.pulse_fwhm <= 0:
            raise ValueError("Pulse FWHM must be positive")
        if not np.isfinite(self.pulse_amplitude):
            raise ValueError("Pulse amplitude must be finite")
        self.omgea_laser = self.pulse_freq

    @property
    def active_time_range(self, n_fwhm: float = 1.094) -> Tuple[float, float]:
        """
        Get the active time range for the pulse (default ±1.094FWHM).
        Within +/- 1 FWHM, the 95% of the pulse energy is contained.
        Within +/- 1.094 FWHM, the 99% of the pulse energy is contained.
        """
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
            f"type = {self.pulse_type:<7}"
        )

    def to_dict(self) -> dict:
        return {
            "pulse_index": self.pulse_index,
            "pulse_peak_time": self.pulse_peak_time,
            "pulse_phase": self.pulse_phase,
            "pulse_fwhm": self.pulse_fwhm,
            "pulse_amplitude": self.pulse_amplitude,
            "pulse_freq": self.pulse_freq,
            "pulse_type": self.pulse_type,
        }

    @staticmethod
    def from_dict(data: dict) -> "Pulse":
        return Pulse(**data)


@dataclass
class LaserPulseSystem:
    pulses: List[Pulse] = field(default_factory=list)

    pulse_indices: List[int] = field(init=False)
    pulse_peak_times: List[float] = field(init=False)
    pulse_phases: List[float] = field(init=False)
    pulse_fwhms: List[float] = field(init=False)
    pulse_freqs: List[float] = field(init=False)
    pulse_types: List[str] = field(init=False)
    omega_laser: Optional[float] = (
        None  # Frequency of the laser, if all pulses have the same frequency
    )

    def __post_init__(self):
        self.pulse_indices = [pulse.pulse_index for pulse in self.pulses]
        self.pulse_peak_times = [pulse.pulse_peak_time for pulse in self.pulses]
        self.pulse_phases = [pulse.pulse_phase for pulse in self.pulses]
        self.pulse_fwhms = [pulse.pulse_fwhm for pulse in self.pulses]
        self.pulse_freqs = [pulse.pulse_freq for pulse in self.pulses]
        self.pulse_types = [pulse.pulse_type for pulse in self.pulses]

        if all(freq == self.pulse_freqs[0] for freq in self.pulse_freqs):
            self.omega_laser = self.pulse_freqs[0]

    @staticmethod
    def from_delays(
        delays: List[float],
        base_amplitude: float = 0.05,
        pulse_fwhm: float = 15.0,
        carrier_freq_cm: float = 16000.0,
        pulse_type: str = "gaussian",
        relative_E0s: Optional[List[float]] = None,
        phases: Optional[List[float]] = None,
    ) -> "LaserPulseSystem":
        """Create a LaserPulseSystem from a list of delays and other pulse parameters.

        Args:
            delays (List[float]): List of pulse delays.
            base_amplitude (float, optional): Base amplitude for the pulses. Defaults to 0.05.
            pulse_fwhm (float, optional): Full width at half maximum (FWHM) for the pulses. Defaults to 15.0.
            carrier_freq_cm (float, optional): Carrier frequency for the pulses in cm^-1. Defaults to 16000.0.
            pulse_type (str, optional): Type of the pulse (e.g., "gaussian"). Defaults to "gaussian".
            relative_E0s (Optional[List[float]], optional): Relative electric field amplitudes for each pulse. Defaults to None.
            phases (Optional[List[float]], optional): Phase shifts for each pulse in radians. Defaults to None.

        Raises:
            ValueError: If the lengths of the input lists do not match.

        Returns:
            LaserPulseSystem: A LaserPulseSystem object containing the defined pulses.
        """
        n_pulses = len(delays)
        if relative_E0s is None:
            relative_E0s = [1.0] * n_pulses
        if phases is None:
            phases = [0.0] * n_pulses
        if not (len(relative_E0s) == len(phases) == n_pulses):
            raise ValueError("Lengths of delays, relative_E0s, and phases must match")

        sorted_indices = sorted(range(n_pulses), key=lambda i: delays[i])
        delays = [delays[i] for i in sorted_indices]
        relative_E0s = [relative_E0s[i] for i in sorted_indices]
        phases = [phases[i] for i in sorted_indices]
        carrier_freq_fs = convert_cm_to_fs(carrier_freq_cm)

        pulse_indices = list(range(n_pulses))
        pulses = []
        for i in range(n_pulses):
            pulses.append(
                Pulse(
                    pulse_index=pulse_indices[i],
                    pulse_peak_time=delays[i],
                    pulse_phase=phases[i],
                    pulse_freq=carrier_freq_fs,
                    pulse_fwhm=pulse_fwhm,
                    pulse_amplitude=base_amplitude * relative_E0s[i],
                    pulse_type=pulse_type,
                )
            )

        return LaserPulseSystem(pulses=pulses)

    @staticmethod
    def from_general_specs(
        pulse_peak_times: Union[float, List[float]],
        pulse_phases: Union[float, List[float]],
        pulse_amplitudes: Union[float, List[float]],
        pulse_fwhms: Union[float, List[float]],
        pulse_freqs: Union[float, List[float]],
        pulse_types: Union[str, List[str]],
        pulse_indices: Optional[List[int]] = None,
    ) -> "LaserPulseSystem":
        if isinstance(pulse_peak_times, (float, int)):
            pulse_peak_times = [pulse_peak_times]
        n_pulses = len(pulse_peak_times)

        def expand(param, name):
            if isinstance(param, (float, int, str)):
                return [param] * n_pulses
            elif isinstance(param, list):
                if len(param) != n_pulses:
                    raise ValueError(f"{name} must have length {n_pulses}")
                return param
            raise TypeError(f"{name} must be float, str or list")

        pulse_phases = expand(pulse_phases, "pulse_phases")
        freq_list = expand(pulse_freqs, "pulse_freqs")
        fwhm_list = expand(pulse_fwhms, "pulse_fwhms")
        amp_list = expand(pulse_amplitudes, "pulse_amplitudes")
        type_list = expand(pulse_types, "pulse_types")

        if pulse_indices is None:
            pulse_indices = list(range(n_pulses))
        elif len(pulse_indices) != n_pulses:
            raise ValueError("pulse_indices must match number of pulses")

        pulses = []
        for i in range(n_pulses):
            pulses.append(
                Pulse(
                    pulse_index=pulse_indices[i],
                    pulse_peak_time=pulse_peak_times[i],
                    pulse_phase=pulse_phases[i],
                    pulse_freq=freq_list[i],
                    pulse_fwhm=fwhm_list[i],
                    pulse_amplitude=amp_list[i],
                    pulse_type=type_list[i],
                )
            )

        pulses.sort(key=lambda p: p.pulse_peak_time)
        return LaserPulseSystem(pulses=pulses)

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

    def update_first_two_pulse_phases(self, phase1: float, phase2: float) -> None:
        """
        Update the pulse_phases of the first two pulses in the LaserPulseSystem.

        Args:
            phase1 (float): Phase to set for the first pulse.
            phase2 (float): Phase to set for the second pulse.

        Raises:
            ValueError: If there are fewer than two pulses in the system.
        """
        if len(self.pulses) < 2:
            raise ValueError(
                "LaserPulseSystem must contain at least two pulses to update their phases."
            )

        self.pulses[0].pulse_phase = phase1
        self.pulses[1].pulse_phase = phase2

        # Update the cached pulse_phases list
        self.pulse_phases[0] = phase1
        self.pulse_phases[1] = phase2

    # turn the LaserPulseSystem into a list-like object
    def __len__(self):
        return len(self.pulses)

    def __getitem__(self, index):
        return self.pulses[index]

    def __iter__(self):
        return iter(self.pulses)

    def summary(self):
        header = f"LaserPulseSystem Summary\n{'-' * 80}"
        print(header)
        for p in self.pulses:
            print(p.summary_line())

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict:
        return {"pulses": [pulse.to_dict() for pulse in self.pulses]}

    @staticmethod
    def from_dict(data: dict) -> "LaserPulseSystem":
        pulses = [Pulse.from_dict(d) for d in data["pulses"]]
        return LaserPulseSystem(pulses=pulses)

    def to_json(self, indent: int = 2) -> str:

        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(json_str: str) -> "LaserPulseSystem":

        data = json.loads(json_str)
        return LaserPulseSystem.from_dict(data)


# How to recreate a LaserPulseSystem and serialize it to JSON
"""
seq = LaserPulseSystem.from_delays(delays=[100.0, 200.0], pulse_fwhm=10.0)
json_data = seq.to_json()
print("Serialized LaserPulseSystem:\n", json_data)

reconstructed = LaserPulseSystem.from_json(json_data)
reconstructed.summary()
"""


def main():
    print("=" * 60)
    print("PULSE SEQUENCE DEMO")
    print("=" * 60)

    print("\n1. Three-pulse sequence:")
    seq = LaserPulseSystem.from_delays(
        delays=[100.0, 200.0, 300.0],
        base_amplitude=0.05,
        pulse_fwhm=10.0,
        carrier_freq_cm=15800.0,
        relative_E0s=[1.0, 1.0, 0.1],
        phases=[0.0, 0.5, 1.0],
    )
    seq.summary()

    print("\n1. 5-pulse sequence from general specs:")
    seq = LaserPulseSystem.from_general_specs(
        pulse_peak_times=[100.0, 200.0, 300.0, 400.0, 500.0],
        pulse_phases=[0.0, 0.5, 1.0, 1.5, 2.0],
        pulse_amplitudes=[0.05, 0.05, 0.05, 0.05, 0.05],
        pulse_fwhms=[10.0, 10.0, 10.0, 10.0, 10.0],
        pulse_freqs=[15800.0, 15800.0, 15800.0, 15800.0, 15800.0],
        pulse_types=["gaussian", "gaussian", "gaussian", "gaussian", "gaussian"],
    )
    seq.summary()

    print("\n2. Active time ranges for each pulse:")
    for pulse in seq:
        start, end = pulse.active_time_range
        print(f"Pulse {pulse.pulse_index} active from {start:.2f} fs to {end:.2f} fs")

    times_to_check = [95.0, 300.0]  # adjust as needed
    print("\n3. Checking active pulses at various times:")
    for t in times_to_check:
        info = seq.get_field_info_at_time(t)
        print(f"At time {t} fs:")
        print(f"  Number of active pulses: {info['num_active_pulses']}")
        print(f"  Total amplitude: {info['total_amplitude']:.3e}")
        print(f"  Active pulse indices: {info['pulse_indices']}")

    # Example usage of the new functions
    times = np.linspace(0, 600, 1201)  # Time array from 0 to 600 fs
    active_regions = identify_non_zero_pulse_regions(times, seq)

    print("\n4. Identified active regions:")
    for i, active in enumerate(active_regions):
        if active:
            print(f"Time {times[i]:.2f} fs is in an active region.")

    # Split the time array based on active regions
    split_times = split_by_active_regions(times, active_regions)
    print("\n5. Split time segments:")
    for segment in split_times:
        print(f"Segment: {segment}")


if __name__ == "__main__":
    main()
