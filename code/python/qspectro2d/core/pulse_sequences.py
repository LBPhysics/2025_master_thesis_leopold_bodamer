# =============================
# Pulse and PulseSequence classes for structured pulse handling
# =============================
from dataclasses import dataclass, field  # for the class definiton
from typing import List, Tuple, Optional, Union


@dataclass
class Pulse:
    """
    Represents a single optical pulse with its temporal and spectral properties.

    This class defines a pulse with configurable envelope shape, timing, and spectral
    characteristics for use in optical spectroscopy simulations.
    """

    pulse_index: int  # Index indicating which pulse this is in a sequence (0-based)
    pulse_peak_time: float  # Time when pulse reaches maximum intensity [fs]
    pulse_phase: float  # Phase offset of the pulse [rad]
    pulse_fwhm: float  # Full width at half maximum duration [fs]
    pulse_amplitude: float  # Peak amplitude of the electric field []
    pulse_freq: float  # Central frequency of the pulse [rad/fs]
    pulse_type: str = "cos2"  # Envelope shape: 'cos2' or 'gaussian'

    @property
    def active_time_range(self) -> Tuple[float, float]:
        """
        Calculate the time range where the pulse is active.

        The active range is defined as the time interval where the pulse envelope
        has significant amplitude, extending one fwhm before and after the peak time.

        Returns:
            Tuple[float, float]: (start_time, end_time) where the pulse is active
                                start_time = t_peak - fwhm [fs]
                                end_time = t_peak + fwhm [fs]
        """
        start_time = self.pulse_peak_time - self.pulse_fwhm
        end_time = self.pulse_peak_time + self.pulse_fwhm
        return (start_time, end_time)


@dataclass
class PulseSequence:
    """
    Container for managing a sequence of optical pulses.

    provides methods for creating, manipulating, and analyzing sequences
    of optical pulses used in spectroscopy simulations. It supports factory methods
    for convenient pulse sequence creation and analysis methods for determining
    pulse overlaps and field strengths.

    Attributes:
        pulses (list): List of Pulse objects in the sequence
    """

    pulses: list = field(default_factory=list)  # List of Pulse objects
    pulse_specs: List[Tuple[int, float, float]] = field(
        default_factory=list
    )  # Optional: store specs

    @staticmethod
    def from_pulse_specs(
        pulse_indices: List[int],
        pulse_peak_times: List[float],
        pulse_phases: List[float],
        pulse_freqs: Union[float, List[float]],
        pulse_fwhms: Union[float, List[float]],
        pulse_amplitudes: Union[float, List[float]],
        pulse_type: str = "cos2",
    ) -> "PulseSequence":
        """
        Factory method to create a PulseSequence from pulse specifications.

        This method directly specifies pulse indices, times, and phases
        providing a clean and intuitive interface for pulse sequence creation.

        Parameters:
            pulse_indices (List[int]): List of pulse indices for parameter lookup
            pulse_peak_times (List[float]): Times when pulses reach maximum intensity [fs]
            pulse_phases (List[float]): Phase offsets of the pulses [rad]
            pulse_freqs (Union[float, List[float]]): Central frequency(ies) [rad/fs]
                If float: same frequency for all pulses
                If List: individual frequency for each pulse index
            pulse_fwhms (Union[float, List[float]]): FWHM duration(s) [fs]
                If float: same FWHM for all pulses
                If List: individual FWHM for each pulse index
            pulse_amplitudes (Union[float, List[float]]): Amplitude(s) []
                If float: same amplitude for all pulses
                If List: individual amplitude for each pulse index
            pulse_type (str): Envelope shape: 'cos2' or 'gaussian'

        Returns:
            PulseSequence: An instance containing the specified pulses

        Example:
            # Create a 3-pulse sequence with individual parameters
            seq = PulseSequence.from_pulse_specs(
                pulse_indices=[0, 1, 2],
                pulse_peak_times=[100.0, 200.0, 300.0],
                pulse_phases=[0.0, 1.57, 0.5],
                pulse_freqs=[1.0, 1.1, 1.2],
                pulse_fwhms=[50.0, 60.0, 55.0],
                pulse_amplitudes=[1e6, 1.2e6, 0.8e6]
            )
        """
        # Validate input lengths
        if not pulse_indices:
            raise ValueError("pulse_indices cannot be empty")

        if len(pulse_peak_times) != len(pulse_indices):
            raise ValueError("Length of pulse_peak_times must match pulse_indices")

        if len(pulse_phases) != len(pulse_indices):
            raise ValueError("Length of pulse_phases must match pulse_indices")

        # Convert single values to lists if needed
        max_index = max(pulse_indices)

        if isinstance(pulse_freqs, (int, float)):
            freq_list = [pulse_freqs] * (max_index + 1)
        else:
            freq_list = list(pulse_freqs)

        if isinstance(pulse_fwhms, (int, float)):
            fwhm_list = [pulse_fwhms] * (max_index + 1)
        else:
            fwhm_list = list(pulse_fwhms)

        if isinstance(pulse_amplitudes, (int, float)):
            amp_list = [pulse_amplitudes] * (max_index + 1)
        else:
            amp_list = list(pulse_amplitudes)

        # Validate parameter list lengths
        for pulse_idx in pulse_indices:
            if (
                pulse_idx >= len(fwhm_list)
                or pulse_idx >= len(amp_list)
                or pulse_idx >= len(freq_list)
            ):
                raise ValueError(
                    f"Pulse index {pulse_idx} exceeds available parameters"
                )

        # Create pulses from specifications
        pulses = []
        pulse_specs = []  # Keep for backward compatibility

        for pulse_index, peak_time, phase in zip(
            pulse_indices, pulse_peak_times, pulse_phases
        ):
            pulses.append(
                Pulse(
                    pulse_peak_time=peak_time,
                    pulse_phase=phase,
                    pulse_fwhm=fwhm_list[pulse_index],
                    pulse_amplitude=amp_list[pulse_index],
                    pulse_freq=freq_list[pulse_index],
                    pulse_index=pulse_index,
                    pulse_type=pulse_type,
                )
            )
            pulse_specs.append((pulse_index, peak_time, phase))

        return PulseSequence(pulses=pulses, pulse_specs=pulse_specs)

    @staticmethod
    def create_sequence(
        times: List[float],
        phases: Optional[List[float]] = None,
        indices: Optional[List[int]] = None,
        pulse_freqs: Union[float, List[float]] = 0,
        pulse_fwhms: Union[float, List[float]] = 15.0,
        pulse_amplitudes: Union[float, List[float]] = 1,
        pulse_type: str = "cos2",
    ) -> "PulseSequence":
        """
        Convenience method to create a PulseSequence with automatic indexing.

        This is the simplest way to create a pulse sequence - just specify the times
        and optionally the phases. Pulse indices are assigned automatically (0, 1, 2, ...).

        Parameters:
            times (List[float]): Peak times for each pulse [fs]
            phases (List[float], optional): Phase for each pulse [rad]. Defaults to all zeros.
            indices (List[int], optional): Pulse indices. Defaults to [0, 1, 2, ...]
            pulse_freqs (Union[float, List[float]]): Central frequency(ies) [rad/fs]
            pulse_fwhms (Union[float, List[float]]): FWHM duration(s) [fs]
            pulse_amplitudes (Union[float, List[float]]): Amplitude(s) []
            pulse_type (str): Envelope shape: 'cos2' or 'gaussian'

        Returns:
            PulseSequence: An instance containing the specified pulses

        Example:
            # Simple 3-pulse sequence with default parameters
            seq = PulseSequence.create_sequence([100.0, 200.0, 300.0])

            # With custom parameters
            seq = PulseSequence.create_sequence(
                times=[100.0, 200.0, 300.0],
                phases=[0.0, 1.57, 0.5],
                pulse_freqs=[1.0, 1.1, 1.2],
                pulse_fwhms=50.0,  # Same FWHM for all
                pulse_amplitudes=[1e6, 1.2e6, 0.8e6]
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

        # Use the new from_pulse_specs method
        return PulseSequence.from_pulse_specs(
            pulse_indices=indices,
            pulse_peak_times=times,
            pulse_phases=phases,
            pulse_freqs=pulse_freqs,
            pulse_fwhms=pulse_fwhms,
            pulse_amplitudes=pulse_amplitudes,
            pulse_type=pulse_type,
        )

    '''
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
    '''

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


def main():
    """
    Demonstration of Pulse and PulseSequence classes usage.

    Shows various ways to create pulse sequences and analyze them.
    """
    print("=" * 60)
    print("PULSE SEQUENCE DEMONSTRATION")
    print("=" * 60)

    # =============================
    # EXAMPLE 1: Simple sequence creation
    # =============================
    print("\n1. Creating a simple 3-pulse sequence:")

    ### Define pulse times
    times = [100.0, 200.0, 350.0]  # fs

    ### Create sequence with default parameters
    simple_seq = PulseSequence.create_sequence(times)

    print(f"   Created sequence with {len(simple_seq.pulses)} pulses")
    for i, pulse in enumerate(simple_seq.pulses):
        print(
            f"   Pulse {i}: t={pulse.pulse_peak_time:.1f} fs, "
            f"FWHM={pulse.pulse_fwhm:.1f} fs, "
            f"A={pulse.pulse_amplitude:.1e}"
        )

    # =============================
    # EXAMPLE 2: Custom parameters for each pulse
    # =============================
    print("\n2. Creating sequence with individual pulse parameters:")

    ### Define pulse parameters
    times = [50.0, 150.0, 250.0, 400.0]  # fs
    phases = [0.0, 1.57, 3.14, 4.71]  # rad (0, π/2, π, 3π/2)
    freqs = [1.0, 1.1, 0.9, 1.05]  # rad/fs
    fwhms = [40.0, 50.0, 60.0, 45.0]  # fs
    amplitudes = [1e6, 1.5e6, 0.8e6, 1.2e6]  #

    ### Create advanced sequence
    advanced_seq = PulseSequence.create_sequence(
        times=times,
        phases=phases,
        pulse_freqs=freqs,
        pulse_fwhms=fwhms,
        pulse_amplitudes=amplitudes,
        pulse_type="gaussian",
    )

    print(f"   Created advanced sequence with {len(advanced_seq.pulses)} pulses")
    for i, pulse in enumerate(advanced_seq.pulses):
        print(
            f"   Pulse {i}: t={pulse.pulse_peak_time:.1f} fs, "
            f"φ={pulse.pulse_phase:.2f} rad, "
            f"ω={pulse.pulse_freq:.2f} rad/fs, "
            f"FWHM={pulse.pulse_fwhm:.1f} fs"
        )

    # =============================
    # EXAMPLE 3: Using pulse specifications
    # =============================
    print("\n3. Creating sequence from pulse specifications:")

    ### Define pulse parameters for each pulse type (index)
    spec_freqs = [1.0, 1.2, 0.8]  # rad/fs for indices 0,1,2
    spec_fwhms = [45.0, 55.0, 35.0]  # fs for indices 0,1,2
    spec_amps = [1.1e6, 1.4e6, 0.9e6]  # for indices 0,1,2

    ### Define the actual pulse sequence
    spec_indices = [0, 1, 0, 2]  # Pulse types to use
    spec_times = [80.0, 180.0, 280.0, 380.0]  # fs
    spec_phases = [0.0, 1.57, 3.14, 0.5]  # rad

    ### Create sequence from specifications
    spec_seq = PulseSequence.from_pulse_specs(
        pulse_indices=spec_indices,
        pulse_peak_times=spec_times,
        pulse_phases=spec_phases,
        pulse_freqs=spec_freqs,
        pulse_fwhms=spec_fwhms,
        pulse_amplitudes=spec_amps,
        pulse_type="cos2",
    )

    print(f"   Created spec sequence with {len(spec_seq.pulses)} pulses")
    for i, pulse in enumerate(spec_seq.pulses):
        print(
            f"   Pulse {i} (type {pulse.pulse_index}): "
            f"t={pulse.pulse_peak_time:.1f} fs, "
            f"φ={pulse.pulse_phase:.2f} rad, "
            f"A={pulse.pulse_amplitude:.1e}"
        )

    # =============================
    # EXAMPLE 4: Analyzing pulse overlaps
    # =============================
    print("\n4. Analyzing pulse field at different times:")

    ### Test times to analyze
    test_times = [75.0, 125.0, 175.0, 225.0, 275.0]  # fs

    print("   Time analysis for advanced sequence:")
    for t in test_times:
        field_info = advanced_seq.get_field_info_at_time(t)
        n_active = field_info["num_active_pulses"]
        total_amp = field_info["total_amplitude"]

        if n_active > 0:
            active_indices = field_info["pulse_indices"]
            print(
                f"   t={t:5.1f} fs: {n_active} active pulse(s) "
                f"(indices {active_indices}), "
                f"total E₀={total_amp:.2e}"
            )
        else:
            print(f"   t={t:5.1f} fs: No active pulses")

    # =============================
    # EXAMPLE 5: Individual pulse properties
    # =============================
    print("\n5. Individual pulse active time ranges:")

    ### Analyze first sequence
    print("   Simple sequence pulse ranges:")
    for i, pulse in enumerate(simple_seq.pulses):
        start_time, end_time = pulse.active_time_range
        duration = end_time - start_time
        print(
            f"   Pulse {i}: active from {start_time:.1f} to {end_time:.1f} fs "
            f"(duration: {duration:.1f} fs)"
        )

    # =============================
    # EXAMPLE 6: Error handling demonstration
    # =============================
    print("\n6. Error handling examples:")

    ### Try to create empty sequence
    try:
        empty_seq = PulseSequence.create_sequence([])
    except ValueError as e:
        print(f"   Empty sequence error: {e}")

    ### Try mismatched parameter lengths
    try:
        bad_seq = PulseSequence.create_sequence(
            times=[100.0, 200.0], phases=[0.0, 1.57, 3.14]  # Wrong length
        )
    except ValueError as e:
        print(f"   Parameter mismatch error: {e}")

    ### Try invalid pulse index in specs
    try:
        bad_spec_seq = PulseSequence.from_pulse_specs(
            pulse_indices=[0, 5],  # Index 5 doesn't exist
            pulse_peak_times=[100.0, 200.0],
            pulse_phases=[0.0, 1.57],
            pulse_freqs=[1.0, 1.1],
            pulse_fwhms=[50.0, 55.0],
            pulse_amplitudes=[1e6, 1.1e6],
        )
    except ValueError as e:
        print(f"   Invalid pulse index error: {e}")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
