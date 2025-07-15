"""
Tests for the LaserSystem classes and functions.

This module contains comprehensive tests for the LaserPulse and LaserPulseSequence classes,
including initialization, properties, factory methods, and utility functions.
"""

import pytest
import numpy as np
from qspectro2d.core.laser_system.laser_class import (
    LaserPulse,
    LaserPulseSequence,
    identify_non_zero_pulse_regions,
    split_by_active_regions,
)
from qspectro2d.core.utils_and_config import convert_cm_to_fs


class TestLaserPulse:
    """Test LaserPulse class initialization and properties."""

    def test_basic_initialization(self):
        """Test basic LaserPulse initialization."""
        pulse = LaserPulse(
            pulse_index=0,
            pulse_peak_time=100.0,
            pulse_phase=0.0,
            pulse_fwhm=10.0,
            pulse_amplitude=0.05,
            pulse_freq=16000.0,  # Will be converted to rad/fs
            envelope_type="gaussian",
        )

        assert pulse.pulse_index == 0
        assert pulse.pulse_peak_time == 100.0
        assert pulse.pulse_phase == 0.0
        assert pulse.pulse_fwhm == 10.0
        assert pulse.pulse_amplitude == 0.05
        assert pulse.envelope_type == "gaussian"
        assert pulse._freq_converted == True
        assert np.isclose(pulse.pulse_freq, convert_cm_to_fs(16000.0))

        print("✓ LaserPulse basic initialization successful")
        print(f"  - Index: {pulse.pulse_index}")
        print(f"  - Peak time: {pulse.pulse_peak_time} fs")
        print(f"  - Phase: {pulse.pulse_phase} rad")
        print(f"  - FWHM: {pulse.pulse_fwhm} fs")
        print(f"  - Amplitude: {pulse.pulse_amplitude}")
        print(f"  - Frequency: {pulse.pulse_freq:.6f} rad/fs")
        print(f"  - Envelope: {pulse.envelope_type}")

    def test_frequency_conversion(self):
        """Test automatic frequency conversion from cm^-1 to rad/fs."""
        freq_cm = 16000.0
        pulse = LaserPulse(
            pulse_index=0,
            pulse_peak_time=0.0,
            pulse_phase=0.0,
            pulse_fwhm=10.0,
            pulse_amplitude=1.0,
            pulse_freq=freq_cm,
            envelope_type="gaussian",
        )

        expected_freq = convert_cm_to_fs(freq_cm)
        assert np.isclose(pulse.pulse_freq, expected_freq)
        assert pulse.omega_laser == pulse.pulse_freq

        print("✓ LaserPulse frequency conversion successful")
        print(f"  - Original: {freq_cm} cm^-1")
        print(f"  - Converted: {pulse.pulse_freq:.6f} rad/fs")
        print(f"  - omega_laser: {pulse.omega_laser:.6f} rad/fs")

    def test_active_time_range(self):
        """Test active time range calculation."""
        pulse = LaserPulse(
            pulse_index=0,
            pulse_peak_time=100.0,
            pulse_phase=0.0,
            pulse_fwhm=10.0,
            pulse_amplitude=0.05,
            pulse_freq=16000.0,
            envelope_type="gaussian",
        )

        start, end = pulse.active_time_range
        expected_duration = 2 * 1.094 * 10.0  # 2 * n_fwhm * fwhm

        assert np.isclose(end - start, expected_duration)
        assert np.isclose(start, 100.0 - 1.094 * 10.0)
        assert np.isclose(end, 100.0 + 1.094 * 10.0)

        print("✓ LaserPulse active time range calculation successful")
        print(f"  - Peak time: {pulse.pulse_peak_time} fs")
        print(f"  - FWHM: {pulse.pulse_fwhm} fs")
        print(f"  - Active range: [{start:.2f}, {end:.2f}] fs")
        print(f"  - Duration: {end - start:.2f} fs")

    def test_validation_errors(self):
        """Test validation errors for invalid parameters."""
        # Test negative FWHM
        with pytest.raises(ValueError, match="Pulse FWHM must be positive"):
            LaserPulse(
                pulse_index=0,
                pulse_peak_time=0.0,
                pulse_phase=0.0,
                pulse_fwhm=-5.0,  # Invalid
                pulse_amplitude=1.0,
                pulse_freq=16000.0,
            )

        # Test infinite amplitude
        with pytest.raises(ValueError, match="Pulse amplitude must be finite"):
            LaserPulse(
                pulse_index=0,
                pulse_peak_time=0.0,
                pulse_phase=0.0,
                pulse_fwhm=10.0,
                pulse_amplitude=np.inf,  # Invalid
                pulse_freq=16000.0,
            )

        # Test negative frequency
        with pytest.raises(ValueError, match="Pulse frequency must be positive"):
            LaserPulse(
                pulse_index=0,
                pulse_peak_time=0.0,
                pulse_phase=0.0,
                pulse_fwhm=10.0,
                pulse_amplitude=1.0,
                pulse_freq=-16000.0,  # Invalid
            )

        print("✓ LaserPulse validation errors work correctly")

    def test_summary_line(self):
        """Test summary line formatting."""
        pulse = LaserPulse(
            pulse_index=1,
            pulse_peak_time=123.45,
            pulse_phase=1.57,
            pulse_fwhm=12.5,
            pulse_amplitude=0.025,
            pulse_freq=16000.0,
            envelope_type="cos2",
        )

        summary = pulse.summary_line()

        # Check that key information is present
        assert "Pulse  1" in summary
        assert "123.45" in summary
        assert "2.500e-02" in summary
        assert "12.5" in summary
        assert "1.570" in summary
        assert "cos2" in summary

        print("✓ LaserPulse summary line formatting successful")
        print(f"  - Summary: {summary}")

    def test_serialization(self):
        """Test pulse serialization and deserialization."""
        original_pulse = LaserPulse(
            pulse_index=2,
            pulse_peak_time=50.0,
            pulse_phase=0.5,
            pulse_fwhm=8.0,
            pulse_amplitude=0.1,
            pulse_freq=15800.0,
            envelope_type="gaussian",
        )

        # Serialize to dict
        pulse_dict = original_pulse.to_dict()

        # Deserialize from dict
        reconstructed_pulse = LaserPulse.from_dict(pulse_dict)

        # Compare essential attributes
        assert reconstructed_pulse.pulse_index == original_pulse.pulse_index
        assert reconstructed_pulse.pulse_peak_time == original_pulse.pulse_peak_time
        assert reconstructed_pulse.pulse_phase == original_pulse.pulse_phase
        assert reconstructed_pulse.pulse_fwhm == original_pulse.pulse_fwhm
        assert reconstructed_pulse.pulse_amplitude == original_pulse.pulse_amplitude
        assert np.isclose(reconstructed_pulse.pulse_freq, original_pulse.pulse_freq)
        assert reconstructed_pulse.envelope_type == original_pulse.envelope_type

        print("✓ LaserPulse serialization successful")
        print(f"  - Original index: {original_pulse.pulse_index}")
        print(f"  - Reconstructed index: {reconstructed_pulse.pulse_index}")
        print(f"  - Dictionary keys: {list(pulse_dict.keys())}")


class TestLaserPulseSequence:
    """Test LaserPulseSequence class and its factory methods."""

    def test_empty_initialization(self):
        """Test initialization of empty pulse sequence."""
        seq = LaserPulseSequence()

        assert len(seq.pulses) == 0
        assert seq.E0 == 0.0
        assert seq.omega_laser is None
        assert seq.pulse_peak_times == []
        assert seq.pulse_amplitudes == []

        print("✓ Empty LaserPulseSequence initialization successful")
        print(f"  - Number of pulses: {len(seq.pulses)}")
        print(f"  - E0: {seq.E0}")
        print(f"  - omega_laser: {seq.omega_laser}")

    def test_basic_initialization_with_pulses(self):
        """Test initialization with a list of pulses."""
        pulse1 = LaserPulse(
            pulse_index=0,
            pulse_peak_time=50.0,
            pulse_phase=0.0,
            pulse_fwhm=10.0,
            pulse_amplitude=0.1,
            pulse_freq=16000.0,
            envelope_type="gaussian",
        )

        pulse2 = LaserPulse(
            pulse_index=1,
            pulse_peak_time=100.0,
            pulse_phase=1.57,
            pulse_fwhm=12.0,
            pulse_amplitude=0.05,
            pulse_freq=16000.0,
            envelope_type="cos2",
        )

        seq = LaserPulseSequence(pulses=[pulse1, pulse2])

        assert len(seq.pulses) == 2
        assert seq.E0 == 0.1  # First pulse amplitude
        assert np.isclose(seq.omega_laser, convert_cm_to_fs(16000.0))
        assert seq.pulse_peak_times == [50.0, 100.0]
        assert seq.pulse_amplitudes == [0.1, 0.05]

        print("✓ LaserPulseSequence initialization with pulses successful")
        print(f"  - Number of pulses: {len(seq.pulses)}")
        print(f"  - E0: {seq.E0}")
        print(f"  - omega_laser: {seq.omega_laser:.6f} rad/fs")
        print(f"  - Peak times: {seq.pulse_peak_times}")
        print(f"  - Amplitudes: {seq.pulse_amplitudes}")

    def test_from_delays_factory(self):
        """Test creation from delays using factory method."""
        delays = [200.0, 300.0]
        seq = LaserPulseSequence.from_delays(
            delays=delays,
            base_amplitude=0.05,
            pulse_fwhm=10.0,
            carrier_freq_cm=15800.0,
            relative_E0s=[1.0, 0.5, 0.1],
            phases=[0.0, 0.5, 1.0],
            envelope_type="gaussian",
        )

        assert len(seq) == 3
        assert np.allclose(seq.pulse_amplitudes, [0.05, 0.025, 0.005])
        assert np.allclose(seq.pulse_fwhms, [10.0] * len(seq))
        assert np.allclose(seq.pulse_freqs, [convert_cm_to_fs(15800)] * len(seq))
        assert seq.pulse_phases == [0.0, 0.5, 1.0]
        assert all(env == "gaussian" for env in seq.envelope_types)

        print("✓ LaserPulseSequence from_delays factory successful")
        print(f"  - Number of pulses: {len(seq)}")
        print(f"  - Peak times: {seq.pulse_peak_times}")
        print(f"  - Amplitudes: {seq.pulse_amplitudes}")
        print(f"  - Phases: {seq.pulse_phases}")
        print(f"  - FWHMs: {seq.pulse_fwhms}")

    def test_from_general_specs_factory(self):
        """Test creation from general specifications."""
        seq = LaserPulseSequence.from_general_specs(
            pulse_peak_times=[10.0, 20.0],
            pulse_phases=[0.0, 1.0],
            pulse_amplitudes=[1.0, 2.0],
            pulse_fwhms=[5.0, 8.0],
            pulse_freqs=[16000.0, 15800.0],
            envelope_types=["gaussian", "cos2"],
        )

        assert len(seq) == 2
        assert seq.pulse_peak_times == [10.0, 20.0]
        assert seq.pulse_phases == [0.0, 1.0]
        assert seq.pulse_amplitudes == [1.0, 2.0]
        assert seq.pulse_fwhms == [5.0, 8.0]
        assert seq.envelope_types == ["gaussian", "cos2"]

        print("✓ LaserPulseSequence from_general_specs factory successful")
        print(f"  - Number of pulses: {len(seq)}")
        print(f"  - Peak times: {seq.pulse_peak_times}")
        print(f"  - Phases: {seq.pulse_phases}")
        print(f"  - Amplitudes: {seq.pulse_amplitudes}")
        print(f"  - FWHMs: {seq.pulse_fwhms}")
        print(f"  - Envelope types: {seq.envelope_types}")

    def test_from_general_specs_scalar_expansion(self):
        """Test scalar parameter expansion in from_general_specs."""
        seq = LaserPulseSequence.from_general_specs(
            pulse_peak_times=[10.0, 20.0, 30.0],
            pulse_phases=0.5,  # Scalar - should expand
            pulse_amplitudes=1.0,  # Scalar - should expand
            pulse_fwhms=10.0,  # Scalar - should expand
            pulse_freqs=16000.0,  # Scalar - should expand
            envelope_types="gaussian",  # Scalar - should expand
        )

        assert len(seq) == 3
        assert seq.pulse_phases == [0.5, 0.5, 0.5]
        assert seq.pulse_amplitudes == [1.0, 1.0, 1.0]
        assert seq.pulse_fwhms == [10.0, 10.0, 10.0]
        assert seq.envelope_types == ["gaussian", "gaussian", "gaussian"]

        print("✓ LaserPulseSequence scalar expansion successful")
        print(f"  - Number of pulses: {len(seq)}")
        print(f"  - Expanded phases: {seq.pulse_phases}")
        print(f"  - Expanded amplitudes: {seq.pulse_amplitudes}")
        print(f"  - Expanded envelope types: {seq.envelope_types}")

    def test_pulse_sorting(self):
        """Test automatic sorting of pulses by peak time."""
        # Create pulses out of order
        pulse1 = LaserPulse(0, 100.0, 0.0, 10.0, 1.0, 16000.0)
        pulse2 = LaserPulse(1, 50.0, 0.0, 10.0, 1.0, 16000.0)  # Earlier time
        pulse3 = LaserPulse(2, 150.0, 0.0, 10.0, 1.0, 16000.0)

        seq = LaserPulseSequence(pulses=[pulse1, pulse2, pulse3])

        # Should be sorted by peak time
        assert seq.pulse_peak_times == [50.0, 100.0, 150.0]

        print("✓ LaserPulseSequence automatic sorting successful")
        print(f"  - Original order: [100.0, 50.0, 150.0]")
        print(f"  - Sorted order: {seq.pulse_peak_times}")

    def test_update_phases(self):
        """Test phase updating functionality."""
        seq = LaserPulseSequence.from_delays([0.0, 10.0])

        # Update phases
        new_phases = [0.5, 1.5, 2.5]  # New phases for each pulse
        seq.update_phases(phases=new_phases)

        assert np.isclose(seq.pulses[0].pulse_phase, 0.5)
        assert np.isclose(seq.pulses[1].pulse_phase, 1.5)
        assert np.allclose(seq.pulse_phases, new_phases)

        print("✓ LaserPulseSequence phase update successful")
        print(f"  - Updated phases: {seq.pulse_phases}")

    def test_update_phases_validation(self):
        """Test phase update validation."""
        seq = LaserPulseSequence.from_delays([0.0])  # Single pulse

        # Should raise error for wrong number of phases
        with pytest.raises(ValueError):
            seq.update_phases(phases=[0.1, 0.2, 0.3])  # Too many phases

        print("✓ LaserPulseSequence phase update validation works")

    def test_active_pulses_at_time(self):
        """Test getting active pulses at a specific time."""
        seq = LaserPulseSequence.from_delays(
            [100.0, 200.0], base_amplitude=1.0, pulse_fwhm=10.0
        )

        # Test at pulse peak time
        active_at_100 = seq.get_active_pulses_at_time(100.0)
        assert len(active_at_100) >= 1
        assert active_at_100[0].pulse_peak_time == 100.0

        # Test at time with no active pulses
        active_at_0 = seq.get_active_pulses_at_time(0.0)
        assert len(active_at_0) == 0

        print("✓ LaserPulseSequence active pulses calculation successful")
        print(f"  - Active pulses at t=100: {len(active_at_100)}")
        print(f"  - Active pulses at t=0: {len(active_at_0)}")

    def test_total_amplitude_at_time(self):
        """Test total amplitude calculation at a specific time."""
        seq = LaserPulseSequence.from_delays(
            [100.0, 200.0], base_amplitude=1.0, pulse_fwhm=10.0
        )

        # Test at pulse peak time
        total_amp_at_100 = seq.get_total_amplitude_at_time(100.0)
        assert total_amp_at_100 >= 1.0

        # Test at time with no active pulses
        total_amp_at_0 = seq.get_total_amplitude_at_time(50.0)
        assert total_amp_at_0 == 0.0

        print("✓ LaserPulseSequence total amplitude calculation successful")
        print(f"  - Total amplitude at t=100: {total_amp_at_100:.6f}")
        print(f"  - Total amplitude at t=0: {total_amp_at_0:.6f}")

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        original_seq = LaserPulseSequence.from_delays(
            [200.0, 300.0],
            base_amplitude=0.05,
            pulse_fwhm=10.0,
            carrier_freq_cm=15800.0,
            relative_E0s=[1.0, 0.5, 0.1],
            phases=[0.0, 0.5, 1.0],
            envelope_type="gaussian",
        )

        # Serialize to dict
        seq_dict = original_seq.to_dict()

        # Deserialize from dict
        reconstructed_seq = LaserPulseSequence.from_dict(seq_dict)

        # Compare essential attributes
        assert len(reconstructed_seq) == len(original_seq)
        assert np.allclose(
            reconstructed_seq.pulse_peak_times, original_seq.pulse_peak_times
        )
        assert np.allclose(
            reconstructed_seq.pulse_amplitudes, original_seq.pulse_amplitudes
        )
        assert np.allclose(reconstructed_seq.pulse_fwhms, original_seq.pulse_fwhms)
        assert np.allclose(reconstructed_seq.pulse_freqs, original_seq.pulse_freqs)
        assert np.allclose(reconstructed_seq.pulse_phases, original_seq.pulse_phases)
        assert reconstructed_seq.envelope_types == original_seq.envelope_types

        print("✓ LaserPulseSequence serialization roundtrip successful")
        print(f"  - Original length: {len(original_seq)}")
        print(f"  - Reconstructed length: {len(reconstructed_seq)}")
        print(f"  - Dictionary keys: {list(seq_dict.keys())}")
        print(f"  - Pulse attributes match: ✓")


class TestLaserUtilityFunctions:
    """Test utility functions for laser pulse handling."""

    def test_identify_non_zero_pulse_regions(self):
        """Test identification of non-zero pulse regions."""
        seq = LaserPulseSequence.from_delays(
            [100.0, 200.0], base_amplitude=1.0, pulse_fwhm=10.0
        )

        times = np.linspace(0, 300, 301)
        active_regions = identify_non_zero_pulse_regions(times, seq)

        assert active_regions.dtype == bool
        assert len(active_regions) == len(times)
        assert np.any(active_regions)  # Should have some active regions

        # Check that active regions are around pulse times
        pulse_times = seq.pulse_peak_times
        for pulse_time in pulse_times:
            idx = np.abs(times - pulse_time).argmin()
            assert active_regions[idx], f"Should be active at pulse time {pulse_time}"

        print("✓ identify_non_zero_pulse_regions successful")
        print(f"  - Time points: {len(times)}")
        print(f"  - Active regions: {np.sum(active_regions)}")
        print(f"  - Pulse times: {pulse_times}")

    def test_split_by_active_regions(self):
        """Test splitting time array by active regions."""
        seq = LaserPulseSequence.from_delays(
            [100.0, 200.0], base_amplitude=1.0, pulse_fwhm=10.0
        )

        times = np.linspace(0, 300, 301)
        active_regions = identify_non_zero_pulse_regions(times, seq)
        segments = split_by_active_regions(times, active_regions)

        assert isinstance(segments, list)
        assert len(segments) > 0
        assert all(isinstance(seg, np.ndarray) for seg in segments)

        # Check that segments cover the original time range
        all_segment_times = np.concatenate(segments) if segments else np.array([])
        assert len(all_segment_times) == len(times)

        print("✓ split_by_active_regions successful")
        print(f"  - Number of segments: {len(segments)}")
        print(f"  - Segment lengths: {[len(seg) for seg in segments]}")
        print(f"  - Total points: {len(all_segment_times)}")

    def test_combined_utility_workflow(self):
        """Test combined workflow of utility functions."""
        # Create a more complex pulse sequence
        seq = LaserPulseSequence.from_delays(
            [50.0, 100.0, 200.0],
            base_amplitude=1.0,
            pulse_fwhm=8.0,
            envelope_type="gaussian",
        )

        times = np.linspace(0, 250, 251)
        active_regions = identify_non_zero_pulse_regions(times, seq)
        segments = split_by_active_regions(times, active_regions)

        # Verify that active segments contain pulse times
        active_segments = [
            seg for seg in segments if len(seg) > 0 and active_regions[times == seg[0]]
        ]

        assert len(active_segments) > 0

        # Check that each pulse time is in an active segment
        for pulse_time in seq.pulse_peak_times:
            found_in_segment = False
            for segment in active_segments:
                if pulse_time >= segment[0] and pulse_time <= segment[-1]:
                    found_in_segment = True
                    break
            assert (
                found_in_segment
            ), f"Pulse time {pulse_time} not found in any active segment"

        print("✓ Combined utility workflow successful")
        print(f"  - Pulse times: {seq.pulse_peak_times}")
        print(f"  - Active segments: {len(active_segments)}")
        print(f"  - All pulse times covered: ✓")


def test_laser_pulse_sequence_len():
    """Test __len__ method of LaserPulseSequence."""
    seq = LaserPulseSequence.from_delays([0.0, 10.0, 20.0])
    assert len(seq) == 4

    empty_seq = LaserPulseSequence()
    assert len(empty_seq) == 0

    print("✓ LaserPulseSequence __len__ method works")
    print(f"  - 3-pulse sequence length: {len(seq)}")
    print(f"  - Empty sequence length: {len(empty_seq)}")


if __name__ == "__main__":
    """Run tests when executed directly."""
    print("=" * 60)
    print("RUNNING LASER SYSTEM TESTS")
    print("=" * 60)

    # Run all test classes
    test_classes = [
        TestLaserPulse,
        TestLaserPulseSequence,
        TestLaserUtilityFunctions,
    ]

    for test_class in test_classes:
        print(f"\n--- Running {test_class.__name__} ---")
        test_instance = test_class()

        # Run all test methods in the class
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                print(f"\n{method_name}:")
                try:
                    method = getattr(test_instance, method_name)
                    method()
                except Exception as e:
                    print(f"  ✗ FAILED: {e}")
                    import traceback

                    traceback.print_exc()

    # Run standalone tests
    print(f"\n--- Running standalone tests ---")
    print(f"\ntest_laser_pulse_sequence_len:")
    try:
        test_laser_pulse_sequence_len()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
