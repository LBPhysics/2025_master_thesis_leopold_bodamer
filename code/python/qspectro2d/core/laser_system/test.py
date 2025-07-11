import numpy as np
import pytest

from qspectro2d.core.laser_system.laser_class import (
    LaserPulse,
    LaserPulseSequence,
    identify_non_zero_pulse_regions,
    split_by_active_regions,
    convert_cm_to_fs,
)


def test_laserpulse_init_and_properties():
    pulse = LaserPulse(
        pulse_index=0,
        pulse_peak_time=100.0,
        pulse_phase=0.0,
        pulse_fwhm=10.0,
        pulse_amplitude=0.05,
        pulse_freq=2 * np.pi * 16000 * 2.998e-5,
        envelope_type="gaussian",
    )
    # Test active_time_range property
    start, end = pulse.active_time_range
    assert np.isclose(end - start, 2 * 1.094 * 10.0)


def test_laserpulsesequence_from_delays_and_dict():
    delays = [100.0, 200.0, 300.0]
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
    assert np.allclose(seq.pulse_peak_times, sorted(delays))
    assert np.allclose(seq.pulse_amplitudes, [0.05, 0.025, 0.005])
    assert np.allclose(seq.pulse_fwhms, [10.0] * len(delays))
    assert np.allclose(seq.pulse_freqs, [convert_cm_to_fs(15800)] * len(delays))
    # Test to_dict and from_dict
    d = seq.to_dict()
    print("Serialized LaserPulseSequence to dict:", d, flush=True)
    seq2 = LaserPulseSequence.from_dict(d)
    assert isinstance(seq2, LaserPulseSequence)
    assert np.allclose(seq2.pulse_peak_times, seq.pulse_peak_times)
    assert np.allclose(seq2.pulse_amplitudes, seq.pulse_amplitudes)
    assert np.allclose(seq2.pulse_fwhms, seq.pulse_fwhms)
    print("freqs:", seq.pulse_freqs, seq2.pulse_freqs, flush=True)
    assert np.allclose(seq2.pulse_freqs, seq.pulse_freqs)


def test_laserpulsesequence_from_general_specs():
    seq = LaserPulseSequence.from_general_specs(
        pulse_peak_times=[10.0, 20.0],
        pulse_phases=[0.0, 1.0],
        pulse_amplitudes=[1.0, 2.0],
        pulse_fwhms=[5.0, 5.0],
        pulse_freqs=[100.0, 100.0],
        envelope_types=["gaussian", "cos2"],
    )
    assert len(seq) == 2
    assert seq.envelope_types == ["gaussian", "cos2"]
    assert seq.pulse_amplitudes == [1.0, 2.0]


def test_update_phases():
    seq = LaserPulseSequence.from_delays([0.0, 10.0])
    seq.update_phases(phases=[0.5, 1.5])
    assert np.isclose(seq.pulses[0].pulse_phase, 0.5)
    assert np.isclose(seq.pulses[1].pulse_phase, 1.5)
    assert np.isclose(seq.pulse_phases[0], 0.5)
    assert np.isclose(seq.pulse_phases[1], 1.5)
    # Should raise if less than two pulses
    seq1 = LaserPulseSequence.from_delays([0.0])
    with pytest.raises(ValueError):
        seq1.update_phases(phases=[0.1, 0.2])


def test_get_active_pulses_and_total_amplitude():
    seq = LaserPulseSequence.from_delays(
        [100.0, 200.0], base_amplitude=1.0, pulse_fwhm=10.0
    )
    t = 100.0
    active = seq.get_active_pulses_at_time(t)
    assert len(active) >= 1
    total_amp = seq.get_total_amplitude_at_time(t)
    assert total_amp >= 1.0


def test_identify_non_zero_pulse_regions_and_split():
    seq = LaserPulseSequence.from_delays(
        [100.0, 200.0], base_amplitude=1.0, pulse_fwhm=10.0
    )
    times = np.linspace(0, 300, 301)
    active_regions = identify_non_zero_pulse_regions(times, seq)
    assert active_regions.dtype == bool
    assert np.any(active_regions)
    segments = split_by_active_regions(times, active_regions)
    assert isinstance(segments, list)
    assert all(isinstance(seg, np.ndarray) for seg in segments)
    # There should be at least one segment with active region
    assert any(
        np.all(active_regions[times == seg[0]]) for seg in segments if len(seg) > 0
    )


def test_laserpulsesequence_to_dict():
    seq = LaserPulseSequence.from_delays(
        [0.0, 10.0], base_amplitude=0.1, pulse_fwhm=5.0
    )
    d = seq.to_dict()
    assert isinstance(d, dict)
    assert "pulses" in d
    assert isinstance(d["pulses"], list)
    assert all("pulse_index" in p for p in d["pulses"])


if __name__ == "__main__":
    pytest.main(["-s", __file__])
