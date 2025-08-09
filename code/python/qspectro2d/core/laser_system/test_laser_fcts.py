"""Tests for laser_fcts.py functions.

Covers:
- _single_pulse_envelope (cos2 & gaussian)
- pulse_envelope (scalar vs vector, multi-pulse combination)
- E_pulse (phase handling)
- Epsilon_pulse (carrier inclusion)
"""

import numpy as np
import pytest

from qspectro2d.core.laser_system.laser_class import LaserPulse, LaserPulseSequence
from qspectro2d.core.laser_system import laser_fcts
from qspectro2d.core.laser_system.laser_fcts import (
    pulse_envelope,
    E_pulse,
    Epsilon_pulse,
)


# =============================
# Helpers
# =============================


def _make_pulse(
    index: int,
    peak: float,
    fwhm: float,
    amp: float,
    freq_cm: float,
    phase: float,
    env: str,
):
    return LaserPulse(
        pulse_index=index,
        pulse_peak_time=peak,
        pulse_phase=phase,
        pulse_fwhm=fwhm,
        pulse_amplitude=amp,
        pulse_freq=freq_cm,
        envelope_type=env,
    )


# =============================
# _single_pulse_envelope
# =============================


def test_single_pulse_envelope_cos2_peak_and_edges():
    fwhm = 10.0
    peak_t = 50.0
    pulse = _make_pulse(0, peak_t, fwhm, 1.0, 16000.0, 0.0, "cos2")
    t = np.linspace(peak_t - fwhm * 1.5, peak_t + fwhm * 1.5, 301)

    env_vals = laser_fcts._single_pulse_envelope(t, pulse)

    # Peak should be ~1
    peak_idx = np.abs(t - peak_t).argmin()
    assert np.isclose(env_vals[peak_idx], 1.0, atol=1e-12)

    # At +/- FWHM envelope should be zero (within numerical tolerance)
    edge_minus_idx = np.abs(t - (peak_t - fwhm)).argmin()
    edge_plus_idx = np.abs(t - (peak_t + fwhm)).argmin()
    assert env_vals[edge_minus_idx] == pytest.approx(0.0, abs=1e-12)
    assert env_vals[edge_plus_idx] == pytest.approx(0.0, abs=1e-12)

    # Outside window also zero
    assert env_vals[0] == 0.0 and env_vals[-1] == 0.0


def test_single_pulse_envelope_gaussian_peak_and_baseline():
    fwhm = 8.0
    peak_t = 20.0
    pulse = _make_pulse(0, peak_t, fwhm, 1.0, 16000.0, 0.0, "gaussian")
    # Cover extended window (≈1.094 * FWHM) with margin
    extended = pulse._t_end - pulse.pulse_peak_time
    t = np.linspace(peak_t - 1.2 * extended, peak_t + 1.2 * extended, 601)

    env_vals = laser_fcts._single_pulse_envelope(t, pulse)

    sigma = pulse._sigma
    boundary_val = pulse._boundary_val  # now value at extended edge
    peak_expected = 1.0 - boundary_val

    peak_idx = np.abs(t - peak_t).argmin()
    assert np.isclose(env_vals[peak_idx], peak_expected, rtol=1e-10)

    # At ±FWHM value should be positive (tails retained)
    fwhm_minus_idx = np.abs(t - (peak_t - fwhm)).argmin()
    fwhm_plus_idx = np.abs(t - (peak_t + fwhm)).argmin()
    assert env_vals[fwhm_minus_idx] > 0.0
    assert env_vals[fwhm_plus_idx] > 0.0

    # Near extended edge the envelope should approach zero
    edge_minus_idx = np.abs(t - (peak_t - extended)).argmin()
    edge_plus_idx = np.abs(t - (peak_t + extended)).argmin()
    assert env_vals[edge_minus_idx] == pytest.approx(0.0, abs=1e-6)
    assert env_vals[edge_plus_idx] == pytest.approx(0.0, abs=1e-6)


# =============================
# pulse_envelope
# =============================


def test_pulse_envelope_scalar_vs_array():
    pulse = _make_pulse(0, 0.0, 10.0, 1.0, 16000.0, 0.0, "cos2")
    seq = LaserPulseSequence([pulse])
    t_arr = np.linspace(-15, 15, 121)

    env_arr = pulse_envelope(t_arr, seq)
    # Compare each scalar call to array result
    for i, t in enumerate(t_arr):
        env_scalar = pulse_envelope(float(t), seq)
        assert env_scalar == pytest.approx(env_arr[i])


def test_pulse_envelope_multiple_pulses_is_additive():
    p1 = _make_pulse(0, 0.0, 6.0, 1.0, 16000.0, 0.0, "cos2")
    p2 = _make_pulse(1, 30.0, 6.0, 1.0, 16000.0, 0.0, "cos2")
    seq = LaserPulseSequence([p1, p2])
    t = np.array([0.0, 30.0])

    env_vals = pulse_envelope(t, seq)
    assert env_vals.shape == (2,)
    # At each peak only its own pulse contributes
    assert env_vals[0] == pytest.approx(1.0, abs=1e-12)
    assert env_vals[1] == pytest.approx(1.0, abs=1e-12)


# =============================
# E_pulse & Epsilon_pulse
# =============================


def test_E_pulse_phase_factor():
    phi = 0.3
    E0 = 2.5
    pulse = _make_pulse(0, 10.0, 5.0, E0, 16000.0, phi, "cos2")
    seq = LaserPulseSequence([pulse])

    val = E_pulse(10.0, seq)  # peak
    # envelope at peak = 1 → should equal E0 * exp(-i phi)
    expected = E0 * np.exp(-1j * phi)
    assert val == pytest.approx(expected)


def test_Epsilon_pulse_includes_carrier():
    phi = 0.1
    E0 = 1.2
    peak = 5.0
    pulse = _make_pulse(0, peak, 4.0, E0, 16000.0, phi, "cos2")
    seq = LaserPulseSequence([pulse])

    t_val = peak
    val = Epsilon_pulse(t_val, seq)

    # Manual expectation: E0 * envelope(peak)=E0 * exp(-i*(omega t + phi))
    omega = pulse.pulse_freq
    expected = E0 * np.exp(-1j * (omega * t_val + phi))
    assert np.isclose(val, expected)


def test_E_pulse_vectorization_matches_loop():
    pulse = _make_pulse(0, 0.0, 5.0, 1.0, 16000.0, 0.0, "cos2")
    seq = LaserPulseSequence([pulse])
    t_arr = np.linspace(-10, 10, 51)

    vec_vals = E_pulse(t_arr, seq)
    loop_vals = np.array([E_pulse(float(t), seq) for t in t_arr])
    assert np.allclose(vec_vals, loop_vals)


def test_Epsilon_pulse_vectorization_matches_loop():
    pulse = _make_pulse(0, 0.0, 5.0, 1.0, 16000.0, 0.2, "cos2")
    seq = LaserPulseSequence([pulse])
    t_arr = np.linspace(-10, 10, 41)

    vec_vals = Epsilon_pulse(t_arr, seq)
    loop_vals = np.array([Epsilon_pulse(float(t), seq) for t in t_arr])
    assert np.allclose(vec_vals, loop_vals)


# =============================
# Error handling
# =============================


def test_pulse_envelope_type_validation():
    # Create a pulse with unsupported envelope type
    bad_pulse = _make_pulse(0, 0.0, 5.0, 1.0, 16000.0, 0.0, "triangle")
    seq = LaserPulseSequence([bad_pulse])
    with pytest.raises(ValueError, match="Unknown envelope_type"):
        pulse_envelope(0.0, seq)


# =============================
# Visualization Helper (optional interactive diagnostic)
# =============================


def _maybe_show():  # pragma: no cover
    """Return True if running interactively (not inside a quiet pytest run)."""
    import os

    return not os.environ.get("PYTEST_CURRENT_TEST")


def visualize_pulse_envelopes_and_fields():  # pragma: no cover
    """Visual comparison: cos^2 vs Gaussian envelopes & fields.

    Generates three stacked plots (envelopes, envelope-only field, carrier field real part).
    Safe to call manually; assertions kept light so it can double as a quick sanity check.
    """
    import matplotlib.pyplot as plt
    from qspectro2d.core.laser_system.laser_fcts import (
        pulse_envelope as _pulse_env,
        E_pulse as _E_pulse,
        Epsilon_pulse as _Eps_pulse,
    )

    peak_t = 0.0
    fwhm = 30.0
    freq = 16000.0
    amp = 1.0
    phase = 0.0

    cos2_pulse = _make_pulse(0, peak_t, fwhm, amp, freq, phase, "cos2")
    gauss_pulse = _make_pulse(1, peak_t, fwhm, amp, freq, phase, "gaussian")
    seq_cos2 = LaserPulseSequence([cos2_pulse])
    seq_gauss = LaserPulseSequence([gauss_pulse])

    t = np.linspace(-60.0, 60.0, 1001)
    env_cos2 = _pulse_env(t, seq_cos2)
    env_gauss = _pulse_env(t, seq_gauss)
    field_cos2 = _E_pulse(t, seq_cos2)
    field_gauss = _E_pulse(t, seq_gauss)
    eps_cos2 = _Eps_pulse(t, seq_cos2)
    eps_gauss = _Eps_pulse(t, seq_gauss)

    # Normalized envelopes for shape comparison
    env_cos2_n = env_cos2 / (env_cos2.max() if env_cos2.max() else 1.0)
    env_gauss_n = env_gauss / (env_gauss.max() if env_gauss.max() else 1.0)

    if _maybe_show():
        plt.figure(figsize=(10, 9))
        # Envelopes
        plt.subplot(3, 1, 1)
        plt.plot(t, env_cos2_n, label=r"cos$^2$", color="C0", linestyle="solid")
        plt.plot(t, env_gauss_n, label="Gaussian", color="C1", linestyle="dashed")
        plt.ylabel(r"$\mathcal{E}(t)/\max$")
        plt.title(r"Pulse Envelopes (FWHM=30 fs)")
        plt.legend()

        # Envelope-only fields (real parts identical to envelopes * amp for phase=0)
        plt.subplot(3, 1, 2)
        plt.plot(t, np.real(field_cos2), label=r"Re $E_{cos^2}$", color="C0")
        plt.plot(t, np.real(field_gauss), label=r"Re $E_{Gauss}$", color="C1")
        plt.ylabel(r"$E(t)$")
        plt.legend()

        # Carrier-including fields (real part)
        plt.subplot(3, 1, 3)
        plt.plot(t, np.real(eps_cos2), label=r"Re $\varepsilon_{cos^2}$", color="C0")
        plt.plot(t, np.real(eps_gauss), label=r"Re $\varepsilon_{Gauss}$", color="C1")
        plt.xlabel(r"Time $t$ (fs)")
        plt.ylabel(r"$\mathrm{Re}[\varepsilon(t)]$")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Light sanity checks
    assert env_cos2_n.max() == pytest.approx(1.0)
    assert env_gauss_n.max() == pytest.approx(1.0)
    # Cos2 strictly zero outside ±FWHM, Gaussian retains tails
    outside = np.abs(t) > fwhm + 1e-9
    assert np.all(env_cos2[outside] == 0.0)
    assert env_gauss[outside].sum() > 0.0


if __name__ == "__main__":  # pragma: no cover
    visualize_pulse_envelopes_and_fields()
    pytest.main([__file__])
