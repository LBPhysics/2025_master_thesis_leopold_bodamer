#!/usr/bin/env python3
"""
Test script to verify that 1D spectroscopy data length is consistent
regardless of t_coh values.
"""

import numpy as np
import sys

sys.path.insert(0, "/home/leopold/Projects/Master_thesis/code/python")

from qspectro2d.core.simulation_class import (
    AtomicSystem,
    LaserPulseSequence,
    SimulationConfig,
    SimulationModuleOQS,
)
from qspectro2d.core.bath_system.bath_class import BathSystem
from qspectro2d.spectroscopy.calculations import compute_1d_polarization


def test_length_consistency():
    """Test that different t_coh values produce same length output."""

    # Test parameters
    t_coh_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    t_det_max = 100.0
    t_wait = 0.0
    dt = 0.2

    expected_length = int(np.round(t_det_max / dt)) + 1
    print(f"Expected data length: {expected_length}")

    data_lengths = []

    for t_coh in t_coh_values:
        print(f"\nTesting t_coh = {t_coh:.1f} fs")

        # Create simulation configuration
        atomic_config = {
            "N_atoms": 1,
            "freqs_cm": [16000],
            "dip_moments": [1.0],
            "Delta_cm": 0.0,
        }

        pulse_config = {
            "pulse_fwhm": 15.0,
            "base_amplitude": 0.005,
            "envelope_type": "gaussian",
            "carrier_freq_cm": 16000,
            "delays": [t_coh, t_coh + t_wait],
        }

        simulation_config_dict = {
            "simulation_type": "1d",
            "max_workers": 1,
            "IFT_component": (-1, 1, 1),
            "ODE_Solver": "Paper_eqs",
            "RWA_SL": True,
            "keep_track": "basis",
            "t_coh": t_coh,
            "t_wait": t_wait,
            "t_det_max": t_det_max,
            "dt": dt,
            "n_phases": 4,
            "n_freqs": 1,
        }

        bath_config = {
            "bath_type": "paper",
            "Temp": 1e-5,
            "cutoff_": 1e2,
            "gamma_0": 1 / 300.0,
            "gamma_phi": 1 / 100.0,
        }

        # Create simulation objects
        system = AtomicSystem.from_dict(atomic_config)
        laser = LaserPulseSequence.from_delays(**pulse_config)
        bath = BathSystem.from_dict(bath_config)

        sim_oqs = SimulationModuleOQS(
            simulation_config=SimulationConfig(**simulation_config_dict),
            system=system,
            laser=laser,
            bath=bath,
        )

        try:
            # Compute 1D polarization
            data = compute_1d_polarization(sim_oqs)
            actual_length = len(data)
            data_lengths.append(actual_length)

            print(f"  Data length: {actual_length}")
            print(f"  Expected: {expected_length}")
            print(f"  Match: {actual_length == expected_length}")

        except Exception as e:
            print(f"  ERROR: {e}")
            data_lengths.append(None)

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY:")
    print(f"Expected length: {expected_length}")
    print("Results:")
    for i, (t_coh, length) in enumerate(zip(t_coh_values, data_lengths)):
        if length is not None:
            status = "✓" if length == expected_length else "✗"
            print(f"  t_coh={t_coh:.1f}: {length} {status}")
        else:
            print(f"  t_coh={t_coh:.1f}: ERROR ✗")

    # Check if all lengths are consistent
    valid_lengths = [l for l in data_lengths if l is not None]
    if len(set(valid_lengths)) == 1 and valid_lengths[0] == expected_length:
        print("\n✅ SUCCESS: All data lengths are consistent!")
        return True
    else:
        print("\n❌ FAILURE: Data lengths are inconsistent!")
        return False


if __name__ == "__main__":
    success = test_length_consistency()
    sys.exit(0 if success else 1)
