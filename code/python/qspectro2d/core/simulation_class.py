"""Legacy monolithic simulation module.

This file now primarily re-exports the refactored components located under
`qspectro2d.core.simulation` to preserve backwards compatibility.

The original content has been split into:
    - simulation/config.py
    - simulation/builders.py
    - (future) simulation/liouvillian_paper.py
    - (future) simulation/redfield.py

Only a thin compatibility layer plus still-unmigrated solver helper
functions remain below. New development should target the modular
files instead of extending this one.
"""

from dataclasses import dataclass
import numpy as np
from qutip import Qobj, stacked_index
import warnings

# One-time deprecation warning (emitted when module is first imported)
if not globals().get("_LEGACY_SIMULATION_DEPRECATED", False):
    warnings.warn(
        "qspectro2d.core.simulation_class is deprecated. Import from qspectro2d.core.simulation.* instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _LEGACY_SIMULATION_DEPRECATED = True

from .simulation.config import SimulationConfig  # noqa: F401
from .simulation.builders import SimulationModuleOQS, H_int_  # noqa: F401
from .simulation.liouvillian_paper import matrix_ODE_paper  # noqa: F401
from .simulation.redfield import R_paper  # noqa: F401
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
from qspectro2d.core.atomic_system.system_class import AtomicSystem


## REMOVED: SimulationConfig definition (now in simulation.config)


## REMOVED: SimulationModuleOQS definition (now in simulation.builders)


## REMOVED: H_int_ (now in simulation.builders)


def main():
    """
    Test function for SimulationModuleOQS class functionality.
    Tests H_int_sl method with different system configurations.
    """
    print("\n" + "=" * 80)
    print("TESTING simulation_config MODEL")

    # =============================
    # SETUP TEST CONFIGURATIONS
    # =============================

    ### Create simulation_config configuration
    sim_config = SimulationConfig(
        ode_solver="Paper_BR", rwa_sl=True, dt=0.1, t_coh=50.0
    )

    ### Create test pulse sequence
    test_pulse_seq = LaserPulseSequence.from_general_specs(
        pulse_peak_times=[0.0, 50.0],
        pulse_phases=[0.0, 1.57],  # 0, π/2
        pulse_freqs=[1.0, 1.0],
        pulse_fwhms=[15.0, 15.0],
        pulse_amplitudes=[0.05, 0.05],
        envelope_types="gaussian",
    )

    # =============================
    # TEST 1: Single atom system
    # =============================
    print("\n--- Test 1: Single Atom System ---")

    ### Create 1-atom system

    system_1 = AtomicSystem(n_atoms=1)
    from qutip import OhmicEnvironment

    bath = OhmicEnvironment(T=300, alpha=1.0, wc=100, s=1.0)

    ### Create simulation_config model
    sim_model_1 = SimulationModuleOQS(
        simulation_config=sim_config,
        system=system_1,
        laser=test_pulse_seq,
        bath=bath,
    )
    try:
        ### Create 1-atom system

        system_1 = AtomicSystem(n_atoms=1)
        from qutip import OhmicEnvironment

        bath = OhmicEnvironment(T=300, alpha=1.0, wc=100, s=1.0)

        ### Create simulation_config model
        sim_model_1 = SimulationModuleOQS(
            simulation_config=sim_config,
            system=system_1,
            laser=test_pulse_seq,
            bath=bath,
        )

        ### Test H_int_sl method
        test_time = 1.0
        H_int_1 = sim_model_1.H_int_sl(test_time)

        print(f"✓ Single atom H_int_sl created successfully")
        print(f"  - Type: {type(H_int_1)}")
        print(f"  - Dimensions: {H_int_1.dims}")
        print(f"  - Is Hermitian: {H_int_1.isherm}")
        print(f"  - Matrix shape: {H_int_1.shape}")

    except Exception as e:
        print(f"✗ Error in single atom test: {e}")

    # =============================
    # TEST 2: Two atom system
    # =============================
    print("\n--- Test 2: Two Atom System ---")

    try:
        ### Create 2-atom system
        system_2 = AtomicSystem(
            n_atoms=2, at_freqs_cm=[16000.0, 16100.0], dip_moments=[1.0, 2.0]
        )

        ### Create simulation_config model
        sim_model_2 = SimulationModuleOQS(
            simulation_config=sim_config,
            system=system_2,
            laser=test_pulse_seq,
            bath=bath,
        )

        ### Test H_int_sl method
        H_int_2 = sim_model_2.H_int_sl(test_time)

        print(f"✓ Two atom H_int_sl created successfully")
        print(f"  - Type: {type(H_int_2)}")
        print(f"  - Dimensions: {H_int_2.dims}")
        print(f"  - Is Hermitian: {H_int_2.isherm}")
        print(f"  - Matrix shape: {H_int_2.shape}")

    except Exception as e:
        print(f"✗ Error in two atom test: {e}")

    # =============================
    # TEST 3: Time evolution of H_int_sl
    # =============================
    print("\n--- Test 3: Time Evolution ---")

    try:
        test_times = np.linspace(0, 100, 5)  # Test at different times

        print(f"Testing H_int_sl at different times for single atom system:")
        for t in test_times:
            H_t = sim_model_1.H_int_sl(t)
            max_element = np.max(np.abs(H_t.full()))
            print(f"  t = {t:6.1f} fs: max|H_int_sl| = {max_element:.2e}")

    except Exception as e:
        print(f"✗ Error in time evolution test: {e}")

    # =============================
    # TEST 4: RWA vs non-RWA comparison
    # =============================
    print("\n--- Test 4: RWA vs Non-RWA ---")

    try:
        ### Create non-RWA configuration
        sim_config_no_rwa = SimulationConfig(
            ode_solver="Paper_BR", rwa_sl=False, dt=0.1, t_coh=100.0  # No RWA
        )

        sim_model_no_rwa = SimulationModuleOQS(
            simulation_config=sim_config_no_rwa,
            system=system_1,
            laser=test_pulse_seq,
            bath=bath,
        )

        ### Compare H_int_sl with and without RWA
        H_rwa = sim_model_1.H_int_sl(test_time)
        H_no_rwa = sim_model_no_rwa.H_int_sl(test_time)

        print(f"✓ RWA comparison successful")
        print(f"  - RWA H_int_sl max element: {np.max(np.abs(H_rwa.full())):.2e}")
        print(
            f"  - Non-RWA H_int_sl max element: {np.max(np.abs(H_no_rwa.full())):.2e}"
        )

    except Exception as e:
        print(f"✗ Error in RWA comparison test: {e}")


if __name__ == "__main__":
    main()
    print("All tests completed.")
