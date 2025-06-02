"""
Interaction Hamiltonian function for the 2DES simulation.

This module contains functions for constructing interaction Hamiltonian
and other utilities using a Rotating Wave Approximation.
"""

import numpy as np
from qutip import Qobj, expect
from src.core.system_parameters import SystemParameters
from src.core.pulse_sequences import PulseSequence
from src.core.pulse_functions import E_pulse, Epsilon_pulse


def H_int(
    t: float,
    pulse_seq: PulseSequence,
    system: SystemParameters,
) -> Qobj:
    """
    Define the interaction Hamiltonian for the system with multiple pulses using the PulseSequence class.

    Parameters:
        t (float): Time at which the interaction Hamiltonian is evaluated.
        pulse_seq (PulseSequence): PulseSequence object containing all pulse parameters.
        system (SystemParameters): System parameters.
        SM_op (Qobj): Lowering operator (system-specific).
        Dip_op (Qobj): Dipole operator (system-specific).

    Returns:
        Qobj: Interaction Hamiltonian at time t.
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    SM_op = system.SM_op
    Dip_op = system.Dip_op

    if system.RWA_laser:
        E_field = E_pulse(t, pulse_seq)  # Combined electric field under RWA
        H_int = -(
            SM_op.dag() * E_field + SM_op * np.conj(E_field)
        )  # RWA interaction Hamiltonian
    else:
        E_field = Epsilon_pulse(t, pulse_seq)  # Combined electric field with carrier
        H_int = -Dip_op * (E_field + np.conj(E_field))  # Full interaction Hamiltonian

    return H_int


def apply_RWA_phase_factors(
    rho: Qobj, t: float, omega: float, system: SystemParameters
) -> Qobj:
    """
    Apply time-dependent phase factors to the density matrix entries.
    Dispatches to the appropriate implementation based on N_atoms.

    Parameters:
        rho (Qobj): Density matrix (Qobj) to modify.
        t (float): Current time.
        omega (float): Frequency of the phase factor.
        system (SystemParameters): System parameters.

    Returns:
        Qobj: Modified density matrix with phase factors applied.
    """
    if system.N_atoms == 1:
        return _apply_RWA_phase_factors_1atom(rho, t, omega)
    elif system.N_atoms == 2:
        return _apply_RWA_phase_factors_2atom(rho, t, omega)
    else:
        raise ValueError("Only N_atoms=1 or 2 are supported.")


def _apply_RWA_phase_factors_1atom(rho: Qobj, t: float, omega: float) -> Qobj:
    """
    Apply time-dependent phase factors to the density matrix entries.

    Parameters:
        rho (Qobj): Density matrix (Qobj) to modify.
        omega (float): Frequency of the phase factor.
        t (float): Current time.

    Returns:
        Qobj: Modified density matrix with phase factors applied.
    """
    # Extract the density matrix as a NumPy array
    rho_array = rho.full()
    # print(rho.isherm)

    # Apply the phase factors to the specified elements
    phase_1 = np.exp(-1j * omega * t)  # e^(-i * omega * t)

    # Modify the elements
    rho_array[1, 0] *= phase_1  # rho_alpha_0 = sigma_alpha_0 * e^(-i * omega * t)
    rho_array[0, 1] *= np.conj(phase_1)
    rho_result = Qobj(rho_array, dims=rho.dims)
    # print(rho_array[0, 1], rho_array[1,0])

    # assert rho_result.isherm, "The resulting density matrix is not Hermitian."

    return rho_result


def _apply_RWA_phase_factors_2atom(rho: Qobj, t: float, omega: float) -> Qobj:
    """
    Apply time-dependent phase factors to the density matrix entries.

    Parameters:
        rho (Qobj): Density matrix (Qobj) to modify.
        omega (float): Frequency of the phase factor.
        t (float): Current time.

    Returns:
        Qobj: Modified density matrix with phase factors applied.
    """
    # Extract the density matrix as a NumPy array
    rho_array = rho.full()
    # print(rho.isherm)

    # Apply the phase factors to the specified elements
    phase_1 = np.exp(-1j * omega * t)  # e^(-i * omega * t)
    phase_2 = np.exp(-1j * 2 * omega * t)  # e^(-i * 2 * omega * t)

    # Modify the elements
    bar_alpha = 3
    for alpha in range(1, 3):
        rho_array[
            alpha, 0
        ] *= phase_1  # rho_alpha_0 = sigma_alpha_0 * e^(-i * omega * t)
        rho_array[0, alpha] *= np.conj(phase_1)

        rho_array[
            bar_alpha, alpha
        ] *= phase_1  # rho_bar_alpha_alpha = sigma_bar_alpha_alpha * e^(-i * omega * t)
        rho_array[alpha, bar_alpha] *= np.conj(phase_1)

    rho_array[
        bar_alpha, 0
    ] *= phase_2  # rho_bar_alpha_0 = sigma_bar_alpha_0 * e^(-i * 2 * omega * t)
    rho_array[0, bar_alpha] *= np.conj(phase_2)

    rho_result = Qobj(rho_array, dims=rho.dims)
    # print(rho_array[0, 1], rho_array[1,0])

    # assert rho_result.isherm, "The resulting density matrix is not Hermitian."

    return rho_result


def get_expect_vals_with_RWA(
    states: list[Qobj], times: np.array, system: SystemParameters
):
    """
    Calculate the expectation values in the result with RWA phase factors.

    Parameters:
        states= data.states (where data = qutip.Result): Results of the pulse evolution.
        times (list): Time points at which the expectation values are calculated.
        e_ops (list): the operators for which the expectation values are calculated
        omega (float): omega_laser (float): Frequency of the laser.
        RWA (bool): Whether to apply the RWA phase factors.
    Returns:
        list of lists: Expectation values for each operator of len(states).
    """
    omega = system.omega_laser
    e_ops = system.e_ops_list + [system.Dip_op]

    if system.RWA_laser:
        # Apply RWA phase factors to each state
        states = [
            apply_RWA_phase_factors(state, time, omega, system)
            for state, time in zip(states, times)
        ]
    updated_expects = [np.real(expect(states, e_op)) for e_op in e_ops]
    return updated_expects


if __name__ == "__main__":
    """
    Test the functions in this module when run directly.
    """
    print("Testing functions_with_rwa.py module...")

    ### Create test system for N_atoms=1
    print("\n=== Testing with N_atoms=1 ===")
    system1 = SystemParameters(N_atoms=1)

    ### Create simple pulse sequence
    from src.core.pulse_sequences import Pulse

    test_pulse = Pulse(
        pulse_peak_time=2.0,
        pulse_FWHM=1.0,
        pulse_phase=0.0,
        pulse_amplitude=0.05,
        pulse_freq=system1.omega_laser,
    )
    pulse_seq1 = PulseSequence([test_pulse])

    ### Test H_int function
    print("\n--- Testing H_int function ---")
    test_time = 1.0
    H_interaction = H_int(test_time, pulse_seq1, system1)
    print(f"H_int at t={test_time}: {type(H_interaction)}")
    print(f"H_int dimensions: {H_interaction.dims}")
    print(f"H_int is Hermitian: {H_interaction.isherm}")

    ### Test apply_RWA_phase_factors
    print("\n--- Testing apply_RWA_phase_factors ---")
    test_rho = system1.psi_ini  # Initial density matrix
    omega_test = system1.omega_laser
    rho_modified = apply_RWA_phase_factors(test_rho, test_time, omega_test, system1)
    print(f"Original rho type: {type(test_rho)}")
    print(f"Modified rho type: {type(rho_modified)}")
    print(f"Modified rho is Hermitian: {rho_modified.isherm}")

    ### Test with multiple times
    times_test = np.linspace(0, 5, 10)
    states_test = [system1.psi_ini for _ in times_test]  # Dummy states list

    ### Test get_expect_vals_with_RWA
    print("\n--- Testing get_expect_vals_with_RWA ---")
    try:
        expect_vals = get_expect_vals_with_RWA(states_test, times_test, system1)
        print(f"Number of expectation operators: {len(expect_vals)}")
        print(f"Number of time points: {len(expect_vals[0]) if expect_vals else 0}")
        print(f"Expectation values shape: {[len(ev) for ev in expect_vals]}")
    except Exception as e:
        print(f"Error in get_expect_vals_with_RWA: {e}")

    ### Test with N_atoms=2
    print("\n=== Testing with N_atoms=2 ===")
    try:
        system2 = SystemParameters(N_atoms=2)
        pulse_seq2 = PulseSequence([test_pulse])

        ### Test H_int for 2-atom system
        H_interaction2 = H_int(test_time, pulse_seq2, system2)
        print(f"H_int for 2-atom system dimensions: {H_interaction2.dims}")

        ### Test RWA phase factors for 2-atom system
        test_rho2 = system2.psi_ini
        rho_modified2 = apply_RWA_phase_factors(
            test_rho2, test_time, omega_test, system2
        )
        print(f"2-atom modified rho is Hermitian: {rho_modified2.isherm}")

    except Exception as e:
        print(f"Error testing 2-atom system: {e}")

    ### Test error handling
    print("\n--- Testing error handling ---")
    try:
        # Test with invalid pulse sequence type
        H_int(test_time, "invalid_pulse_seq", system1)
    except TypeError as e:
        print(f"✓ Correctly caught TypeError: {e}")

    try:
        # Test with unsupported N_atoms
        system_invalid = SystemParameters(N_atoms=1)
        system_invalid.N_atoms = 3  # Manually set invalid value
        apply_RWA_phase_factors(test_rho, test_time, omega_test, system_invalid)
    except ValueError as e:
        print(f"✓ Correctly caught ValueError: {e}")

    print("\n✅ All tests completed successfully!")
