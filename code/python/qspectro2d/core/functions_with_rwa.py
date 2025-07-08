"""
Interaction Hamiltonian function for the 2DES simulation.

This module contains functions for constructing interaction Hamiltonian
and other utilities using a Rotating Wave Approximation.
"""

import numpy as np
from typing import Union, List, overload
from qutip import Qobj, expect
from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.core.laser_system.laser_class import LaserPulseSystem


@overload
def apply_RWA_phase_factors(
    states: List[Qobj], times: np.ndarray, N_atoms: int, omega_laser: float
) -> List[Qobj]: ...
@overload
def apply_RWA_phase_factors(
    rho: Qobj, t: float, N_atoms: int, omega_laser: float
) -> Qobj: ...


def apply_RWA_phase_factors(
    states_or_rho: Union[List[Qobj], Qobj],
    times_or_t: Union[np.ndarray, float],
    N_atoms: int,
    omega_laser: float,
) -> Union[List[Qobj], Qobj]:
    """
    Apply RWA phase factors to states.

    Parameters:
        states_or_rho: Either a list of density matrices or a single density matrix
        times_or_t: Either an array of times or a single time value
        system: System parameters

    Returns:
        Either a list of modified density matrices or a single modified density matrix
    """
    if isinstance(states_or_rho, Qobj) and isinstance(times_or_t, (float, int)):
        return _apply_single_rwa(states_or_rho, times_or_t, N_atoms, omega_laser)
    elif isinstance(states_or_rho, list) and isinstance(times_or_t, np.ndarray):
        return [
            _apply_single_rwa(rho, t, N_atoms, omega_laser)
            for rho, t in zip(states_or_rho, times_or_t)
        ]
    else:
        raise TypeError(
            "Invalid input. Expected either (Qobj, float) or (List[Qobj], np.ndarray)."
        )


def _apply_single_rwa(rho: Qobj, t: float, N_atoms: int, omega: float) -> Qobj:
    """
    Apply time-dependent phase factors to the density matrix entries.
    Dispatches to the appropriate implementation based on N_atoms.
    """
    # Extract the density matrix as a NumPy array
    rho_array = rho.full()
    e_m_iwt = np.exp(-1j * omega * t)

    # Apply the phase factors to the specified elements
    if N_atoms == 1:
        rho_array[1, 0] *= e_m_iwt
        rho_array[0, 1] *= np.conj(e_m_iwt)

    elif N_atoms == 2:
        e_m_2iwt = np.exp(-1j * 2 * omega * t)
        bar_alpha = 3
        for alpha in range(1, 3):
            rho_array[alpha, 0] *= e_m_iwt
            rho_array[0, alpha] *= np.conj(e_m_iwt)

            rho_array[bar_alpha, alpha] *= e_m_iwt
            rho_array[alpha, bar_alpha] *= np.conj(e_m_iwt)

        rho_array[bar_alpha, 0] *= e_m_2iwt
        rho_array[0, bar_alpha] *= np.conj(e_m_2iwt)

    else:
        raise ValueError("TODO implement RWA for N_atoms > 2")

    return Qobj(rho_array, dims=rho.dims)


def get_expect_vals_with_RWA(
    states: list[Qobj],
    times: np.array,
    N_atoms: int,
    e_ops: List[Qobj],
    omega_laser: float,
    RWA_SL: bool,
    Dip_op: Qobj = None,
):
    """
    Calculate the expectation values in the result with RWA phase factors.

    Parameters:
        states= data.states (where data = qutip.Result): Results of the pulse evolution.
        times (list): Time points at which the expectation values are calculated.
        e_ops (list): the operators for which the expectation values are calculated
        omega_laser (float): Frequency of the laser.
        RWA_SL (bool): Whether to apply the RWA phase factors.
        Dip_op (Qobj, optional): Dipole operator to include in expectation values.
    Returns:
        list of lists: Expectation values for each operator of len(states).
    """
    if Dip_op is not None:
        e_ops += [Dip_op]

    if RWA_SL:
        states = apply_RWA_phase_factors(states, times, N_atoms, omega_laser)

    # Calculate expectation values for each state and each operator
    # This should return a list where each element corresponds to an operator
    # and contains an array of expectation values (one for each state)

    updated_expects = []
    for e_op in e_ops:
        # Calculate expectation value for each state with this operator
        expect_vals = np.array([np.real(expect(e_op, state)) for state in states])
        updated_expects.append(expect_vals)

    return updated_expects


if __name__ == "__main__":
    """
    Test the functions in this module when run directly.
    """
    print(
        "Testing functions_with_rwa.py module..."
    )  ### Create test system for N_atoms=1
    print("\n=== Testing with N_atoms=1 ===")
    system1 = AtomicSystem(N_atoms=1)

    ### Create simple pulse sequence
    from qspectro2d.core.laser_system.laser_class import Pulse

    test_pulse = Pulse(
        pulse_index=0,
        pulse_type="gaussian",
        pulse_peak_time=2.0,
        pulse_fwhm=1.0,
        pulse_phase=0.0,
        pulse_amplitude=0.05,
        pulse_freq=system1.freqs_cm[0],
    )
    pulse_seq1 = LaserPulseSystem([test_pulse])

    ### Test apply_RWA_phase_factors
    print("\n--- Testing apply_RWA_phase_factors ---")
    test_rho = system1.psi_ini  # Initial density matrix
    test_time = 1.0
    N_atoms = system1.N_atoms
    omega_laser = pulse_seq1.omega_laser

    rho_modified = apply_RWA_phase_factors(test_rho, test_time, N_atoms, omega_laser)
    print(f"Original rho type: {type(test_rho)}")
    print(f"Modified rho type: {type(rho_modified)}")
    print(f"Modified rho is Hermitian: {rho_modified.isherm}")

    ### Test with multiple times
    times_test = np.linspace(0, 5, 10)
    states_test = [system1.psi_ini for _ in times_test]  # Dummy states list

    ### Test get_expect_vals_with_RWA
    print("\n--- Testing get_expect_vals_with_RWA ---")
    try:
        RWA_SL = True  # Set to True to apply RWA phase factors
        Dip_op = system1.Dip_op
        e_ops = system1.basis
        expect_vals = get_expect_vals_with_RWA(
            states_test, times_test, N_atoms, e_ops, omega_laser, RWA_SL, Dip_op
        )
        print(f"Number of expectation operators: {len(expect_vals)}")
        print(f"Number of time points: {len(expect_vals[0]) if expect_vals else 0}")
        print(f"Expectation values shape: {[len(ev) for ev in expect_vals]}")
    except Exception as e:
        print(f"Error in get_expect_vals_with_RWA: {e}")

    ### Test with N_atoms=2
    print("\n=== Testing with N_atoms=2 ===")
    system2 = AtomicSystem(
        N_atoms=2,
        freqs_cm=[system1.freqs_cm[0], system1.freqs_cm[0] + 10],
        dip_moments=[system1.dip_moments[0], system1.dip_moments[0] + 0.1],
    )
    pulse_seq2 = LaserPulseSystem([test_pulse])
    test_time = 1.0

    ### Test RWA phase factors for 2-atom system
    test_rho2 = system2.psi_ini
    N_atoms = system2.N_atoms
    omega_laser = pulse_seq2.omega_laser
    rho_modified2 = apply_RWA_phase_factors(test_rho2, test_time, N_atoms, omega_laser)
    print(f"2-atom modified rho is Hermitian: {rho_modified2.isherm}")
