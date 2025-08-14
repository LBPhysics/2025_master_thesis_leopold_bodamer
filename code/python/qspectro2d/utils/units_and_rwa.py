import numpy as np
from typing import Union, List, overload
from qutip import Qobj, expect

# =============================
# ROTATING WAVE APPROXIMATION FUNCTIONS
# =============================


@overload
def apply_RWA_phase_factors(
    states: List[Qobj], times: np.ndarray, n_atoms: int, omega_laser: float
) -> List[Qobj]: ...
@overload
def apply_RWA_phase_factors(
    rho: Qobj, t: float, n_atoms: int, omega_laser: float
) -> Qobj: ...


def apply_RWA_phase_factors(
    states_or_rho: Union[List[Qobj], Qobj],
    times_or_t: Union[np.ndarray, float],
    n_atoms: int,
    omega_laser: float,
) -> Union[List[Qobj], Qobj]:
    """
    Apply RWA phase factors to states.

    Parameters:
        states_or_rho: Either a list of density matrices or a single density matrix
        times_or_t: Either an array of times or a single time value
        n_atoms: Number of atoms in the system
        omega_laser: Laser frequency

    Returns:
        Either a list of modified density matrices or a single modified density matrix
    """
    if isinstance(states_or_rho, Qobj) and isinstance(times_or_t, (float, int)):
        return _apply_single_rwa(states_or_rho, times_or_t, n_atoms, omega_laser)
    elif isinstance(states_or_rho, list) and isinstance(times_or_t, np.ndarray):
        return [
            _apply_single_rwa(rho, t, n_atoms, omega_laser)
            for rho, t in zip(states_or_rho, times_or_t)
        ]
    else:
        raise TypeError(
            "Invalid input. Expected either (Qobj, float) or (List[Qobj], np.ndarray)."
        )


def _apply_single_rwa(rho: Qobj, t: float, n_atoms: int, omega: float) -> Qobj:
    """
    Apply RWA phase factors for arbitrary n_atoms (supports up to 2 excitations).

    Each density matrix element rho_ij picks up a phase factor
    exp(-i * omega * t * (n_i - n_j)), where n_i is excitation number of basis state i.
    """
    from qspectro2d.constants import HBAR

    rho_array = rho.full()
    dim = rho_array.shape[0]

    # Build excitation number list
    excitation_numbers = [0] * dim
    for i in range(1, n_atoms + 1):
        excitation_numbers[i] = 1
    for i in range(n_atoms + 1, dim):
        excitation_numbers[i] = 2

    # Apply phase factors
    for i in range(dim):
        n_i = excitation_numbers[i]
        for j in range(dim):
            n_j = excitation_numbers[j]
            delta_n = n_i - n_j
            if delta_n != 0:
                rho_array[i, j] *= np.exp(-1j * delta_n * HBAR * omega * t)

    return Qobj(rho_array, dims=rho.dims)


def get_expect_vals_with_RWA(
    states: List[Qobj],
    times: np.ndarray,
    n_atoms: int,
    e_ops: List[Qobj],
    omega_laser: float,
    rwa_sl: bool,
    dip_op: Qobj = None,
) -> List[np.ndarray]:
    """
    Calculate the expectation values in the result with RWA phase factors.

    Parameters:
        states: Results of the pulse evolution (data.states from qutip.Result)
        times: Time points at which the expectation values are calculated
        n_atoms: Number of atoms in the system
        e_ops: The operators for which the expectation values are calculated
        omega_laser: Frequency of the laser
        rwa_sl: Whether to apply the RWA phase factors
        dip_op: Dipole operator to include in expectation values

    Returns:
        List of arrays containing expectation values for each operator
    """
    # if dip_op is not None:
    #     e_ops = e_ops + [dip_op]  # Avoid modifying the original list

    if rwa_sl:
        states = apply_RWA_phase_factors(states, times, n_atoms, omega_laser)

    ## Calculate expectation values for each state and each operator
    updated_expects = []
    for e_op in e_ops:
        # Calculate expectation value for each state with this operator
        expect_vals = np.array(np.real(expect(e_op, states)))
        updated_expects.append(expect_vals)
    if dip_op is not None:
        from qspectro2d.spectroscopy.calculations import complex_polarization

        # Calculate expectation value for the dipole operator if provided
        expect_vals_dip = np.array(complex_polarization(dip_op, states))
        updated_expects.append(expect_vals_dip)

    return updated_expects
