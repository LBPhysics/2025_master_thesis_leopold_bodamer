import numpy as np
from typing import Union, List, overload
from qutip import Qobj, expect


def convert_cm_to_fs(value):
    """
    Convert the wavenumber-frequencies from cm^-1 to angular frequency fs^-1

    Parameters:
        value (float): Value in cm^-1

    Returns:
        float: Value in fs^-1
    """
    return value * 2.998 * 2 * np.pi * 10**-5


def convert_fs_to_cm(value):
    """
    Convert angular frequency fs^-1 to wavenumber-frequencies cm^-1

    Parameters:
        value (float): Value in fs^-1

    Returns:
        float: Value in cm^-1
    """
    return value / (2.998 * 2 * np.pi * 10**-5)


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
    Apply time-dependent phase factors to the density matrix entries.
    Dispatches to the appropriate implementation based on n_atoms.
    """
    # Extract the density matrix as a NumPy array
    rho_array = rho.full()
    e_m_iwt = np.exp(-1j * omega * t)

    # Apply the phase factors to the specified elements
    if n_atoms == 1:
        rho_array[1, 0] *= e_m_iwt
        rho_array[0, 1] *= np.conj(e_m_iwt)

    elif n_atoms == 2:
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
        raise ValueError("TODO implement RWA for n_atoms > 2")

    return Qobj(rho_array, dims=rho.dims)


def get_expect_vals_with_RWA(
    states: List[Qobj],
    times: np.ndarray,
    n_atoms: int,
    e_ops: List[Qobj],
    omega_laser: float,
    rwa_sl: bool,
    Dip_op: Qobj = None,
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
        Dip_op: Dipole operator to include in expectation values

    Returns:
        List of arrays containing expectation values for each operator
    """
    if Dip_op is not None:
        e_ops = e_ops + [Dip_op]  # Avoid modifying the original list

    if rwa_sl:
        states = apply_RWA_phase_factors(states, times, n_atoms, omega_laser)

    ### Calculate expectation values for each state and each operator
    updated_expects = []
    for e_op in e_ops:
        # Calculate expectation value for each state with this operator
        expect_vals = np.array([np.real(expect(e_op, state)) for state in states])
        updated_expects.append(expect_vals)

    return updated_expects
