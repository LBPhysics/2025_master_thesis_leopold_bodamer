import numpy as np
from typing import Union, List, Sequence
from qutip import Qobj, expect


# ROTATING WAVE APPROXIMATION FUNCTIONS
def apply_RWA_phase_factors(
    states_or_rho: Union[List[Qobj], Qobj, Sequence[Qobj], np.ndarray],
    times_or_t: Union[np.ndarray, Sequence[float], float],
    n_atoms: int,
    omega_laser: float,
) -> Union[List[Qobj], Qobj]:
    # Single-state path
    if isinstance(states_or_rho, Qobj) and isinstance(times_or_t, (float, int, np.floating)):
        return _apply_single_rwa(states_or_rho, float(times_or_t), n_atoms, omega_laser)

    # Sequence path: accept list/tuple/ndarray of states and times
    if isinstance(states_or_rho, (list, tuple, np.ndarray)) and isinstance(
        times_or_t, (np.ndarray, list, tuple)
    ):
        # Normalize states to a Python list of Qobj
        states_list = (
            list(states_or_rho)
            if not isinstance(states_or_rho, np.ndarray)
            else list(states_or_rho.ravel())
        )

        # Validate all states are Qobj
        if not all(isinstance(s, Qobj) for s in states_list):
            raise TypeError("All states must be Qobj instances.")

        # Normalize times to 1D numpy array of floats
        times_arr = np.asarray(times_or_t, dtype=float).reshape(-1)

        if len(states_list) != times_arr.shape[0]:
            raise ValueError(
                f"Length mismatch: {len(states_list)} states vs {times_arr.shape[0]} times"
            )

        return [
            _apply_single_rwa(rho, t, n_atoms, omega_laser)
            for rho, t in zip(states_list, times_arr)
        ]

    raise TypeError("Invalid input. Expected (Qobj, float) or (Sequence[Qobj], array-like times).")


def _apply_single_rwa(rho: Qobj, t: float, n_atoms: int, omega: float) -> Qobj:
    """
    Apply RWA phase factors for arbitrary n_atoms (supports up to 2 excitations).
    Phase: exp(-i * ω * t * (n_i - n_j)). Assumes ω in 1/fs and t in fs (dimensionless product).
    """
    rho_array = rho.full()
    dim = rho_array.shape[0]

    # Build excitation number vector n for basis ordering [0-ex, 1-ex (n_atoms states), 2-ex (rest)]
    n = np.zeros(dim, dtype=int)
    # 1-ex manifold: indices [1 .. min(n_atoms, dim-1)]
    one_ex_end = min(n_atoms, dim - 1)
    if one_ex_end >= 1:
        n[1 : one_ex_end + 1] = 1
    # 2-ex manifold: indices [n_atoms+1 .. dim-1], only if there is room
    two_ex_start = n_atoms + 1
    if dim > two_ex_start:
        n[two_ex_start:] = 2

    # Vectorized phase application
    delta_n = n[:, None] - n[None, :]
    phase = np.exp(-1j * delta_n * omega * t)
    rho_array = rho_array * phase

    return Qobj(rho_array, dims=rho.dims)


def get_expect_vals_with_RWA(
    states: List[Qobj],
    times: np.ndarray,
    n_atoms: int,
    e_ops: List[Qobj],
    omega_laser: float,
    rwa_sl: bool,
    dipole_op: Qobj = None,
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
        dipole_op: Dipole operator to include in expectation values

    Returns:
        List of arrays containing expectation values for each operator
    """
    # if dipole_op is not None:
    #     e_ops = e_ops + [dipole_op]  # Avoid modifying the original list

    if rwa_sl:
        states = apply_RWA_phase_factors(states, times, n_atoms, omega_laser)

    ## Calculate expectation values for each state and each operator
    updated_expects = []
    for e_op in e_ops:
        # Calculate expectation value for each state with this operator
        expect_vals = np.array(np.real(expect(e_op, states)))
        updated_expects.append(expect_vals)
    if dipole_op is not None:
        # Import locally to avoid circular imports and depend directly on polarization module
        from qspectro2d.spectroscopy.polarization import complex_polarization

        # Calculate expectation value for the dipole operator if provided
        expect_vals_dip = np.array(complex_polarization(dipole_op, states))
        updated_expects.append(expect_vals_dip)

    return updated_expects
