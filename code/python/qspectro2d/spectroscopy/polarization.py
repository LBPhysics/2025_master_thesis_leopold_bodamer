"""Polarization related helper functions.

Separated from calculations to break circular import chains between
`qspectro2d.utils.units_and_rwa` and `qspectro2d.spectroscopy.calculations`.
"""

from __future__ import annotations
from typing import Union, List
import numpy as np
from qutip import Qobj, ket2dm


def complex_polarization(
    dipole_op: Qobj, state: Union[Qobj, List[Qobj]]
) -> Union[complex, np.ndarray]:
    """Return complex polarization(s) for state(s) given dipole operator.

    Accepts a single Qobj (ket or density matrix) or list of Qobj.
    """
    if isinstance(state, Qobj):
        return _single_qobj__complex_pol(dipole_op, state)
    if isinstance(state, list):
        if len(state) == 0:
            return np.array([], dtype=np.complex64)
        return np.array(
            [_single_qobj__complex_pol(dipole_op, s) for s in state], dtype=np.complex64
        )
    raise TypeError(f"State must be Qobj or list[Qobj], got {type(state)}")


def _single_qobj__complex_pol(dipole_op: Qobj, state: Qobj) -> complex:
    """
    Calculate polarization for a single quantum state or density matrix.

    Parameters
    ----------
    dipole_op : Qobj
        Dipole operator
    state : Qobj
        Quantum state (ket) or density matrix.

    Returns
    -------
    complex
        Complex polarization value.

    Raises
    ------
    TypeError
        If state is not a ket or density matrix.
    """
    rho = ket2dm(state) if state.isket else state
    N = dipole_op.shape[0]
    x = 0j
    # diagonal half
    for i in range(N):
        x += 0.5 * dipole_op[i, i] * rho[i, i]
    # off-diagonal i<j
    for i in range(N):
        for j in range(i + 1, N):
            x += dipole_op[i, j] * rho[j, i]
    return complex(x)


__all__ = ["complex_polarization"]
