"""
Hamiltonian functions for the 2DES simulation.

This module contains functions for constructing interaction Hamiltonians
and other Hamiltonian-related utilities.
"""

import numpy as np
from qutip import Qobj
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
        H_int = -Dip_op * E_field  # Full interaction Hamiltonian

    return H_int
