"""Redfield tensor helpers (paper variants).

Contains functions to build the (time independent) Redfield super-operator R
used in the custom solvers. Extracted from the legacy monolithic file.
"""

from __future__ import annotations

import numpy as np
from qutip import Qobj, stacked_index

from .builders import SimulationModuleOQS

__all__ = ["R_paper"]


def R_paper(sim_oqs: SimulationModuleOQS) -> Qobj:
    """Dispatcher selecting appropriate implementation based on n_atoms."""
    n_atoms = sim_oqs.system.n_atoms
    if n_atoms == 1:
        return _R_paper_1atom(sim_oqs)
    if n_atoms == 2:
        return _R_paper_2atom(sim_oqs)
    raise ValueError("Only n_atoms=1 or 2 are supported.")


def _R_paper_1atom(sim_oqs: SimulationModuleOQS) -> Qobj:
    """Redfield tensor for a single 2-level system."""
    from qspectro2d.core.bath_system.bath_fcts import bath_to_rates

    size = 2
    idx_00 = stacked_index(size, 0, 0)
    idx_01 = stacked_index(size, 0, 1)
    idx_10 = stacked_index(size, 1, 0)
    idx_11 = stacked_index(size, 1, 1)

    w0 = sim_oqs.system.frequencies[0]
    deph_rate_pure = bath_to_rates(sim_oqs.bath, mode="deph")  # TODO THIS IS NON-SENSE
    down_rate, up_rate = bath_to_rates(
        sim_oqs.bath, w0, mode="decay"
    )  # TODO THIS IS NON-SENSE
    deph_rate_tot = deph_rate_pure + 0.5 * (down_rate + up_rate)

    R = np.zeros((size * size, size * size), dtype=complex)
    R[idx_10, idx_10] = -deph_rate_tot
    R[idx_01, idx_01] = -deph_rate_tot
    R[idx_00, idx_00] = -up_rate
    R[idx_00, idx_11] = down_rate
    R[idx_11, idx_00] = up_rate
    R[idx_11, idx_11] = -down_rate
    return Qobj(R, dims=[[[size], [size]], [[size], [size]]])


def _R_paper_2atom(sim_oqs: SimulationModuleOQS) -> Qobj:
    """Redfield tensor for a coupled dimer (n_atoms=2)."""
    size = 4
    idx_00 = stacked_index(size, 0, 0)
    idx_01 = stacked_index(size, 0, 1)
    idx_02 = stacked_index(size, 0, 2)
    idx_03 = stacked_index(size, 0, 3)
    idx_10 = stacked_index(size, 1, 0)
    idx_11 = stacked_index(size, 1, 1)
    idx_12 = stacked_index(size, 1, 2)
    idx_13 = stacked_index(size, 1, 3)
    idx_20 = stacked_index(size, 2, 0)
    idx_21 = stacked_index(size, 2, 1)
    idx_22 = stacked_index(size, 2, 2)
    idx_23 = stacked_index(size, 2, 3)
    idx_30 = stacked_index(size, 3, 0)
    idx_31 = stacked_index(size, 3, 1)
    idx_32 = stacked_index(size, 3, 2)
    idx_33 = stacked_index(size, 3, 3)

    R = np.zeros((size * size, size * size), dtype=complex)
    omega_laser = sim_oqs.laser.omega_laser

    # One-excitation coherences
    term = -1j * (
        sim_oqs.system.omega_ij(1, 0) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(1, 0)
    R[idx_10, idx_10] = term
    R[idx_01, idx_01] = np.conj(term)
    term = -1j * (
        sim_oqs.system.omega_ij(2, 0) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(2, 0)
    R[idx_20, idx_20] = term
    R[idx_02, idx_02] = np.conj(term)

    # Double-excited coherences
    term = -1j * (
        sim_oqs.system.omega_ij(3, 0) - 2 * omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(3, 0)
    R[idx_30, idx_30] = term
    R[idx_03, idx_03] = np.conj(term)

    # Cross-coherences
    term = -1j * sim_oqs.system.omega_ij(1, 2) - sim_oqs.sb_coupling.Gamma_big_ij(1, 2)
    R[idx_12, idx_12] = term
    R[idx_21, idx_21] = np.conj(term)
    term = -1j * (
        sim_oqs.system.omega_ij(3, 1) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(3, 1)
    R[idx_31, idx_31] = term
    R[idx_13, idx_13] = np.conj(term)
    term = -1j * (
        sim_oqs.system.omega_ij(3, 2) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(3, 2)
    R[idx_32, idx_32] = term
    R[idx_23, idx_23] = np.conj(term)

    # Populations
    R[idx_11, idx_11] = -sim_oqs.sb_coupling.Gamma_big_ij(1, 1)
    R[idx_11, idx_22] = sim_oqs.sb_coupling.gamma_small_ij(1, 2)
    R[idx_22, idx_22] = -sim_oqs.sb_coupling.Gamma_big_ij(2, 2)
    R[idx_22, idx_11] = sim_oqs.sb_coupling.gamma_small_ij(2, 1)
    R[idx_33, :] = -R[idx_00, :] - R[idx_11, :] - R[idx_22, :]

    return Qobj(R, dims=[[[size], [size]], [[size], [size]]])
