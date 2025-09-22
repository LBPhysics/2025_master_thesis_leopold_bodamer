"""Redfield tensor helpers (paper variants).

Contains functions to build the (time independent) Redfield super-operator R
used in the custom solvers. Extracted from the legacy monolithic file.
"""

from __future__ import annotations

import numpy as np
from qutip import Qobj, stacked_index, BosonicEnvironment

from .simulation_class import SimulationModuleOQS

__all__ = ["redfield_paper"]


# TODO THIS IS NOT USED ANYMORE, DELETE?
def redfield_paper(sim_oqs: SimulationModuleOQS) -> Qobj:
    """Dispatcher selecting appropriate implementation based on n_atoms."""
    n_atoms = sim_oqs.system.n_atoms
    if n_atoms == 1:
        return _redfield_paper_1atom(sim_oqs)
    if n_atoms == 2:
        return _redfield_paper_2atom(sim_oqs)
    raise ValueError("Only n_atoms=1 or 2 are supported.")


def _redfield_paper_1atom(sim_oqs: SimulationModuleOQS) -> Qobj:
    """Redfield tensor for a single 2-level system in eigenbasis."""
    from qspectro2d.core.bath_system.bath_fcts import bath_to_rates

    size = 2
    idx_00 = stacked_index(size, 0, 0)
    idx_01 = stacked_index(size, 0, 1)
    idx_10 = stacked_index(size, 1, 0)
    idx_11 = stacked_index(size, 1, 1)

    w0 = sim_oqs.system.frequencies_fs[0]
    deph_rate_pure = bath_to_rates(sim_oqs.bath, mode="deph")  # TODO THIS IS NON-SENSE
    down_rate, up_rate = bath_to_rates(sim_oqs.bath, w0, mode="decay")  # TODO THIS IS NON-SENSE
    deph_rate_tot = deph_rate_pure + 0.5 * (down_rate + up_rate)

    R = np.zeros((size * size, size * size), dtype=complex)
    R[idx_10, idx_10] = -deph_rate_tot
    R[idx_01, idx_01] = -deph_rate_tot
    R[idx_00, idx_00] = -up_rate
    R[idx_00, idx_11] = down_rate
    R[idx_11, idx_00] = up_rate
    R[idx_11, idx_11] = -down_rate
    return Qobj(R, dims=[[[size], [size]], [[size], [size]]])


def _redfield_paper_2atom(sim_oqs: SimulationModuleOQS) -> Qobj:
    """Redfield tensor for a coupled dimer (n_atoms=2) in eigenbasis ."""
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
    omega_laser = sim_oqs.laser.carrier_freq_fs

    # One-excitation coherences
    term = -1j * (sim_oqs.system.omega_ij(1, 0) - omega_laser) - sim_oqs.sb_coupling.paper_Gamma_ij(
        1, 0
    )
    R[idx_10, idx_10] = term
    R[idx_01, idx_01] = np.conj(term)
    term = -1j * (sim_oqs.system.omega_ij(2, 0) - omega_laser) - sim_oqs.sb_coupling.paper_Gamma_ij(
        2, 0
    )
    R[idx_20, idx_20] = term
    R[idx_02, idx_02] = np.conj(term)

    # Double-excited coherences
    term = -1j * (
        sim_oqs.system.omega_ij(3, 0) - 2 * omega_laser
    ) - sim_oqs.sb_coupling.paper_Gamma_ij(3, 0)
    R[idx_30, idx_30] = term
    R[idx_03, idx_03] = np.conj(term)

    # Cross-coherences
    term = -1j * sim_oqs.system.omega_ij(1, 2) - sim_oqs.sb_coupling.paper_Gamma_ij(1, 2)
    R[idx_12, idx_12] = term
    R[idx_21, idx_21] = np.conj(term)
    term = -1j * (sim_oqs.system.omega_ij(3, 1) - omega_laser) - sim_oqs.sb_coupling.paper_Gamma_ij(
        3, 1
    )
    R[idx_31, idx_31] = term
    R[idx_13, idx_13] = np.conj(term)
    term = -1j * (sim_oqs.system.omega_ij(3, 2) - omega_laser) - sim_oqs.sb_coupling.paper_Gamma_ij(
        3, 2
    )
    R[idx_32, idx_32] = term
    R[idx_23, idx_23] = np.conj(term)

    # Populations
    R[idx_11, idx_11] = -sim_oqs.sb_coupling.paper_Gamma_ij(1, 1)
    R[idx_11, idx_22] = sim_oqs.sb_coupling.paper_gamma_ij(1, 2)
    R[idx_22, idx_22] = -sim_oqs.sb_coupling.paper_Gamma_ij(2, 2)
    R[idx_22, idx_11] = sim_oqs.sb_coupling.paper_gamma_ij(2, 1)
    R[idx_33, :] = -R[idx_00, :] - R[idx_11, :] - R[idx_22, :]

    return Qobj(R, dims=[[[size], [size]], [[size], [size]]])


# TODO THIS IS COMPLETE TRASH, DELETE?
def my_compute_br_tensor(
    a_ops: list[tuple[Qobj, BosonicEnvironment]],
    skew: np.ndarray,
    cutoff: float = np.inf,
    hbar: float = 1.0,
) -> Qobj:
    """Compute Bloch-Redfield tensor (multiple uncorrelated baths, Hermitian A).

    Implements (units with optional ħ):
        R_abcd = - (1/(2 ħ^2)) Σ_α { δ_bd Σ_n A_an A_nc S_α(ω_cn)
                                     - A_ac A_db S_α(ω_ca)
                                     + δ_ac Σ_n A_dn A_nb S_α(ω_dn)
                                     - A_ac A_db S_α(ω_db) }

    Parameters
    ----------
    a_ops : list[(Qobj, BosonicEnvironment)]
        List of (A_α, env_α) system operators (eigenbasis) with their baths.
    skew : np.ndarray
        Frequency difference matrix ω_i - ω_j (shape (N,N)).
    cutoff : float
        Secular cutoff. Terms with |ω_ab - ω_cd| > cutoff are discarded.
        Use np.inf for no secular approximation.

        Returns
    -------
    Qobj
        Redfield super-operator (N^2 x N^2).
    """
    N = a_ops[0][0].shape[0]
    if skew.shape != (N, N):
        raise ValueError("skew must have shape (N,N).")

    R = np.zeros((N * N, N * N), dtype=complex)
    secular = np.isfinite(cutoff)

    for A, env in a_ops:
        A_data = A.full()

        # Optional Hermitian check (can be relaxed if needed)
        if not np.allclose(A_data, A_data.conj().T):
            raise ValueError("Operator A must be Hermitian for this implementation.")

        # Build spectrum matrix S_ij = S(ω_i - ω_j)
        S = np.empty_like(skew, dtype=complex)
        for i in range(N):
            for j in range(N):
                S[i, j] = env.power_spectrum(skew[i, j])

        # Precompute sums:
        # T_ac = Σ_n A_an A_nc S_cn  -> T = A @ (A * S^T)
        # V_db = Σ_n A_dn A_nb S_dn  -> V = (A * S) @ A
        B = A_data * S.T
        C = A_data * S
        T = A_data @ B
        V = C @ A_data

        # Loop over a,b,c,d (O(N^4))
        for a in range(N):
            for b in range(N):
                ab = a * N + b
                for c in range(N):
                    for d in range(N):
                        if secular and abs(skew[a, b] - skew[c, d]) > cutoff:
                            continue
                        cd = c * N + d
                        term = 0.0 + 0.0j
                        if b == d:
                            term += T[a, c]  # δ_bd Σ_n ...
                        term -= A_data[a, c] * A_data[d, b] * S[c, a]  # - A_ac A_db S_ca
                        if a == c:
                            term += V[d, b]  # δ_ac Σ_n ...
                        term -= A_data[a, c] * A_data[d, b] * S[d, b]  # - A_ac A_db S_db
                        R[ab, cd] += term

    scale = -0.5 / (hbar * hbar)
    R *= scale
    return Qobj(R, dims=[[[N], [N]], [[N], [N]]])
