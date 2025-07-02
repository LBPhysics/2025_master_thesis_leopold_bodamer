import numpy as np
from qutip import Qobj, stacked_index
from .system_parameters import SystemParameters
from .pulse_sequences import PulseSequence
from .pulse_functions import E_pulse


# =============================
# "Paper_eqs" OWN ODE SOLVER
# =============================
def matrix_ODE_paper(
    t: float, pulse_seq: PulseSequence, system: SystemParameters
) -> Qobj:
    """
    Dispatches to the appropriate implementation based on N_atoms.
    Solves the equation drho_dt = L(t) * rho,
    in natural units: L = -i/hbar(Hrho - rho H) + R * rho,  with [hbar] = 1 and [R] = [1] = [power Spectrum S(w)] = [all the Gammas: like gamma_phi].
    """
    if system.N_atoms == 1:
        return _matrix_ODE_paper_1atom(t, pulse_seq, system)
    elif system.N_atoms == 2:
        return _matrix_ODE_paper_2atom(t, pulse_seq, system)
    else:
        raise ValueError("Only N_atoms=1 or 2 are supported.")


def _matrix_ODE_paper_1atom(
    t: float, pulse_seq: PulseSequence, system: SystemParameters
) -> Qobj:
    """
    Constructs the matrix L(t) for the equation
    drho_dt = L(t) · vec(rho),   QuTiP-kompatibel (column stacking).
    Uses gamma values from the provided system.

    Parameters:
        t (float): Time at which to evaluate the matrix.
        pulse_seq (PulseSequence): PulseSequence object for the electric field.
        system (SystemParameters): System parameters containing Gamma, gamma_0, and mu_eg.

    Returns:
        Qobj: Liouvillian matrix as a Qobj.
    """
    Et = E_pulse(t, pulse_seq)
    Et_conj = np.conj(Et)
    μ = system.mu_A

    gamma0 = system.gamma_0
    Γ = system.Gamma  # pure dephasing

    size = 2  # 2 states |g>, |e>

    idx_00 = stacked_index(size, row=0, col=0)  # ρ_gg
    idx_01 = stacked_index(size, row=0, col=1)  # ρ_ge
    idx_10 = stacked_index(size, row=1, col=0)  # ρ_eg
    idx_11 = stacked_index(size, row=1, col=1)  # ρ_ee

    L = np.zeros((4, 4), dtype=complex)

    # ----- d/dt ρ_gg
    L[idx_00, idx_11] = gamma0
    L[idx_00, idx_01] = -1j * Et * μ
    L[idx_00, idx_10] = +1j * Et_conj * μ

    # ----- d/dt ρ_ee
    L[idx_11, :] = -L[idx_00, :]  # Ensures trace conservation

    # ----- d/dt ρ_eg
    L[idx_10, idx_00] = +1j * Et * μ  # ρ_gg
    L[idx_10, idx_11] = -1j * Et * μ  # ρ_ee

    L[idx_10, idx_10] = -Γ  # Decay term for coherence

    # ----- d/dt ρ_ge  – complex conjugate
    L[idx_01, idx_00] = -1j * Et_conj * μ  # ρ_gg
    L[idx_01, idx_11] = +1j * Et_conj * μ  # ρ_ee

    L[idx_01, idx_01] = -Γ  # Decay term for coherence

    return Qobj(L, dims=[[[2], [2]], [[2], [2]]])  # 'super' wird aus den Dims erkannt


# carefull i changed this function with GPT
def _matrix_ODE_paper_2atom(
    t: float,
    pulse_seq: PulseSequence,
    system: SystemParameters,
) -> Qobj:
    """
    Column-stacked Liouvillian L(t) such that         d/dt vec(rho) = L(t) · vec(rho)

    Index-Konvention (column stacking, wie in QuTiP):
        vec(rho)[ i + 4*j ]   ↔   rho_{ij}     für i,j = 0…3
    """
    # --------------------------------------------------------------
    # Helpers & short-hands
    # --------------------------------------------------------------
    Et = E_pulse(t, pulse_seq)
    Et_conj = np.conj(Et)

    size = 4  # 4 states |0>, |1>, |2>, |3>

    # Define all indices using stacked_index
    idx_00 = stacked_index(size, row=0, col=0)  # ρ_00
    idx_01 = stacked_index(size, row=0, col=1)  # ρ_01
    idx_02 = stacked_index(size, row=0, col=2)  # ρ_02
    idx_03 = stacked_index(size, row=0, col=3)  # ρ_03
    idx_10 = stacked_index(size, row=1, col=0)  # ρ_10
    idx_11 = stacked_index(size, row=1, col=1)  # ρ_11
    idx_12 = stacked_index(size, row=1, col=2)  # ρ_12
    idx_13 = stacked_index(size, row=1, col=3)  # ρ_13
    idx_20 = stacked_index(size, row=2, col=0)  # ρ_20
    idx_21 = stacked_index(size, row=2, col=1)  # ρ_21
    idx_22 = stacked_index(size, row=2, col=2)  # ρ_22
    idx_23 = stacked_index(size, row=2, col=3)  # ρ_23
    idx_30 = stacked_index(size, row=3, col=0)  # ρ_30
    idx_31 = stacked_index(size, row=3, col=1)  # ρ_31
    idx_32 = stacked_index(size, row=3, col=2)  # ρ_32
    idx_33 = stacked_index(size, row=3, col=3)  # ρ_33

    L = np.zeros((size * size, size * size), dtype=complex)

    # --------------------------------------------------------------
    # 1) Off-diagonal one-excited-coherences
    # --------------------------------------------------------------
    # ρ_10   (|1⟩⟨0|)
    term = -1j * (system.omega_ij(1, 0) - system.omega_laser) - system.Gamma_big_ij(
        1, 0
    )
    L[idx_10, idx_10] = term
    L[idx_10, idx_00] = 1j * Et * system.Dip_op[1, 0]
    L[idx_10, idx_11] = -1j * Et * system.Dip_op[1, 0]
    L[idx_10, idx_12] = -1j * Et * system.Dip_op[2, 0]
    L[idx_10, idx_30] = 1j * Et_conj * system.Dip_op[3, 1]

    # ρ_01   = (ρ_10)†
    L[idx_01, idx_01] = np.conj(term)
    L[idx_01, idx_00] = np.conj(L[idx_10, idx_00])
    L[idx_01, idx_11] = np.conj(L[idx_10, idx_11])
    L[idx_01, idx_21] = np.conj(L[idx_10, idx_12])
    L[idx_01, idx_03] = np.conj(L[idx_10, idx_30])

    # ρ_20   (|2⟩⟨0|)
    term = -1j * (system.omega_ij(2, 0) - system.omega_laser) - system.Gamma_big_ij(
        2, 0
    )
    L[idx_20, idx_20] = term
    L[idx_20, idx_00] = 1j * Et * system.Dip_op[2, 0]
    L[idx_20, idx_22] = -1j * Et * system.Dip_op[2, 0]
    L[idx_20, idx_21] = -1j * Et * system.Dip_op[1, 0]
    L[idx_20, idx_30] = 1j * Et_conj * system.Dip_op[3, 2]

    # ρ_02
    L[idx_02, idx_02] = np.conj(term)
    L[idx_02, idx_00] = np.conj(L[idx_20, idx_00])
    L[idx_02, idx_22] = np.conj(L[idx_20, idx_22])
    L[idx_02, idx_12] = np.conj(L[idx_20, idx_21])
    L[idx_02, idx_03] = np.conj(L[idx_20, idx_30])

    # --------------------------------------------------------------
    # 2) double-excited-coherences
    # --------------------------------------------------------------
    # ρ_30   (|3⟩⟨0|)
    term = -1j * (system.omega_ij(3, 0) - 2 * system.omega_laser) - system.Gamma_big_ij(
        3, 0
    )
    L[idx_30, idx_30] = term
    L[idx_30, idx_10] = 1j * Et * system.Dip_op[3, 1]
    L[idx_30, idx_20] = 1j * Et * system.Dip_op[3, 2]
    L[idx_30, idx_31] = -1j * Et * system.Dip_op[1, 0]
    L[idx_30, idx_32] = -1j * Et * system.Dip_op[2, 0]

    # ρ_03
    L[idx_03, idx_03] = np.conj(term)
    L[idx_03, idx_01] = np.conj(L[idx_30, idx_10])
    L[idx_03, idx_02] = np.conj(L[idx_30, idx_20])
    L[idx_03, idx_13] = np.conj(L[idx_30, idx_31])
    L[idx_03, idx_23] = np.conj(L[idx_30, idx_32])

    # --------------------------------------------------------------
    # 3) cross-coherences inside one excitation manifold
    # --------------------------------------------------------------
    # ρ_12   (|1⟩⟨2|)
    term = -1j * system.omega_ij(1, 2) - system.Gamma_big_ij(1, 2)
    L[idx_12, idx_12] = term
    L[idx_12, idx_02] = 1j * Et * system.Dip_op[1, 0]
    L[idx_12, idx_13] = -1j * Et * system.Dip_op[3, 2]
    L[idx_12, idx_32] = 1j * Et_conj * system.Dip_op[3, 1]
    L[idx_12, idx_10] = -1j * Et_conj * system.Dip_op[2, 0]

    # ρ_21
    L[idx_21, idx_21] = np.conj(term)
    L[idx_21, idx_20] = np.conj(L[idx_12, idx_02])
    L[idx_21, idx_31] = np.conj(L[idx_12, idx_13])
    L[idx_21, idx_23] = np.conj(L[idx_12, idx_32])
    L[idx_21, idx_01] = np.conj(L[idx_12, idx_10])

    # ρ_31   (|3⟩⟨1|)
    term = -1j * (system.omega_ij(3, 1) - system.omega_laser) - system.Gamma_big_ij(
        3, 1
    )
    L[idx_31, idx_31] = term
    L[idx_31, idx_11] = 1j * Et * system.Dip_op[3, 1]
    L[idx_31, idx_21] = 1j * Et * system.Dip_op[3, 2]
    L[idx_31, idx_30] = -1j * Et_conj * system.Dip_op[1, 0]

    # ρ_13
    L[idx_13, idx_13] = np.conj(term)
    L[idx_13, idx_11] = np.conj(L[idx_31, idx_11])
    L[idx_13, idx_12] = np.conj(L[idx_31, idx_21])
    L[idx_13, idx_03] = np.conj(L[idx_31, idx_30])

    # ρ_32   (|3⟩⟨2|)
    term = -1j * (system.omega_ij(3, 2) - system.omega_laser) - system.Gamma_big_ij(
        3, 2
    )
    L[idx_32, idx_32] = term
    L[idx_32, idx_22] = 1j * Et * system.Dip_op[3, 2]
    L[idx_32, idx_12] = 1j * Et * system.Dip_op[3, 1]
    L[idx_32, idx_30] = -1j * Et_conj * system.Dip_op[2, 0]

    # ρ_23
    L[idx_23, idx_23] = np.conj(term)
    L[idx_23, idx_22] = np.conj(L[idx_32, idx_22])
    L[idx_23, idx_21] = np.conj(L[idx_32, idx_12])
    L[idx_23, idx_03] = np.conj(L[idx_32, idx_30])

    # --------------------------------------------------------------
    # 4) Populations (diagonals)
    # --------------------------------------------------------------
    # ρ_00
    L[idx_00, idx_01] = -1j * Et * system.Dip_op[1, 0]
    L[idx_00, idx_02] = -1j * Et * system.Dip_op[2, 0]
    L[idx_00, idx_10] = 1j * Et_conj * system.Dip_op[1, 0]
    L[idx_00, idx_20] = 1j * Et_conj * system.Dip_op[2, 0]

    # ρ_11
    L[idx_11, idx_11] = -system.Gamma_big_ij(1, 1)
    L[idx_11, idx_22] = system.gamma_small_ij(1, 2)
    L[idx_11, idx_01] = 1j * Et * system.Dip_op[1, 0]
    L[idx_11, idx_13] = -1j * Et * system.Dip_op[3, 1]
    L[idx_11, idx_31] = 1j * Et_conj * system.Dip_op[3, 1]
    L[idx_11, idx_10] = -1j * Et_conj * system.Dip_op[1, 0]

    # ρ_22
    L[idx_22, idx_22] = -system.Gamma_big_ij(2, 2)
    L[idx_22, idx_11] = system.gamma_small_ij(2, 1)
    L[idx_22, idx_02] = 1j * Et * system.Dip_op[2, 0]
    L[idx_22, idx_23] = -1j * Et * system.Dip_op[3, 2]
    L[idx_22, idx_32] = 1j * Et_conj * system.Dip_op[3, 2]
    L[idx_22, idx_20] = -1j * Et_conj * system.Dip_op[2, 0]

    # ρ_33  – Spurbedingung: dρ_00 + dρ_11 + dρ_22 + dρ_33 = 0
    L[idx_33, :] = -L[idx_00, :] - L[idx_11, :] - L[idx_22, :]

    return Qobj(L, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]])


# only use the Redfield tensor as a matrix:
def R_paper(system: SystemParameters) -> Qobj:
    """Dispatches to the appropriate implementation based on N_atoms."""
    if system.N_atoms == 1:
        return _R_paper_1atom(system)
    elif system.N_atoms == 2:
        return _R_paper_2atom(system)
    else:
        raise ValueError("Only N_atoms=1 or 2 are supported.")


def _R_paper_1atom(system: SystemParameters) -> Qobj:
    """
    Constructs the Redfield Tensor R for the equation drho_dt = -i(Hrho - rho H) + R * rho,
    where rho is the flattened density matrix. Uses gamma values from the provided system.

    Parameters:
        system (SystemParameters): System parameters containing Gamma and gamma_0.

    Returns:
        Qobj: Redfield tensor as a Qobj.
    """
    R = np.zeros((4, 4), dtype=complex)  # Redfield tensor initialization

    # --- d/dt rho_eg ---
    R[2, 2] = -system.Gamma  # Decay term for coherence
    # --- d/dt rho_ge ---
    R[1, 1] = -system.Gamma

    # --- d/dt rho_ee ---
    R[3, 3] = -system.gamma_0  # Decay term for population
    # --- d/dt rho_gg ---
    R[0, 3] = system.gamma_0  # Ensures trace conservation

    return Qobj(R, dims=[[[2], [2]], [[2], [2]]])


def _R_paper_2atom(system: SystemParameters) -> Qobj:
    """
    including RWA
    Constructs the Redfield Tensor R for the equation drho_dt = -i(Hrho - rho H) + R * rho,
    where rho is the flattened density matrix.
    """
    size = 4  # 4 states |0>, |1>, |2>, |3>

    # Define all indices using stacked_index
    idx_00 = stacked_index(size, row=0, col=0)  # ρ_00
    idx_01 = stacked_index(size, row=0, col=1)  # ρ_01
    idx_02 = stacked_index(size, row=0, col=2)  # ρ_02
    idx_03 = stacked_index(size, row=0, col=3)  # ρ_03
    idx_10 = stacked_index(size, row=1, col=0)  # ρ_10
    idx_11 = stacked_index(size, row=1, col=1)  # ρ_11
    idx_12 = stacked_index(size, row=1, col=2)  # ρ_12
    idx_13 = stacked_index(size, row=1, col=3)  # ρ_13
    idx_20 = stacked_index(size, row=2, col=0)  # ρ_20
    idx_21 = stacked_index(size, row=2, col=1)  # ρ_21
    idx_22 = stacked_index(size, row=2, col=2)  # ρ_22
    idx_23 = stacked_index(size, row=2, col=3)  # ρ_23
    idx_30 = stacked_index(size, row=3, col=0)  # ρ_30
    idx_31 = stacked_index(size, row=3, col=1)  # ρ_31
    idx_32 = stacked_index(size, row=3, col=2)  # ρ_32
    idx_33 = stacked_index(size, row=3, col=3)  # ρ_33

    R = np.zeros((size * size, size * size), dtype=complex)

    # --------------------------------------------------------------
    # 1) Off-diagonal one-excited-coherences
    # --------------------------------------------------------------
    # --- d/dt rho_10 ---
    term = -1j * (system.omega_ij(1, 0) - system.omega_laser) - system.Gamma_big_ij(
        1, 0
    )
    R[idx_10, idx_10] = term  # ρ₁₀ ← ρ₁₀

    # --- d/dt rho_01 ---
    R[idx_01, idx_01] = np.conj(term)  # ρ₀₁ ← ρ₀₁

    # --- d/dt rho_20 --- = ANSATZ = (d/dt s_20 - i omega_laser s_20) e^(-i omega_laser t)
    term = -1j * (system.omega_ij(2, 0) - system.omega_laser) - system.Gamma_big_ij(
        2, 0
    )
    R[idx_20, idx_20] = term  # ρ₂₀ ← ρ₂₀

    # --- d/dt rho_02 ---
    R[idx_02, idx_02] = np.conj(term)  # ρ₀₂ ← ρ₀₂

    # --------------------------------------------------------------
    # 2) double-excited-coherences
    # --------------------------------------------------------------
    # --- d/dt rho_30 ---
    term = -1j * (system.omega_ij(3, 0) - 2 * system.omega_laser) - system.Gamma_big_ij(
        3, 0
    )
    R[idx_30, idx_30] = term  # ρ₃₀ ← ρ₃₀

    # --- d/dt rho_03 ---
    R[idx_03, idx_03] = np.conj(term)  # ρ₀₃ ← ρ₀₃

    # --------------------------------------------------------------
    # 3) cross-coherences inside one excitation manifold
    # --------------------------------------------------------------
    # --- d/dt rho_12 ---
    term = -1j * system.omega_ij(1, 2) - system.Gamma_big_ij(1, 2)
    R[idx_12, idx_12] = term  # ρ₁₂ ← ρ₁₂

    # --- d/dt rho_21 ---
    R[idx_21, idx_21] = np.conj(term)  # ρ₂₁ ← ρ₂₁

    # --- d/dt rho_31 ---
    term = -1j * (system.omega_ij(3, 1) - system.omega_laser) - system.Gamma_big_ij(
        3, 1
    )
    R[idx_31, idx_31] = term  # ρ₃₁ ← ρ₃₁

    # --- d/dt rho_13 ---
    R[idx_13, idx_13] = np.conj(term)  # ρ₁₃ ← ρ₁₃

    # --- d/dt rho_32 ---
    term = -1j * (system.omega_ij(3, 2) - system.omega_laser) - system.Gamma_big_ij(
        3, 2
    )
    R[idx_32, idx_32] = term  # ρ₃₂ ← ρ₃₂

    # --- d/dt rho_23 ---
    R[idx_23, idx_23] = np.conj(term)  # ρ₂₃ ← ρ₂₃

    # --------------------------------------------------------------
    # 4) populations (diagonals)
    # --------------------------------------------------------------
    # --- d/dt rho_11 ---
    R[idx_11, idx_11] = -system.Gamma_big_ij(1, 1)
    R[idx_11, idx_22] = system.gamma_small_ij(
        1, 2
    )  # for the coupled dimer: pop transer

    # --- d/dt rho_22 ---
    R[idx_22, idx_22] = -system.Gamma_big_ij(2, 2)
    R[idx_22, idx_11] = system.gamma_small_ij(
        2, 1
    )  # for the coupled dimer: pop transer

    # NOW THERE IS NO POPULATION CHANGE in 3 || 1 goes to 2 and vice versa
    # --- d/dt rho_00 --- and  --- d/dt rho_33 (sum d/dt rho_ii = 0) (trace condition) ---
    R[idx_33, :] = -R[idx_00, :] - R[idx_11, :] - R[idx_22, :]

    return Qobj(R, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]])

def main():
    pass

if __name__ == "__main__":
    main()