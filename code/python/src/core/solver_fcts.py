import numpy as np
from qutip import *
from src.core.system_parameters import SystemParameters
from src.core.pulse_sequences import PulseSequence
from src.core.pulse_functions import *
from src.spectroscopy.calculations import compute_pulse_evolution


# =============================
# "Paper_eqs" OWN ODE SOLVER
# =============================
def matrix_ODE_paper(
    t: float, pulse_seq: PulseSequence, system: SystemParameters
) -> Qobj:
    """Dispatches to the appropriate implementation based on N_atoms."""
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
    Constructs the matrix L(t) for the equation drho_dt = L(t) * rho,
    where rho is the flattened density matrix. Uses gamma values from the provided system.

    Parameters:
        t (float): Time at which to evaluate the matrix.
        pulse_seq (PulseSequence): PulseSequence object for the electric field.
        system (SystemParameters): System parameters containing Gamma, gamma_0, and mu_eg.

    Returns:
        Qobj: Liouvillian matrix as a Qobj.
    """
    # Calculate the electric field using the pulse sequence
    Et = E_pulse(t, pulse_seq)
    Et_conj = np.conj(Et)

    L = np.zeros((4, 4), dtype=complex)

    # Indices for the flattened density matrix:
    # 0: rho_gg, 1: rho_ge, 2: rho_eg, 3: rho_ee

    # --- d/dt rho_ee ---
    L[3, 3] = -system.gamma_0
    L[3, 1] = 1j * Et * system.mu_A
    L[3, 2] = -1j * Et_conj * system.mu_A

    # --- d/dt rho_gg ---
    # L[0, 1] = -1j * Et * system.mu_A
    # L[0, 2] = 1j * Et_conj * system.mu_A
    L[0, :] += -1 * np.sum(L[[3], :], axis=0)  # Enforce trace conservation

    # --- d/dt rho_eg --- and  --- d/dt rho_ge ---
    L[2, 0] = 1j * Et * system.mu_A
    L[2, 3] = -1j * Et * system.mu_A

    L[1, :] = np.conj(L[2, :])

    L[2, 2] = -system.Gamma  # Decay term for coherence
    L[1, 1] = -system.Gamma  # Decay term for coherence

    return Qobj(L, dims=[[[2], [2]], [[2], [2]]])


def _matrix_ODE_paper_2atom(
    t: float, pulse_seq: PulseSequence, system: SystemParameters
) -> Qobj:
    """including RWA.
    Constructs the matrix L(t) for the equation drho_dt = L(t) * rho,
    where rho is the flattened density matrix.
    """
    Et = E_pulse(t, pulse_seq)
    Et_conj = np.conj(Et)

    L = np.zeros((16, 16), dtype=complex)

    # Indices for the flattened density matrix:
    # 0: rho00, 1: rho01, 2: rho02, 3: rho03
    # 4: rho10, 5: rho11, 6: rho12, 7: rho13
    # 8: rho20, 9: rho21, 10: rho22, 11: rho23
    # 12: rho30, 13: rho31, 14: rho32, 15: rho33

    # --- d/dt rho_10 ---
    term = -1j * (system.omega_ij(1, 0) - system.omega_laser) - system.Gamma_big_ij(
        1, 0
    )
    L[4, 4] = term  # ρ₁₀ ← ρ₁₀
    L[4, 0] = 1j * Et * system.Dip_op[1, 0]  # ρ₁₀ ← ρ₀₀
    L[4, 5] = -1j * Et * system.Dip_op[1, 0]  # ρ₁₀ ← ρ₁₁
    L[4, 6] = -1j * Et * system.Dip_op[2, 0]  # ρ₁₀ ← ρ₁₂
    L[4, 12] = 1j * Et_conj * system.Dip_op[3, 1]  # ρ₁₀ ← ρ₃₀

    # --- d/dt rho_01 ---
    L[1, 1] = np.conj(term)  # ρ₀₁ ← ρ₀₁
    L[1, 0] = -1j * Et_conj * system.Dip_op[1, 0]  # ρ₀₁ ← ρ₀₀
    L[1, 5] = 1j * Et_conj * system.Dip_op[1, 0]  # ρ₀₁ ← ρ₁₁
    L[1, 9] = 1j * Et_conj * system.Dip_op[2, 0]  # ρ₀₁ ← ρ₂₁
    L[1, 3] = -1j * Et * system.Dip_op[3, 1]  # ρ₀₁ ← ρ₀₃

    # --- d/dt rho_20 ---
    term = -1j * (system.omega_ij(2, 0) - system.omega_laser) - system.Gamma_big_ij(
        2, 0
    )
    L[8, 8] = term  # ρ₂₀ ← ρ₂₀
    L[8, 0] = 1j * Et * system.Dip_op[2, 0]  # ρ₂₀ ← ρ₀₀
    L[8, 10] = -1j * Et * system.Dip_op[2, 0]  # ρ₂₀ ← ρ₂₂
    L[8, 9] = -1j * Et * system.Dip_op[1, 0]  # ρ₂₀ ← ρ₂₁
    L[8, 12] = 1j * Et_conj * system.Dip_op[3, 2]  # ρ₂₀ ← ρ₃₀

    # --- d/dt rho_02 ---
    L[2, 2] = np.conj(term)  # ρ₀₂ ← ρ₀₂
    L[2, 0] = -1j * Et_conj * system.Dip_op[2, 0]  # ρ₀₂ ← ρ₀₀
    L[2, 10] = 1j * Et_conj * system.Dip_op[2, 0]  # ρ₀₂ ← ρ₂₂
    L[2, 6] = 1j * Et_conj * system.Dip_op[1, 0]  # ρ₀₂ ← ρ₁₂
    L[2, 3] = -1j * Et * system.Dip_op[3, 2]  # ρ₀₂ ← ρ₀₃

    # --- d/dt rho_30 ---
    term = -1j * (system.omega_ij(3, 0) - 2 * system.omega_laser) - system.Gamma_big_ij(
        3, 0
    )
    L[12, 12] = term  # ρ₃₀ ← ρ₃₀
    L[12, 4] = 1j * Et * system.Dip_op[3, 1]  # ρ₃₀ ← ρ₁₀
    L[12, 8] = 1j * Et * system.Dip_op[3, 2]  # ρ₃₀ ← ρ₂₀
    L[12, 13] = -1j * Et * system.Dip_op[1, 0]  # ρ₃₀ ← ρ₃₁
    L[12, 14] = -1j * Et * system.Dip_op[2, 0]  # ρ₃₀ ← ρ₃₂

    # --- d/dt rho_03 ---
    L[3, 3] = np.conj(term)  # ρ₀₃ ← ρ₀₃
    L[3, 1] = -1j * Et_conj * system.Dip_op[3, 1]  # ρ₀₃ ← ρ₀₁
    L[3, 2] = -1j * Et_conj * system.Dip_op[3, 2]  # ρ₀₃ ← ρ₀₂
    L[3, 7] = 1j * Et_conj * system.Dip_op[1, 0]  # ρ₀₃ ← ρ₁₃
    L[3, 11] = 1j * Et_conj * system.Dip_op[2, 0]  # ρ₀₃ ← ρ₂₃

    # --- d/dt rho_12 ---
    term = -1j * system.omega_ij(1, 2) - system.Gamma_big_ij(1, 2)
    L[6, 6] = term  # ρ₁₂ ← ρ₁₂
    L[6, 2] = 1j * Et * system.Dip_op[1, 0]  # ρ₁₂ ← ρ₀₂
    L[6, 7] = -1j * Et * system.Dip_op[3, 2]  # ρ₁₂ ← ρ₁₃
    L[6, 14] = 1j * Et_conj * system.Dip_op[3, 1]  # ρ₁₂ ← ρ₃₂
    L[6, 4] = -1j * Et_conj * system.Dip_op[2, 0]  # ρ₁₂ ← ρ₁₀

    # --- d/dt rho_21 ---
    L[9, 9] = np.conj(term)  # ρ₂₁ ← ρ₂₁
    L[9, 8] = -1j * Et_conj * system.Dip_op[1, 0]  # ρ₂₁ ← ρ₂₀
    L[9, 13] = 1j * Et_conj * system.Dip_op[3, 2]  # ρ₂₁ ← ρ₃₁
    L[9, 9] = -1j * Et * system.Dip_op[3, 1]  # ρ₂₁ ← ρ₂₃
    L[9, 1] = 1j * Et * system.Dip_op[2, 0]  # ρ₂₁ ← ρ₀₁

    # --- d/dt rho_31 ---
    term = -1j * (system.omega_ij(3, 1) - system.omega_laser) - system.Gamma_big_ij(
        3, 1
    )
    L[13, 13] = term  # ρ₃₁ ← ρ₃₁
    L[13, 5] = 1j * Et * system.Dip_op[3, 1]  # ρ₃₁ ← ρ₁₁
    L[13, 9] = 1j * Et * system.Dip_op[3, 2]  # ρ₃₁ ← ρ₂₁
    L[13, 12] = -1j * Et_conj * system.Dip_op[1, 0]  # ρ₃₁ ← ρ₃₀

    # --- d/dt rho_13 ---
    L[7, 7] = np.conj(term)  # ρ₁₃ ← ρ₁₃
    L[7, 5] = -1j * Et_conj * system.Dip_op[3, 1]  # ρ₁₃ ← ρ₁₁
    L[7, 6] = -1j * Et_conj * system.Dip_op[3, 2]  # ρ₁₃ ← ρ₁₂
    L[7, 3] = 1j * Et * system.Dip_op[1, 0]  # ρ₁₃ ← ρ₀₃

    # --- d/dt rho_32 ---
    term = -1j * (system.omega_ij(3, 2) - system.omega_laser) - system.Gamma_big_ij(
        3, 2
    )
    L[14, 14] = term  # ρ₃₂ ← ρ₃₂
    L[14, 10] = 1j * Et * system.Dip_op[3, 2]  # ρ₃₂ ← ρ₂₂
    L[14, 6] = 1j * Et * system.Dip_op[3, 1]  # ρ₃₂ ← ρ₁₂
    L[14, 12] = -1j * Et_conj * system.Dip_op[2, 0]  # ρ₃₂ ← ρ₃₀

    # --- d/dt rho_23 ---
    L[11, 11] = np.conj(term)  # ρ₂₃ ← ρ₂₃
    L[11, 10] = -1j * Et * system.Dip_op[3, 2]  # ρ₂₃ ← ρ₂₂
    L[11, 9] = -1j * Et * system.Dip_op[3, 1]  # ρ₂₃ ← ρ₂₁
    L[11, 3] = 1j * Et_conj * system.Dip_op[2, 0]  # ρ₂₃ ← ρ₀₃

    ### Diagonals
    # --- d/dt rho_00 ---
    L[0, 1] = -1j * Et * system.Dip_op[1, 0]
    L[0, 2] = -1j * Et * system.Dip_op[2, 0]
    L[0, 4] = 1j * Et_conj * system.Dip_op[1, 0]
    L[0, 8] = 1j * Et_conj * system.Dip_op[2, 0]

    # --- d/dt rho_11 ---
    L[5, 5] = -1 * system.Gamma_big_ij(1, 1)
    L[5, 10] = system.gamma_small_ij(1, 2)
    L[5, 1] = 1j * Et * system.Dip_op[1, 0]
    L[5, 7] = -1j * Et * system.Dip_op[3, 1]
    L[5, 13] = 1j * Et_conj * system.Dip_op[3, 1]
    L[5, 4] = -1j * Et_conj * system.Dip_op[1, 0]

    # --- d/dt rho_22 ---
    L[10, 10] = -1 * system.Gamma_big_ij(2, 2)
    L[10, 5] = system.gamma_small_ij(2, 1)
    L[10, 2] = 1j * Et * system.Dip_op[2, 0]
    L[10, 11] = -1j * Et * system.Dip_op[3, 2]
    L[10, 14] = 1j * Et_conj * system.Dip_op[3, 2]
    L[10, 8] = -1j * Et_conj * system.Dip_op[2, 0]

    # --- d/dt rho_00 --- and  --- d/dt rho_33 (sum d/dt rho_ii = 0) (trace condition) ---
    L[15, :] = -1 * np.sum(
        L[[0, 5, 10], :], axis=0
    )  # TODO not mentioned in paper, i will assume it to conserve the trace
    # print("the trace d/dt (rho_00 + rho_11 + rho_22 + rho_33) = ", np.sum(L[[0, 5, 10, 15], :]), "should be 0")

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
    R = np.zeros((16, 16), dtype=complex)

    # Indices for the flattened density matrix:
    # 0: rho00, 1: rho01, 2: rho02, 3: rho03
    # 4: rho10, 5: rho11, 6: rho12, 7: rho13
    # 8: rho20, 9: rho21, 10: rho22, 11: rho23
    # 12: rho30, 13: rho31, 14: rho32, 15: rho33

    # --- d/dt rho_10 ---
    term = -1j * (system.omega_ij(1, 0) - system.omega_laser) - system.Gamma_big_ij(
        1, 0
    )
    R[4, 4] = term

    # --- d/dt rho_01 ---
    R[1, 1] = np.conj(term)

    # --- d/dt rho_20 --- = ANSATZ = (d/dt s_20 - i omega_laser s_20) e^(-i omega_laser t)
    term = -1j * (system.omega_ij(2, 0) - system.omega_laser) - system.Gamma_big_ij(
        2, 0
    )
    R[8, 8] = term

    # --- d/dt rho_02 ---
    R[2, 2] = np.conj(term)

    # --- d/dt rho_30 ---
    term = -1j * (system.omega_ij(3, 0) - 2 * system.omega_laser) - system.Gamma_big_ij(
        3, 0
    )
    R[12, 12] = term

    # --- d/dt rho_03 ---
    R[3, 3] = np.conj(term)

    # --- d/dt rho_12 ---
    term = -1j * system.omega_ij(1, 2) - system.Gamma_big_ij(1, 2)
    R[6, 6] = term

    # --- d/dt rho_21 ---
    R[9, 9] = np.conj(term)

    # --- d/dt rho_31 ---
    term = -1j * (system.omega_ij(3, 1) - system.omega_laser) - system.Gamma_big_ij(
        3, 1
    )
    R[13, 13] = term

    # --- d/dt rho_13 ---
    R[7, 7] = np.conj(term)

    # --- d/dt rho_32 ---
    term = -1j * (system.omega_ij(3, 2) - system.omega_laser) - system.Gamma_big_ij(
        3, 2
    )
    R[14, 14] = term

    # --- d/dt rho_23 ---
    R[11, 11] = np.conj(term)

    ### Diagonals
    # --- d/dt rho_11 ---
    R[5, 5] = -system.Gamma_big_ij(1, 1)
    R[5, 10] = system.gamma_small_ij(1, 2)

    # --- d/dt rho_22 ---
    R[10, 10] = -system.Gamma_big_ij(2, 2)
    R[10, 5] = system.gamma_small_ij(2, 1)

    # NOW THERE IS NO POPULATION CHANGE in 3 || 1 goes to 2 and vice versa
    # --- d/dt rho_00 --- and  --- d/dt rho_33 (sum d/dt rho_ii = 0) (trace condition) ---
    # R[15, :] = -1 * np.sum(R[[0, 5, 10], :], axis=0)
    # R[0, :] = -1 * np.sum(R[[5, 10, 15], :], axis=0) # i think the ground state should get repopulated

    return Qobj(R, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]])


# ##########################
# independent of system
# ##########################
def check_the_solver(
    times: np.ndarray, system: SystemParameters
) -> tuple[qutip.Result, float]:
    """
    Checks the solver within the compute_pulse_evolution function
    with the provided psi_ini, times, and system.

    Parameters:
        times (np.ndarray): Time array for the evolution.
        system (System): System object containing all relevant parameters, including e_ops_list.
        PulseSequence (type): The PulseSequence class to construct pulse sequences.

    Returns:
        qutip.Result: The result object from compute_pulse_evolution.
    """
    print(f"Checking '{system.ODE_Solver}' solver ")

    # =============================
    # INPUT VALIDATION
    # =============================
    if not hasattr(system, "ODE_Solver"):
        raise AttributeError("system must have attribute 'ODE_Solver'")
    if not hasattr(system, "e_ops_list"):
        raise AttributeError("system must have attribute 'e_ops_list'")
    if not isinstance(system.psi_ini, qutip.Qobj):
        raise TypeError("psi_ini must be a qutip.Qobj")
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be a numpy.ndarray")
    if not isinstance(system.e_ops_list, list) or not all(
        isinstance(op, qutip.Qobj) for op in system.e_ops_list
    ):
        raise TypeError("system.e_ops_list must be a list of qutip.Qobj")
    if len(times) < 2:
        raise ValueError("times must have at least two elements")

    # =============================
    # CONSTRUCT PULSE SEQUENCE (refactored)
    # =============================

    # Define pulse parameters
    phi_0 = np.pi / 2
    phi_1 = np.pi / 4
    phi_2 = 0
    t_start_pulse0 = times[0]
    t_start_pulse1 = times[-1] / 2
    t_start_2 = times[-1] / 1.1

    # Use the from_args static method to construct the sequence
    pulse_seq = PulseSequence.from_args(
        system=system,
        curr=(t_start_2, phi_2),
        prev=(t_start_pulse1, phi_1),
        preprev=(t_start_pulse0, phi_0),
    )
    result = compute_pulse_evolution(system.psi_ini, times, pulse_seq, system=system)
    # =============================
    # CHECK THE RESULT
    # =============================
    if not isinstance(result, qutip.Result):
        raise TypeError("Result must be a qutip.Result object")
    if list(result.times) != list(times):
        raise ValueError("Result times do not match input times")
    if len(result.states) != len(times):
        raise ValueError("Number of output states does not match number of time points")

    # =============================
    # CHECK DENSITY MATRIX PROPERTIES
    # =============================
    strg = ""
    omega = system.omega_laser
    global time_cut
    time_cut = np.inf  # time after which the checks failed
    for index, state in enumerate(result.states):
        # Apply RWA phase factors if needed
        if getattr(system, "RWA_laser", False):
            state = apply_RWA_phase_factors(state, times[index], omega, system)
        time = times[index]
        if not state.isherm:
            strg += f"Density matrix is not Hermitian after t = {time}.\n"
            print(state)
        eigvals = state.eigenenergies()
        if not np.all(
            eigvals >= -1e-3
        ):  # allow for small numerical negative eigenvalues
            strg += f"Density matrix is not positive semidefinite after t = {time}: The lowest eigenvalue is {eigvals.min()}.\n"
            time_cut = time
        if not np.isclose(state.tr(), 1.0):
            strg += f"Density matrix is not trace-preserving after t = {time}: The trace is {state.tr()}.\n"
            time_cut = time
        if strg:
            strg += "Adjust your parameters!"
            print(strg)
            break
    else:
        print(
            "Checks passed. Solver appears to be called correctly, and density matrix remains Hermitian and positive."
        )

    return result, time_cut
