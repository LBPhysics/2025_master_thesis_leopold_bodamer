# =============================
# FUNCTIONS for overlapping pulses
# =============================

from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.pyplot as plt
from qutip.solver import Result
from qutip import *
import numpy as np
import itertools
import pickle
import copy

# Import the outsourced settings / functions
from plot_settings import *
from functions_for_both_cases import *
import matplotlib as mpl

mpl.use("TkAgg")  # open each plot in interactive window


# =============================
# SYSTEM PARAMETERS     (**changeable**)
# =============================

### Phase Cycling for Averaging
phases = [k * np.pi / 2 for k in range(4)]


# ##########################
# dependent of system
# ##########################
def apply_RWA_phase_factors(rho, t, omega):
    """
    Apply time-dependent phase factors to the density matrix entries.

    Parameters:
        rho (Qobj): Density matrix (Qobj) to modify.
        omega (float): Frequency of the phase factor.
        t (float): Current time.

    Returns:
        Qobj: Modified density matrix with phase factors applied.
    """
    # Extract the density matrix as a NumPy array
    rho_array = rho.full()
    # print(rho.isherm)

    # Apply the phase factors to the specified elements
    phase_1 = np.exp(-1j * omega * t)  # e^(-i * omega * t)

    # Modify the elements
    rho_array[1, 0] *= phase_1  # rho_alpha_0 = sigma_alpha_0 * e^(-i * omega * t)
    rho_array[0, 1] *= np.conj(phase_1)
    rho_result = Qobj(rho_array, dims=rho.dims)
    # print(rho_array[0, 1], rho_array[1,0])

    # assert rho_result.isherm, "The resulting density matrix is not Hermitian."

    return rho_result


def Hamilton_tls(system: SystemParameters) -> Qobj:
    """
    Hamiltonian of a two-level system.

    Parameters:
        system (SystemParameters): System parameters containing hbar, omega_A, omega_laser, atom_e, and RWA_laser.

    Returns:
        Qobj: Hamiltonian operator of the two-level system.
    """
    # =============================
    # Build Hamiltonian in energy basis
    # =============================
    H0 = system.hbar * system.omega_A * ket2dm(system.atom_e)
    if system.RWA_laser:
        H0 -= (
            system.hbar * system.omega_laser * ket2dm(system.atom_e)
        )  # shift in rotating frame
    return H0


# =============================
# "Paper_eqs" OWN ODE SOLVER
# =============================
def matrix_ODE_paper(
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
    L[3, 1] = 1j * Et * system.mu_eg_cm
    L[3, 2] = -1j * Et_conj * system.mu_eg_cm

    # --- d/dt rho_gg ---
    # L[0, 1] = -1j * Et * system.mu_eg_cm
    # L[0, 2] = 1j * Et_conj * system.mu_eg_cm
    L[0, :] += -1 * np.sum(L[[3], :], axis=0)  # Enforce trace conservation

    # --- d/dt rho_eg --- and  --- d/dt rho_ge ---
    L[2, 0] = 1j * Et * system.mu_eg_cm
    L[2, 3] = -1j * Et * system.mu_eg_cm

    L[1, :] = np.conj(L[2, :])

    L[2, 2] = -system.Gamma  # Decay term for coherence
    L[1, 1] = -system.Gamma  # Decay term for coherence

    return Qobj(L, dims=[[[2], [2]], [[2], [2]]])


def R_paper(system: SystemParameters) -> Qobj:
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


def compute_pulse_evolution(
    psi_ini: Qobj,
    times: np.ndarray,
    pulse_seq: PulseSequence,
    system: SystemParameters = None,
) -> Result:
    """
    Compute the evolution of the system for a given pulse sequence.

    Parameters:
        psi_ini (Qobj): Initial quantum state.
        times (np.ndarray): Time array for the evolution.
        pulse_seq (PulseSequence): PulseSequence object.
        system (SystemParameters): System parameters.

    Returns:
        qutip.Result: Result of the evolution.
    """
    # =============================
    # Use global parameters if not provided
    # =============================
    if system is None:
        raise ValueError("System parameters must be provided.")

    # Set solver options
    # progress_bar = "enhanced" if preprev != None and times[0] >= times[len(times) // 2] else ""
    options = {
        "store_states": True,
        "progress_bar": "",  # progress_bar,
        #   "nsteps": 10000,  # Increase max number of steps per integration interval
    }

    # =============================
    # Choose solver and compute the evolution
    # =============================
    if system.ODE_Solver not in ["Paper_eqs", "ME", "Paper_BR"]:
        raise ValueError(f"Unknown ODE solver: {system.ODE_Solver}")

    if system.ODE_Solver == "Paper_eqs":
        # You need to adapt Liouville to accept pulse_seq and system if needed
        Liouville = QobjEvo(lambda t, args=None: matrix_ODE_paper(t, pulse_seq, system))
        result = mesolve(
            Liouville,
            psi_ini,
            times,
            options=options,
        )
    else:
        # Build Hamiltonian
        H_free = Hamilton_tls(system)
        H_int_evo = H_free + QobjEvo(lambda t, args=None: H_int(t, pulse_seq, system))
        c_ops = []
        if system.ODE_Solver == "Paper_BR":
            c_ops = [R_paper(system)]
        elif system.ODE_Solver == "ME":
            c_ops = system.c_ops_list

        result = mesolve(
            H_int_evo,
            psi_ini,
            times,
            c_ops=c_ops,
            options=options,
        )

        """
        # =============================
        # Split the evolution into two parts:
        # 1. With H_int for the pulse duration
        # 2. With H0 for the rest
        # =============================
        # Find the index where times = times[0] + 2*Delta_ts[0]
        last_pulse = pulse_seq.pulses[0]# MIGHT be wrong? -> -1
        t_pulse_end = last_pulse.pulse_start_time + 2 * last_pulse.pulse_half_width
        idx_split   = np.abs(times - t_pulse_end).argmin()

        times1 = times[:idx_split]
        if times1.size == 0:
            times1 = [last_pulse.pulse_start_time]
            

        result1 = mesolve(
            H_int_evo,
            psi_ini,
            times1,
            c_ops=[R_paper(system)],
            options=options,
        )

        # --- Second part: with H0 only ---
        # Use the last state as initial state for the second part
        psi_after_pulse = result1.states[-1]
        times2 = times[idx_split:]
        if len(times2) > 0:
            result2 = mesolve(
                H_free,
                psi_after_pulse,
                times2,
                c_ops=c_ops,
                options=options,
            )
            # Combine results
            all_states = list(result1.states) + list(result2.states)
            all_times = list(result1.times) + list(result2.times)
            options_full = options.copy()
            if "store_final_state" not in options_full:
                options_full["store_final_state"] = False
            if "store_states" not in options_full:
                options_full["store_states"] = True
            result = Result(e_ops=[], options=options_full)
            result.states = all_states
            result.times = all_times
            # Copy other attributes if needed
        else:
            result = result1"""

    return result


def get_expect_vals_with_RWA(
    states: list[qutip.Qobj], times: np.array, system: SystemParameters
):
    """
    Calculate the expectation values in the result with RWA phase factors.

    Parameters:
        states= data.states (where data = qutip.Result): Results of the pulse evolution.
        times (list): Time points at which the expectation values are calculated.
        e_ops (list): the operators for which the expectation values are calculated
        omega (float): omega_laser (float): Frequency of the laser.
        RWA (bool): Whether to apply the RWA phase factors.
    Returns:
        list of lists: Expectation values for each operator of len(states).
    """
    omega = system.omega_laser
    e_ops = system.e_ops_list + [system.Dip_op]

    if system.RWA_laser:
        # Apply RWA phase factors to each state
        states = [
            apply_RWA_phase_factors(state, time, omega)
            for state, time in zip(states, times)
        ]
    updated_expects = [np.real(expect(states, e_op)) for e_op in e_ops]
    return updated_expects


# ##########################
# independent of system
# ##########################
def check_the_solver(times: np.ndarray, system: SystemParameters) -> qutip.Result:
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
    t_start_0 = times[0]
    t_start_1 = times[-1] / 2
    t_start_2 = times[-1] / 1.1

    # Use the from_args static method to construct the sequence
    pulse_seq = PulseSequence.from_args(
        system=system,
        curr=(t_start_2, phi_2),
        prev=(t_start_1, phi_1),
        preprev=(t_start_0, phi_0),
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
    global time_cut
    time_cut = np.inf
    for index, state in enumerate(result.states):
        # Apply RWA phase factors if needed
        if getattr(system, "RWA_laser", False):
            state = apply_RWA_phase_factors(
                state, times[index], omega=system.omega_laser
            )
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


def compute_two_dimensional_polarization(
    T_wait: float,
    phi_0: float,
    phi_1: float,
    times: np.ndarray,
    system: SystemParameters,
    plot_example: bool = False,
    **kwargs,
):
    """
    Compute the two-dimensional polarization for a given waiting time (T_wait) and
    the phases of the first and second pulses (phi_0, phi_1).

    Parameters:
        T_wait (float): Waiting time between the second and third pulses.
        phi_0 (float): Phase of the first pulse.
        phi_1 (float): Phase of the second pulse.
        times (np.ndarray): Time array.
        system: System object containing all relevant parameters.
        plot_example (bool, optional): Whether to plot an example evolution.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: (t_det_vals, tau_coh_vals, data)
    """

    # get the symmetric times, tau_coh, t_det
    tau_coh_vals, t_det_vals = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait=T_wait)

    # initialize the time domain Spectroscopy data tr(Dip_op * rho_final(tau_coh, t_det))
    data = np.zeros((len(tau_coh_vals), len(t_det_vals)), dtype=np.complex64)

    idx_start_0 = 0
    t_start_0 = times[idx_start_0]
    idx_end_0 = np.abs(times - (system.Delta_ts[0])).argmin()
    idx_start_1_max = np.abs(times - (tau_coh_vals[-1] - system.Delta_ts[1])).argmin()
    times_0 = times[: idx_start_1_max + 1]
    if times_0.size == 0:
        times_0 = times[idx_start_0 : idx_end_0 + 1]

    # First pulse
    pulse_0 = (t_start_0, phi_0)
    # Instead of directly constructing PulseSequence, use from_args:
    pulse_seq_0 = PulseSequence.from_args(
        system=system,
        curr=pulse_0,
    )
    data_0 = compute_pulse_evolution(
        system.psi_ini, times_0, pulse_seq_0, system=system
    )

    for tau_idx, tau_coh in enumerate(tau_coh_vals):
        idx_start_1 = np.abs(times - (tau_coh - system.Delta_ts[1])).argmin()
        t_start_1 = times[idx_start_1]
        idx_end_1 = np.abs(times - (tau_coh + system.Delta_ts[1])).argmin()
        rho_1 = data_0.states[idx_start_1]

        idx_start_2 = np.abs(times - (tau_coh + T_wait - system.Delta_ts[2])).argmin()
        idx_end_2 = np.abs(times - (tau_coh + T_wait + system.Delta_ts[2])).argmin()
        t_start_2 = times[idx_start_2]

        times_1 = times[idx_start_1 : idx_start_2 + 1]
        if times_1.size == 0:
            times_1 = times[idx_start_1 : idx_end_1 + 1]

        pulse_1 = (t_start_1, phi_1)
        pulse_seq_1 = PulseSequence.from_args(
            system=system,
            curr=pulse_1,
            prev=pulse_0,
        )
        data_1 = compute_pulse_evolution(rho_1, times_1, pulse_seq_1, system=system)

        idx_start_2_in_times_1 = np.abs(times_1 - t_start_2).argmin()
        rho_2 = data_1.states[idx_start_2_in_times_1]

        times_2 = times[idx_start_2:]
        if times_2.size == 0:
            times_2 = times[idx_start_2 : idx_end_2 + 1]

        phi_2 = 0
        pulse_f = (t_start_2, phi_2)
        pulse_seq_f = PulseSequence.from_args(
            system=system,
            curr=pulse_f,
            prev=pulse_1,
            preprev=pulse_0,
        )
        data_f = compute_pulse_evolution(rho_2, times_2, pulse_seq_f, system=system)

        for t_idx, t_det in enumerate(t_det_vals):
            actual_det_time = t_start_2 + system.Delta_ts[2] + t_det
            if actual_det_time < system.t_max:
                t_idx_in_times_2 = np.abs(times_2 - actual_det_time).argmin()

                if actual_det_time < time_cut:
                    rho_f = data_f.states[t_idx_in_times_2]
                    if system.RWA_laser:
                        rho_f = apply_RWA_phase_factors(
                            rho_f, times_2[t_idx_in_times_2], omega=system.omega_laser
                        )
                    value = expect(system.Dip_op, rho_f)
                    data[tau_idx, t_idx] = np.real(value)

                    if (
                        t_idx == 0
                        and tau_idx == len(tau_coh_vals) // 3
                        and plot_example
                    ):
                        print(system.RWA_laser)
                        data_1_expects = get_expect_vals_with_RWA(
                            data_0.states[: idx_start_1 + 1],
                            data_0.times[: idx_start_1 + 1],
                            system,
                        )
                        data_2_expects = get_expect_vals_with_RWA(
                            data_1.states[: idx_start_2_in_times_1 + 1],
                            data_1.times[: idx_start_2_in_times_1 + 1],
                            system,
                        )
                        data_f_expects = get_expect_vals_with_RWA(
                            data_f.states, data_f.times, system
                        )
                        data_expectations = [
                            np.concatenate(
                                [
                                    data_1_expects[idx],
                                    data_2_expects[idx],
                                    data_f_expects[idx],
                                ]
                            )
                            for idx in range(len(system.e_ops_list) + 1)
                        ]

                        Plot_example_evo(
                            times_0[: idx_start_1 + 1],
                            times_1,
                            times_2,
                            data_expectations,
                            pulse_seq_f,
                            tau_coh,
                            T_wait,
                            system=system,
                        )

    return t_det_vals, tau_coh_vals, 1j * data  # because E ~ i*P


def main():
    """
    Main function to run the script.
    """
    # =============================
    # SYSTEM PARAMETERS
    # =============================
    system = SystemParameters(
        N_atoms=1,
        ODE_Solver="Paper_eqs",
        RWA_laser=True,
        Delta_cm=200.0,
        omega_A_cm=16000.0,
        mu_eg_cm=1.0,
        omega_laser_cm=16000.0,
        E0=0.1,
        pulse_duration=15.0,
        t_max=500.0,  # -> determines Δω ∝ 1/t_max
        fine_spacing=0.1,  # -> determines ω_max ∝ 1/Δt
        gamma_0=1 / 300,
        T2=100.0,
    )

    print(system.summary())

    t_max = system.t_max
    fine_spacing_test = system.fine_spacing

    Delta_ts = system.Delta_ts
    times = np.arange(-Delta_ts[0], t_max, fine_spacing_test)
    print("times: ", times[0], times[1], "...", times[-1], "len", len(times))

    # =============================
    test_params_copy = copy.deepcopy(system)
    if "time_cut" not in globals() or test_params_copy.t_max != system.t_max:
        # =============================
        # ALWAYS CHECK Before running a serious simulation
        # =============================
        test_params_copy.t_max = 10 * t_max
        test_params_copy.fine_spacing = 10 * fine_spacing_test
        times_test_ = np.arange(
            -Delta_ts[0], test_params_copy.t_max, test_params_copy.fine_spacing
        )
        result, time_cut = check_the_solver(times_test_, test_params_copy)
        print("the evolution is actually unphisical after:", time_cut, "fs")

    T_wait = times[-1] / 10
    two_d_data = compute_two_dimensional_polarization(
        T_wait,
        phases[0],
        phases[0],
        times=times,
        system=system,
        plot_example=True,
    )
    # =============================
    # Save the result to a file for later use
    # =============================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "two_d_data.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(two_d_data, f)
    print(f"2D data saved to {save_path}")

    # with open(save_path, "rb") as f:
    #     two_d_data = pickle.load(f)

    plot_args_freq = dict(  # (**changeable**)
        space="freq",
        type="real",  # plot the real part (also "imag", "phase", "abs")
        safe=False,  # (dont) save the spectrum
        positive=True,  # only plot the positive spectrum
        use_custom_colormap=True,  # all zeros are white
        section=(  # focus on the non zero part
            1.4,
            1.8,  # xmin, xmax,
            1.4,
            1.8,  # ymin, ymax
        ),
        # add more options as needed
    )

    ts, taus, data = two_d_data[0], two_d_data[1], two_d_data[2]

    extend_for = (0, 100)
    ts, taus, data = extend_time_tau_axes(
        ts, taus, data, pad_rows=extend_for, pad_cols=extend_for
    )
    plot_positive_color_map(
        (ts, taus, data),
        type="imag",  # because E ~ i*P
        T_wait=T_wait,
        safe=False,
        use_custom_colormap=True,
    )

    nu_ts, nu_taus, s2d = compute_2d_fft_wavenumber(ts, taus, data)
    plot_positive_color_map((nu_ts, nu_taus, s2d), T_wait=T_wait, **plot_args_freq)


if __name__ == "__main__":
    main()
