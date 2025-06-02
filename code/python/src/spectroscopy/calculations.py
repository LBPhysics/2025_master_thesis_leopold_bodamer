# -*- coding: utf-8 -*-

import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, Result, mesolve, brmesolve, expect
from qutip.core import QobjEvo
from src.core.system_parameters import SystemParameters
from src.core.pulse_functions import *
from src.core.pulse_sequences import PulseSequence
from src.core.solver_fcts import (
    matrix_ODE_paper,
    R_paper,
    apply_RWA_phase_factors,
)
from src.spectroscopy.inhomogenity import sample_from_sigma
from src.core.functions_with_rwa import H_int, get_expect_vals_with_RWA


def get_tau_cohs_and_t_dets_for_T_wait(
    times: np.ndarray, T_wait: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the time arrays for tau_coh and t_det based on the waiting time T_wait and the time grid.

    Parameters:
        times (np.ndarray): 1D array of time points (must be sorted and equally spaced).
        T_wait (float): Waiting time.

    Returns:
        tuple: Arrays for coherence and detection times (tau_coh, t_det).
               Both have the same length.
    """
    # =============================
    # Validate input
    # =============================
    if times.size == 0:
        raise ValueError("Input 'times' array must not be empty.")
    if times.size == 1:
        return np.array([0.0]), np.array([0.0])
    spacing = times[1] - times[0]
    t_max = times[-1]

    # =============================
    # Check T_wait validity
    # =============================
    if T_wait > t_max:
        print("Warning: T_wait >= t_max, no valid tau_coh/t_det values.")
        return np.array([]), np.array([])
    if np.isclose(T_wait, t_max):
        return np.array([0.0]), np.array([0.0])

    # =============================
    # Calculate tau_coh and t_det arrays
    # =============================
    tau_coh_max = t_max - T_wait
    if tau_coh_max < 0:
        return np.array([]), np.array([])

    tau_coh = np.arange(
        0, tau_coh_max + spacing / 2, spacing
    )  # include endpoint if possible
    # =============================
    # Ensure t_det does not exceed t_max due to floating point
    # =============================
    valid_idx = tau_coh <= t_max + 1e-10
    tau_coh = tau_coh[valid_idx]  # makes sure T_wait < 0 is okay

    return tau_coh, tau_coh


def compute_pulse_evolution(
    psi_ini: Qobj,
    times: np.ndarray,
    pulse_seq: PulseSequence,
    system: SystemParameters = None,
    **solver_options: dict,
) -> Result:
    """
    Compute the evolution of the system for a given pulse sequence.

    Parameters:
        psi_ini (Qobj): Initial quantum state.
        times (np.ndarray): Time array for the evolution.
        pulse_seq (PulseSequence): PulseSequence object.
        system (SystemParameters): System parameters.

    Returns:
        Result: Result of the evolution.
    """
    # =============================
    # Validate input parameters
    # =============================
    if system is None:
        raise ValueError("System parameters must be provided.")

    if not isinstance(psi_ini, Qobj):
        raise TypeError(f"Expected psi_ini to be a Qobj, got {type(psi_ini)}")

    if not isinstance(times, np.ndarray) or len(times) == 0:
        raise ValueError(f"Invalid times array: {times}")

    if not isinstance(pulse_seq, PulseSequence) or len(pulse_seq.pulses) == 0:
        raise ValueError("Invalid or empty pulse sequence")

    # =============================
    # Set solver options
    # =============================
    # TODO get from solver_options
    options = {
        "store_states": True,
        "progress_bar": "",
        # Increasing max steps and atol/rtol for better stability
        #        "nsteps": 20000,
        #        "atol": 1e-8,
        #        "rtol": 1e-6,
    }

    # =============================
    # Choose solver and compute the evolution
    # =============================
    if system.ODE_Solver not in ["ME", "BR", "Paper_eqs", "Paper_BR"]:
        raise ValueError(f"Unknown ODE solver: {system.ODE_Solver}")

    if system.ODE_Solver == "Paper_eqs":
        # Check for time discontinuities (another common source of errors)
        if len(times) > 1:
            dt = np.diff(times)
            if not np.allclose(dt, dt[0], rtol=1e-5):
                jumps = np.where(np.abs(dt - dt[0]) > 1e-5 * dt[0])[0]
                print(
                    f"WARNING: Time step discontinuities detected at indices: {jumps[:10]}..."
                )
                print(f"  Time steps: {dt[jumps[:5]]}")

        # Check pulse timing relative to time array
        """
        for i, pulse in enumerate(pulse_seq.pulses):
            start_idx = np.argmin(np.abs(times - pulse.pulse_peak_time))
            if abs(times[start_idx] - pulse.pulse_peak_time) > 1e-5:
                print(
                    f"WARNING: Pulse {i} start time {pulse.pulse_peak_time:.6f} "
                    f"doesn't align with time grid (closest: {times[start_idx]:.6f})"
                )
        """
        if not system.RWA_laser:
            raise ValueError("The equations of the paper only make sense with RWA")

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
        H_free = system.H0_diagonalized  # already includes the RWA, if present!
        if H_free is None or not isinstance(H_free, Qobj):
            raise ValueError(f"Invalid H0_diagonalized: {H_free}")

        H_int_evo = H_free + QobjEvo(lambda t, args=None: H_int(t, pulse_seq, system))

        c_ops = []

        if system.ODE_Solver == "Paper_BR":
            c_ops = [R_paper(system)]
            if not all(isinstance(op, Qobj) for op in c_ops):
                raise ValueError(f"Invalid Redfield tensor: {c_ops}")

            result = mesolve(
                H_int_evo,
                psi_ini,
                times,
                c_ops=c_ops,
                options=options,
            )

        elif system.ODE_Solver == "ME":
            c_ops = system.c_ops_list
            if not all(isinstance(op, Qobj) for op in c_ops):
                raise ValueError(f"Invalid collapse operators: {c_ops}")

            result = mesolve(
                H_int_evo,
                psi_ini,
                times,
                c_ops=c_ops,
                options=options,
            )

        elif system.ODE_Solver == "BR":
            if not hasattr(system, "a_ops_list") or system.a_ops_list is None:
                raise ValueError("Missing a_ops_list for BR solver")

            result = brmesolve(
                H_int_evo,
                psi_ini,
                times,
                a_ops=system.a_ops_list,
                options=options,
            )
    return result


def check_the_solver(
    times: np.ndarray, system: SystemParameters
) -> tuple[Result, float]:
    """
    Checks the solver within the compute_pulse_evolution function
    with the provided psi_ini, times, and system.

    Parameters:
        times (np.ndarray): Time array for the evolution.
        system (System): System object containing all relevant parameters, including e_ops_list.
        PulseSequence (type): The PulseSequence class to construct pulse sequences.

    Returns:
        Result: The result object from compute_pulse_evolution.
    """
    print(f"Checking '{system.ODE_Solver}' solver ")

    # =============================
    # INPUT VALIDATION
    # =============================
    if not hasattr(system, "ODE_Solver"):
        raise AttributeError("system must have attribute 'ODE_Solver'")
    if not hasattr(system, "e_ops_list"):
        raise AttributeError("system must have attribute 'e_ops_list'")
    if not isinstance(system.psi_ini, Qobj):
        raise TypeError("psi_ini must be a Qobj")
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be a numpy.ndarray")
    if not isinstance(system.e_ops_list, list) or not all(
        isinstance(op, Qobj) for op in system.e_ops_list
    ):
        raise TypeError("system.e_ops_list must be a list of Qobj")
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
    t_start_pulse2 = times[-1] / 1.1

    # Use the from_args static method to construct the sequence
    pulse_seq = PulseSequence.from_args(
        system=system,
        curr=(t_start_pulse2, phi_2),
        prev=(t_start_pulse1, phi_1),
        preprev=(t_start_pulse0, phi_0),
    )

    result = compute_pulse_evolution(system.psi_ini, times, pulse_seq, system=system)
    # =============================
    # CHECK THE RESULT
    # =============================
    if not isinstance(result, Result):
        raise TypeError("Result must be a Result object")
    if list(result.times) != list(times):
        raise ValueError("Result times do not match input times")
    if len(result.states) != len(times):
        raise ValueError("Number of output states does not match number of time points")

    # =============================
    # CHECK DENSITY MATRIX PROPERTIES
    # =============================
    strg = ""
    omega = system.omega_laser
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


# cal one dimensional polarization data for fixed tau_coh and T_wait
def compute_fixed_tau_T(
    tau_coh: float,
    T_wait: float,
    phi_0: float,
    phi_1: float,
    times: np.ndarray,
    system: SystemParameters,
    **kwargs,
):
    """
    Compute the data for a fixed tau_coh and T_wait.

    Parameters
    ----------
    tau_coh : float
        Coherence time.
    T_wait : float
        Waiting time.
    phi_0 : float
        Phase of the first pulse.
    phi_1 : float
        Phase of the second pulse.
    times : np.ndarray
        Time array for the simulation.
    system : SystemParameters
        System parameters object.

    Returns
    -------
    tuple
        (t_det_vals, data) where t_det_vals are the detection times (shifted to start at zero)
        and data is the corresponding computed observable.
    """
    # plot_example = kwargs.get("plot_example", False)

    t_peak_pulse0 = 0
    idx_start_pulse1 = np.abs(times - (tau_coh - system.FWHMs[1])).argmin()

    t_start_1 = times[idx_start_pulse1]  # Start time of the second pulse

    times_0 = times[
        : idx_start_pulse1 + 1
    ]  # definetly not empty except for when T_wait >= t_max
    if times_0.size == 0:
        times_0 = times[:2]  # idx_end_pulse0 + 1

    # calculate the evolution of the first pulse in the desired range for tau_coh

    # First pulse
    pulse_0 = (t_peak_pulse0, phi_0)
    # Instead of directly constructing PulseSequence, use from_args:
    pulse_seq_0 = PulseSequence.from_args(
        system=system,
        curr=pulse_0,
    )
    data_0 = compute_pulse_evolution(
        system.psi_ini, times_0, pulse_seq_0, system=system
    )

    rho_1 = data_0.states[idx_start_pulse1]

    # select range  ->  to reduce computation time
    idx_start_pulse2 = np.abs(times - (tau_coh + T_wait - system.FWHMs[2])).argmin()
    t_start_pulse2 = times[idx_start_pulse2]  # the time at which the third pulse starts

    times_1 = times[
        idx_start_pulse1 : idx_start_pulse2 + 1
    ]  # like this: also take the overlap into account;

    if times_1.size == 0:
        times_1 = [times[idx_start_pulse1]]

    # Handle overlapping pulses: If the second pulse starts before the first pulse ends, combine their contributions
    pulse_1 = (t_start_1, phi_1)
    pulse_seq_1 = PulseSequence.from_args(
        system=system,
        curr=pulse_1,
        prev=pulse_0,
    )
    data_1 = compute_pulse_evolution(rho_1, times_1, pulse_seq_1, system=system)

    idx_start_pulse2_in_times_1 = np.abs(times_1 - (t_start_pulse2)).argmin()

    rho_2 = data_1.states[
        idx_start_pulse2_in_times_1
    ]  # == state where the third pulse starts

    times_2 = times[idx_start_pulse2:]

    if times_2.size == 0:
        times_2 = [times[-1]]  # idx_start_pulse2 : idx_end_pulse2 + 1
    # If the second pulse starts before the first pulse ends, combine their contributions
    phi_2 = 0  # FIXED PHASE!
    pulse_f = (t_start_pulse2, phi_2)
    pulse_seq_f = PulseSequence.from_args(
        system=system,
        curr=pulse_f,
        prev=pulse_1,
        preprev=pulse_0,
    )
    data_f = compute_pulse_evolution(rho_2, times_2, pulse_seq_f, system=system)

    # Just in case I want to plot an example evolution
    plot_example = kwargs.get("plot_example", False)
    if plot_example:
        data_1_expects = get_expect_vals_with_RWA(
            data_0.states[: idx_start_pulse1 + 1],
            data_0.times[: idx_start_pulse1 + 1],
            system,
        )
        data_2_expects = get_expect_vals_with_RWA(
            data_1.states[: idx_start_pulse2_in_times_1 + 1],
            data_1.times[: idx_start_pulse2_in_times_1 + 1],
            system,
        )
        data_f_expects = get_expect_vals_with_RWA(data_f.states, data_f.times, system)
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
        times_plot = np.concatenate([times_0[: idx_start_pulse1 + 1], times_1, times_2])
        additional_info = {
            "phases": (phi_0, phi_1, phi_2),
            "tau_coh": tau_coh,
            "T_wait": T_wait,
            "system": system,
        }
        # Return the data to create a plot with Plot_example_evo(res)!!
        return (
            times_plot,
            data_expectations,
            pulse_seq_f,
            additional_info,
        )

    t_peak_pulse2 = tau_coh + T_wait
    t_det_start_idx_in_times_2 = np.abs(
        times_2 - (t_peak_pulse2)
    ).argmin()  # detection time index in times_2

    # only if we are still in the physical regime
    states = data_f.states[t_det_start_idx_in_times_2:]
    actual_det_times = data_f.times[t_det_start_idx_in_times_2:]
    data = np.zeros((len(actual_det_times)), dtype=np.complex64)

    # print(t_det_vals[0], t_det_vals[1], t_det_vals[-1], len(t_det_vals))

    if system.RWA_laser:
        states = [
            apply_RWA_phase_factors(
                state, time, omega=system.omega_laser, system=system
            )
            for state, time in zip(states, actual_det_times)
        ]

    for t_idx, t_det in enumerate(actual_det_times):
        # only if we are still in the physical regime
        time_cut = kwargs.get("time_cut", np.inf)
        if t_det < time_cut:
            # data[t_idx] = np.real(expect(system.Dip_op, states[t_idx]))
            data[t_idx] = system.Dip_op[1, 0] * states[t_idx][0, 1]

    return np.array(actual_det_times) - actual_det_times[0], data


# 2D polarization calculation
def compute_two_dimensional_polarization(
    T_wait: float,  # TODO somehow assert that this is 0 < T_wait < system.t_max
    phi_0: float,
    phi_1: float,
    times: np.ndarray,  # TODO ACTURALLY a list -> qutip needs a list ?
    system: SystemParameters,
    **kwargs: dict,
) -> tuple[np.array, np.array, np.ndarray]:  # (time axis` + 2d polarization)
    """
    Compute the two-dimensional polarization for a given waiting time (T_wait) and
    the phases of the first and second pulses (phi_0, phi_1).

    Parameters:
        T_wait (float): Waiting time between the second and third pulses.
        phi_0 (float): Phase of the first pulse.
        phi_1 (float): Phase of the second pulse.
        times (np.ndarray): Time array.
        system: System object containing all relevant parameters.
        **kwargs: Additional keyword arguments.
                  Can include 'plot_example' (bool, optional): Whether to plot an example evolution.

    Returns:
        tuple: (t_det_vals, tau_coh_vals, data)
    """

    # get the symmetric times, tau_coh, t_det
    tau_coh_vals, t_det_vals = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait=T_wait)

    # initialize the time domain Spectroscopy data tr(Dip_op * rho_final(tau_coh, t_det))
    data = np.zeros((len(tau_coh_vals), len(t_det_vals)), dtype=np.complex64)

    # information about the first pulse
    t_peak_pulse0 = 0
    pulse_0 = (t_peak_pulse0, phi_0)
    pulse_seq_0 = PulseSequence.from_args(
        system=system,
        curr=pulse_0,
    )

    idx_pulse1_max_peak = np.abs(
        times - (tau_coh_vals[-1])
    ).argmin()  # the last possible coherence time

    times_0 = times[: idx_pulse1_max_peak + 1]  # empty for T_wait >= t_max
    if times_0.size == 0:  # in case T_wait == t_max
        times_0 = times[:2]

    data_0 = compute_pulse_evolution(
        system.psi_ini, times_0, pulse_seq_0, system=system
    )

    for tau_idx, tau_coh in enumerate(tau_coh_vals):
        idx_start_pulse1 = np.abs(
            times - (tau_coh - system.FWHMs[1])
        ).argmin()  # from which point to start the next pulse
        rho_1 = data_0.states[idx_start_pulse1]

        idx_start_pulse2 = np.abs(times - (tau_coh + T_wait - system.FWHMs[2])).argmin()
        t_peak_pulse2 = tau_coh + T_wait

        times_1 = times[idx_start_pulse1 : idx_start_pulse2 + 1]
        if times_1.size == 0:  # The case if T_wait is 0
            times_1 = [times[idx_start_pulse1]]

        t_peak_pulse1 = tau_coh
        pulse_1 = (t_peak_pulse1, phi_1)
        pulse_seq_1 = PulseSequence.from_args(
            system=system,
            curr=pulse_1,
            prev=pulse_0,
        )
        data_1 = compute_pulse_evolution(rho_1, times_1, pulse_seq_1, system=system)

        idx_start_pulse2_in_times_1 = idx_start_pulse2 - idx_start_pulse1
        rho_2 = data_1.states[idx_start_pulse2_in_times_1]

        times_2 = times[idx_start_pulse2:]
        if times_2.size == 0:
            times_2 = [times[-1]]

        phi_2 = 0
        pulse_f = (t_peak_pulse2, phi_2)
        pulse_seq_f = PulseSequence.from_args(
            system=system,
            curr=pulse_f,
            prev=pulse_1,
            preprev=pulse_0,
        )
        data_f = compute_pulse_evolution(rho_2, times_2, pulse_seq_f, system=system)

        # Just in case I want to plot an example evolution
        plot_example = kwargs.get("plot_example", False)
        if plot_example:
            tau_example = kwargs.get(
                "tau_example", tau_coh_vals[len(tau_coh_vals) // 3]
            )
            if tau_coh == tau_example:
                data_1_expects = get_expect_vals_with_RWA(
                    data_0.states[: idx_start_pulse1 + 1],
                    data_0.times[: idx_start_pulse1 + 1],
                    system,
                )
                data_2_expects = get_expect_vals_with_RWA(
                    data_1.states[: idx_start_pulse2_in_times_1 + 1],
                    data_1.times[: idx_start_pulse2_in_times_1 + 1],
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
                times_plot = np.concatenate(
                    [times_0[: idx_start_pulse1 + 1], times_1, times_2]
                )
                additional_info = {
                    "phases": (phi_0, phi_1, phi_2),
                    "tau_coh": tau_coh,
                    "T_wait": T_wait,
                    "system": system,
                }
                # Return the data to create a plot with Plot_example_evo(res)!!
                return (
                    times_plot,
                    data_expectations,
                    pulse_seq_f,
                    additional_info,
                )

        # get the 2D Polarization data:
        for t_idx, t_det in enumerate(t_det_vals):
            actual_det_time = t_peak_pulse2 + t_det

            # Only compute polarization if:
            # 1. tau_coh + t_det <= t_max -> if the result is in the time range
            # 2. tau_coh + t_det < time_cut -> if the result is physically valid
            time_cut = kwargs.get("time_cut", np.inf)
            if actual_det_time < system.t_max and actual_det_time < time_cut:
                t_idx_in_times_2 = np.abs(times_2 - actual_det_time).argmin()

                # FINAL STATE
                rho_f = data_f.states[t_idx_in_times_2]

                if system.RWA_laser:
                    rho_f = apply_RWA_phase_factors(
                        rho_f,
                        times_2[t_idx_in_times_2],
                        omega=system.omega_laser,
                        system=system,
                    )

                value = expect(system.Dip_op, rho_f)
                data[tau_idx, t_idx] = np.real(value)
                # AXIS 0: tau_coh, AXIS 1: t_det

    return (
        t_det_vals,
        tau_coh_vals,
        data,  # (polarization itself is real)
    )


# ##########################
# parallel processing 1d and 2d data
# ##########################
def _process_single_1d_combination(
    omega_at: float,
    phi1: float,
    phi2: float,
    tau_coh: float,
    T_wait: float,
    times: np.ndarray,
    system: SystemParameters,
    **kwargs: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Process a single parameter combination for parallel 1D execution.
    This function runs in a separate process to avoid QUTIP thread safety issues.

    Parameters
    ----------
    omega_at : float
        Frequency for this combination.
    phi1 : float
        First phase value.
    phi2 : float
        Second phase value.
    tau_coh : float
        Coherence time.
    T_wait : float
        Waiting time.
    times : np.ndarray
        Time grid.
    system : SystemParameters
        System parameters (will be deep copied).
    kwargs : dict
        Additional arguments.

    Returns
    -------
    tuple or None
        (t_det_vals, data) for the computed 1D polarization data, or None if failed.
    """
    try:
        # Create a deep copy of the system to avoid threading issues
        system_copy = copy.deepcopy(system)
        system_copy.omega_A_cm = omega_at

        # Compute the 1D polarization for this specific combination
        t_det_vals, data = compute_fixed_tau_T(
            tau_coh=tau_coh,
            T_wait=T_wait,
            phi_0=phi1,
            phi_1=phi2,
            times=times,
            system=system_copy,
            **kwargs,
        )

        return t_det_vals, data

    except Exception as e:
        print(f"Error in _process_single_1d_combination: {str(e)}")
        return None


def _process_single_2d_combination(
    omega_at: float,
    phi1: float,
    phi2: float,
    T_wait: float,
    times: np.ndarray,
    system: SystemParameters,
    **kwargs: dict,
) -> np.ndarray:
    """
    Process a single parameter combination for parallel execution.
    This function runs in a separate process to avoid QUTIP thread safety issues. TODO <- is this actually a problem?

    Parameters
    ----------
    omega_at : float
        Frequency for this combination.
    phi1 : float
        First phase value.
    phi2 : float
        Second phase value.
    T_wait : float
        Waiting time.
    times : np.ndarray
        Time grid.
    system : SystemParameters
        System parameters (will be deep copied).
    kwargs : dict
        Additional arguments.

    Returns
    -------
    np.ndarray or None
        The computed 2D polarization data, or None if failed.
    """
    try:
        # Create system copy with new frequency
        system_new = copy.deepcopy(system)
        system_new.omega_A_cm = omega_at

        # Compute 2D polarization for this combination
        _, _, data = compute_two_dimensional_polarization(
            T_wait=T_wait,
            phi_0=phi1,
            phi_1=phi2,
            times=times,
            system=system_new,
            **kwargs,
        )

        return data

    except Exception as e:
        print(
            f"Error in _process_single_2d_combination: omega={omega_at}, phi1={phi1}, phi2={phi2}: {str(e)}"
        )
        return None


def parallel_compute_2d_polarization_with_inhomogenity(
    omega_ats: list,
    phases: list,
    times_T: np.ndarray,
    times: np.ndarray,
    system: SystemParameters,
    max_workers: int = None,
    **kwargs: dict,
) -> list:
    """
    Compute 2D REAL polarization using batch processing with parallelization of parameter combinations.
    Pre-computes all parameter combinations, processes them in parallel for each T_wait.

    Parameters
    ----------
    omega_ats : list
        List of different frequencies to simulate.
    phases : list
        List of phases for phase cycling.
    times_T : np.ndarray
        Array of T_wait values.
    times : np.ndarray
        Time grid for simulation.
    system : SystemParameters
        System parameters object.
    max_workers : int, optional
        Number of workers for parallel processing (default: CPU count).
    **kwargs : dict
        Additional keyword arguments for compute_two_dimensional_polarization.

    Returns
    -------
    list
        List of averaged 2D data arrays for each T_wait.
    """

    # Set default max_workers
    if max_workers is None:
        max_workers = mp.cpu_count()

    # =============================
    # PRE-COMPUTE ALL PARAMETER COMBINATIONS
    # =============================
    all_combinations = []
    for omega_at in omega_ats:
        for phi1 in phases:
            for phi2 in phases:
                all_combinations.append((omega_at, phi1, phi2))

    print(
        f"Processing {len(all_combinations)} parameter combinations for {len(times_T)} T_wait values"
    )
    print(f"Using {max_workers} parallel workers")

    # =============================
    # ITERATE OVER times_T AND PROCESS ALL COMBINATIONS IN PARALLEL
    # =============================
    all_results = []

    for T_wait_idx, T_wait in enumerate(times_T):
        print(f"Processing T_wait {T_wait_idx + 1}/{len(times_T)}: {T_wait}")

        # Get dimensions for this T_wait
        tau_coh_vals, t_det_vals = get_tau_cohs_and_t_dets_for_T_wait(
            times, T_wait=T_wait
        )

        if len(tau_coh_vals) == 0 or len(t_det_vals) == 0:
            all_results.append(None)
            continue

        # Pre-allocate accumulation array of real polarization data
        accumulated_data = np.zeros(
            (len(tau_coh_vals), len(t_det_vals)), dtype=np.complex64
        )
        total_count = 0

        # Process all combinations in parallel for this T_wait
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all combination tasks
            futures = [
                executor.submit(
                    _process_single_2d_combination,
                    omega_at=omega_at,
                    phi1=phi1,
                    phi2=phi2,
                    T_wait=T_wait,
                    times=times,
                    system=system,
                    kwargs=kwargs,
                )
                for omega_at, phi1, phi2 in all_combinations
            ]

            # Collect results as they complete
            completed_count = 0
            for future in as_completed(futures):
                try:
                    data_result = future.result()
                    if data_result is not None:
                        accumulated_data += data_result
                        total_count += 1

                    completed_count += 1
                    if completed_count % 10 == 0:
                        print(
                            f"  Completed {completed_count}/{len(all_combinations)} combinations"
                        )

                except Exception as e:
                    print(f"Error processing combination: {str(e)}")
                    continue

        # Average the accumulated data for this T_wait
        if total_count > 0:
            averaged_data = accumulated_data / total_count

        all_results.append(averaged_data)
        print(
            f"  Successfully processed {total_count}/{len(all_combinations)} combinations for T_wait={T_wait}"
        )

    return all_results


def parallel_compute_1d_polarization_with_inhomogenity(
    omega_ats: list,
    phases: list,
    tau_coh: float,
    T_wait: float,
    times: np.ndarray,
    system: SystemParameters,
    max_workers: int = None,
    **kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D REAL polarization using batch processing with parallelization of parameter combinations.
    Pre-computes all parameter combinations, processes them in parallel for fixed tau_coh and T_wait.

    Parameters
    ----------
    omega_ats : list
        List of different frequencies to simulate.
    phases : list
        List of phases for phase cycling.
    tau_coh : float
        Coherence time.
    T_wait : float
        Waiting time.
    times : np.ndarray
        Time grid for simulation.
    system : SystemParameters
        System parameters object.
    max_workers : int, optional
        Number of workers for parallel processing (default: CPU count).
    **kwargs : dict
        Additional keyword arguments for compute_fixed_tau_T.

    Returns
    -------
    tuple
        (t_det_vals, data_avg) where data_avg is averaged over all parameter combinations.
    """

    # Set default max_workers
    if max_workers is None:
        max_workers = mp.cpu_count()

    # =============================
    # PRE-COMPUTE ALL PARAMETER COMBINATIONS
    # =============================
    all_combinations = []
    for omega_at in omega_ats:
        for phi1 in phases:
            for phi2 in phases:
                all_combinations.append((omega_at, phi1, phi2))

    print(
        f"Processing {len(all_combinations)} parameter combinations for tau_coh={tau_coh}, T_wait={T_wait}"
    )
    print(f"Using {max_workers} parallel workers")

    # Initialize accumulation variables
    accumulated_data = None
    t_det_vals = None
    total_count = 0

    # Process all combinations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all combination tasks
        futures = [
            executor.submit(
                _process_single_1d_combination,
                omega_at=omega_at,
                phi1=phi1,
                phi2=phi2,
                tau_coh=tau_coh,
                T_wait=T_wait,
                times=times,
                system=system,
                **kwargs,
            )
            for omega_at, phi1, phi2 in all_combinations
        ]

        # Collect results as they complete
        completed_count = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    t_det_result, data_result = result

                    # Initialize accumulation array on first successful result
                    if accumulated_data is None:
                        t_det_vals = t_det_result
                        accumulated_data = np.zeros_like(
                            data_result, dtype=np.complex64
                        )

                    accumulated_data += data_result
                    total_count += 1

                completed_count += 1
                if completed_count % 10 == 0:
                    print(
                        f"  Completed {completed_count}/{len(all_combinations)} combinations"
                    )

            except Exception as e:
                print(f"Error processing combination: {str(e)}")
                continue

    # Average the accumulated data
    if total_count > 0:
        averaged_data = accumulated_data / total_count
        print(
            f"Successfully processed {total_count}/{len(all_combinations)} combinations"
        )
        return t_det_vals, averaged_data
    else:
        print("No successful combinations processed")
        return np.array([]), np.array([])
