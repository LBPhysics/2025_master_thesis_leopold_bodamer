# -*- coding: utf-8 -*-

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from qutip import Qobj, Result, mesolve, brmesolve, expect
from qutip.core import QobjEvo
from qspectro2d.core.system_parameters import SystemParameters
from qspectro2d.core.pulse_sequences import PulseSequence
from qspectro2d.core.pulse_functions import (
    identify_non_zero_pulse_regions,
    split_by_active_regions,
)
from qspectro2d.core.pulse_functions import *
from qspectro2d.core.solver_fcts import (
    matrix_ODE_paper,
    R_paper,
)
from qspectro2d.spectroscopy.inhomogenity import sample_from_sigma
from qspectro2d.core.functions_with_rwa import (
    H_int,
    get_expect_vals_with_RWA,
    apply_RWA_phase_factors,
)


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
        raise ValueError("Invalid or empty pulse sequence")"""

    # =============================
    # Set solver options
    # =============================
    # Initialize with provided solver_options or empty dict
    options = solver_options.copy() if solver_options else {}

    # Add default options if not already present
    default_options = {
        "store_states": True,
        "progress_bar": "",
        # Increasing max steps and atol/rtol for better stability
        "nsteps": 200000,
        "atol": 1e-6,
        "rtol": 1e-4,
    }

    # Update options with defaults only if not already set
    for key, value in default_options.items():
        if key not in options:
            options[key] = value

    # =============================
    # Choose solver and compute the evolution
    # =============================
    if system.ODE_Solver not in ["ME", "BR", "Paper_eqs", "Paper_BR"]:
        raise ValueError(f"Unknown ODE solver: {system.ODE_Solver}")

    # elif system.ODE_Solver == "BR": # TODO added THIS CASE, because i couldnt run with 2 atoms

    else:
        # Build Hamiltonian components
        H_free = system.H0_diagonalized  # already includes the RWA, if present!
        if H_free is None or not isinstance(H_free, Qobj):
            raise ValueError(f"Invalid H0_diagonalized: {H_free}")
        H_int_evo = H_free + QobjEvo(lambda t, args=None: H_int(t, pulse_seq, system))

        # Set up collapse operators based on solver type
        c_ops = []

        if system.ODE_Solver == "Paper_BR" or system.ODE_Solver == "Paper_eqs":
            c_ops = [R_paper(system)]
            if not all(isinstance(op, Qobj) for op in c_ops):
                raise ValueError(f"Invalid Redfield tensor: {c_ops}")
            if system.ODE_Solver == "Paper_eqs":
                if not system.RWA_laser:
                    raise ValueError(
                        "The equations of the paper only make sense with RWA"
                    )

                # For Paper_eqs, we need to define the full Liouville operator
                Liouville_full = QobjEvo(
                    lambda t, args=None: matrix_ODE_paper(t, pulse_seq, system)
                )
                H_int_evo = Liouville_full

        elif system.ODE_Solver == "ME":
            c_ops = system.c_ops_list
            if not all(isinstance(op, Qobj) for op in c_ops):
                raise ValueError(f"Invalid collapse operators: {c_ops}")
        elif system.ODE_Solver == "BR":
            if not hasattr(system, "a_ops_list") or system.a_
            ops_list is None:
                raise ValueError("Missing a_ops_list for BR solver")

        # =============================
        # Split evolution by pulse regions for Paper_eqs
        # =============================
        # Find pulse regions in the time array using the dedicated function
        pulse_regions = identify_non_zero_pulse_regions(times, pulse_seq)
        # BASED ON THIS split the time range into regions where the pulse envelope is zero
        split_times = split_by_active_regions(times, pulse_regions)

        # Initialize result storage for different regions
        all_states = []
        all_times = []
        current_state = psi_ini
        evolution_results = []  # Store individual evolution results

        for i, times_ in enumerate(split_times):
            if len(times_) == 0:
                continue

            # Find the indices in the original times array for this split
            start_idx = np.abs(times - times_[0]).argmin()

            # Check if this region has an active pulse by looking at the first time point
            has_pulse = pulse_regions[start_idx]

            if has_pulse:
                # =============================
                # Evolve with H_int_evo during pulse region
                # =============================
                if system.ODE_Solver == "BR":
                    result = brmesolve(
                        H_int_evo,
                        current_state,
                        times_,
                        a_ops=system.a_ops_list,
                        options=options,
                    )
                elif system.ODE_Solver == "Paper_eqs":
                    result = mesolve(
                        H_int_evo,  # Liouville includes the decay channels
                        current_state,
                        times_,
                        options=options,
                    )
                else:
                    result = mesolve(
                        H_int_evo,
                        current_state,
                        times_,
                        c_ops=c_ops,
                        options=options,
                    )
            else:
                # =============================
                # Evolve with H_free during non-pulse region
                # =============================
                if system.ODE_Solver == "BR":
                    result = brmesolve(
                        H_free,
                        current_state,
                        times_,
                        a_ops=system.a_ops_list,
                        options=options,
                    )
                else:
                    result = mesolve(
                        H_free,
                        current_state,
                        times_,
                        c_ops=c_ops,
                        options=options,
                    )

            # Store states (excluding the last one to avoid duplication, unless this is the last region)
            if i < len(split_times) - 1:
                all_states.extend(result.states)
                all_times.extend(result.times)
            else:
                all_states.extend(result.states)
                all_times.extend(result.times)

            # Update current state for next evolution
            current_state = result.states[-1]

            # Store the result for combining later
            if len(evolution_results) == 0:
                evolution_results.append(result)

        # =============================
        # Create combined result object using first result as base
        # =============================
        # Use the first result as a template and modify its data
        result = evolution_results[0]
        result.states = all_states
        result.times = np.array(all_times)

        # Ensure we have the correct number of states
        if len(result.states) != len(times):
            # Fallback: use full H_int_evo evolution if splitting failed
            print(
                "Warning: Pulse region splitting failed, falling back to full evolution"
            )
            if system.ODE_Solver == "BR":
                result = brmesolve(
                    H_int_evo,
                    psi_ini,
                    times,
                    a_ops=system.a_ops_list,
                    options=options,
                )
            elif system.ODE_Solver == "Paper_eqs":
                result = mesolve(
                    H_int_evo,
                    psi_ini,
                    times,
                    options=options,
                )
            else:
                result = mesolve(
                    H_int_evo,
                    psi_ini,
                    times,
                    c_ops=c_ops,
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
    t_peak_pulse1 = times[-1] / 2
    t_peak_pulse2 = times[-1] / 1.1

    # Use the from_pulse_specs static method to construct the sequence
    pulse_seq = PulseSequence.from_pulse_specs(
        system=system,
        pulse_specs=[
            (0, 0, phi_0),  # preprev: pulse 0 at t=0
            (1, t_peak_pulse1, phi_1),  # prev: pulse 1 at middle time
            (2, t_peak_pulse2, phi_2),  # curr: pulse 2 at end time
        ],
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


# 1D polarization calculation for given tau_coh, T_wait, varialbe t_det
def compute_1d_polarization(
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
    idx_start_pulse1 = np.abs(times - (tau_coh - system.FWHMs[1])).argmin()

    times_0 = times[
        : idx_start_pulse1 + 1
    ]  # definetly not empty except for when T_wait >= t_max
    if times_0.size == 0:
        times_0 = times[:2]  # idx_end_pulse0 + 1

    # calculate the evolution of the first pulse in the desired range for tau_coh

    # First pulse
    t_peak_pulse0 = 0
    pulse_0 = (
        t_peak_pulse0,
        phi_0,
    )  # Instead of directly constructing PulseSequence, use from_pulse_specs:
    pulse_seq_0 = PulseSequence.from_pulse_specs(
        system=system, pulse_specs=[(0, t_peak_pulse0, phi_0)]
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
        times_1 = [
            times[idx_start_pulse1]
        ]  # Handle overlapping pulses: If the second pulse starts before the first pulse ends, combine their contributions
    t_peak_pulse1 = tau_coh
    pulse_1 = (t_peak_pulse1, phi_1)
    pulse_seq_1 = PulseSequence.from_pulse_specs(
        system=system,
        pulse_specs=[
            (0, t_peak_pulse0, phi_0),  # prev: pulse 0
            (1, t_peak_pulse1, phi_1),  # curr: pulse 1
        ],
    )
    data_1 = compute_pulse_evolution(rho_1, times_1, pulse_seq_1, system=system)

    idx_start_pulse2_in_times_1 = np.abs(times_1 - (t_start_pulse2)).argmin()

    rho_2 = data_1.states[
        idx_start_pulse2_in_times_1
    ]  # == state where the third pulse starts

    times_2 = times[idx_start_pulse2:]
    if times_2.size == 0:
        times_2 = [
            times[-1]
        ]  # idx_start_pulse2 : idx_end_pulse2 + 1    # If the second pulse starts before the first pulse ends, combine their contributions
    phi_2 = 0  # FIXED PHASE!
    t_peak_pulse2 = tau_coh + T_wait
    pulse_f = (t_peak_pulse2, phi_2)
    pulse_seq_f = PulseSequence.from_pulse_specs(
        system=system,
        pulse_specs=[
            (0, t_peak_pulse0, phi_0),  # preprev: pulse 0
            (1, t_peak_pulse1, phi_1),  # prev: pulse 1
            (2, t_peak_pulse2, phi_2),  # curr: pulse 2
        ],
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
        times_plot = np.concatenate(
            [
                times_0[: idx_start_pulse1 + 1],
                times_1[: idx_start_pulse2_in_times_1 + 1],
                times_2,
            ]
        )

        additional_info = {
            "phases": (phi_0, phi_1, phi_2),
            "tau_coh": tau_coh,
            "T_wait": T_wait,
            # "system": system,
        }
        # Return the data to create a plot with Plot_example_evo(res)!!
        return (
            times_plot,
            data_expectations,
            pulse_seq_f,
            additional_info,
        )

    # =============================
    # SUBTRACT THE LINEAR SIGNALS for all the combinations (phi_1, phi_2):
    # =============================
    t_det_start_idx_in_times = np.abs(times - (t_peak_pulse2)).argmin()
    t_det_start_idx_in_times_2 = np.abs(
        times_2 - (t_peak_pulse2)
    ).argmin()  # detection time index in times_2    # JUST FIRST PULSE
    psi_0 = system.psi_ini
    pulse_seq_just0 = PulseSequence.from_pulse_specs(
        system=system, pulse_specs=[(0, t_peak_pulse0, phi_0)]
    )
    data_only_pulse0 = compute_pulse_evolution(
        psi_0, times, pulse_seq_just0, system=system
    )  # JUST SECOND PULSE
    pulse_seq_just1 = PulseSequence.from_pulse_specs(
        system=system, pulse_specs=[(1, t_peak_pulse1, phi_1)]
    )
    data_only_pulse1 = compute_pulse_evolution(
        psi_0, times, pulse_seq_just1, system=system
    )  # JUST THIRD PULSE
    pulse_seq_just2 = PulseSequence.from_pulse_specs(
        system=system, pulse_specs=[(2, t_peak_pulse2, phi_2)]
    )
    data_only_pulse2 = compute_pulse_evolution(
        psi_0, times, pulse_seq_just2, system=system
    )

    actual_det_times = data_f.times[t_det_start_idx_in_times_2:]
    states_full = data_f.states[t_det_start_idx_in_times_2:]  # whole numerical signal
    states_only_pulse0 = data_only_pulse0.states[t_det_start_idx_in_times:]
    states_only_pulse1 = data_only_pulse1.states[t_det_start_idx_in_times:]
    states_only_pulse2 = data_only_pulse2.states[t_det_start_idx_in_times:]

    P_full = np.zeros((len(actual_det_times)), dtype=np.complex64)
    P_only0 = np.zeros((len(actual_det_times)), dtype=np.complex64)
    P_only1 = np.zeros((len(actual_det_times)), dtype=np.complex64)
    P_only2 = np.zeros((len(actual_det_times)), dtype=np.complex64)

    # Apply RWA phase factors to the states at detection time t_det
    if system.RWA_laser:
        states_full = [
            apply_RWA_phase_factors(
                state, time, omega=system.omega_laser, system=system
            )
            for state, time in zip(states_full, actual_det_times)
        ]
        states_only_pulse0 = [
            apply_RWA_phase_factors(
                state, time, omega=system.omega_laser, system=system
            )
            for state, time in zip(states_only_pulse0, actual_det_times)
        ]
        states_only_pulse1 = [
            apply_RWA_phase_factors(
                state, time, omega=system.omega_laser, system=system
            )
            for state, time in zip(states_only_pulse1, actual_det_times)
        ]
        states_only_pulse2 = [
            apply_RWA_phase_factors(
                state, time, omega=system.omega_laser, system=system
            )
            for state, time in zip(states_only_pulse2, actual_det_times)
        ]

    for t_idx, t_det in enumerate(actual_det_times):
        # only if we are still in the physical regime
        if t_det < kwargs.get("time_cut", np.inf):

            # complex Polarization calculation -> only extract the third order into data
            P_full[t_idx] = system.Dip_op[1, 0] * states_full[t_idx][0, 1]
            P_only0[t_idx] = system.Dip_op[1, 0] * states_only_pulse0[t_idx][0, 1]
            P_only1[t_idx] = system.Dip_op[1, 0] * states_only_pulse1[t_idx][0, 1]
            P_only2[t_idx] = system.Dip_op[1, 0] * states_only_pulse2[t_idx][0, 1]
            """
            # real Polarization:
            P_full[t_idx] = np.real(expect(system.Dip_op, states_full[t_idx]))
            P_only0[t_idx] = np.real(expect(system.Dip_op, states_only_pulse0[t_idx]))
            P_only1[t_idx] = np.real(expect(system.Dip_op, states_only_pulse1[t_idx]))
            P_only2[t_idx] = np.real(expect(system.Dip_op, states_only_pulse2[t_idx]))
            """

    # If plot_example_Polarization is True, return the full data for plotting
    if kwargs.get("plot_example_Polarization", False):
        return (
            actual_det_times - actual_det_times[0],
            P_full,
            P_only0,
            P_only1,
            P_only2,
        )

    data = P_full - (P_only0 + P_only1 + P_only2)

    return np.array(actual_det_times) - actual_det_times[0], data


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


# 2D polarization calculation for given T_wait, varialbe tau_coh and t_det
def compute_2d_polarization(
    T_wait: float,
    phi_0: float,
    phi_1: float,
    times: np.ndarray,
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
    # Extra input validation
    if not isinstance(times, np.ndarray):
        raise TypeError(f"Expected times to be numpy.ndarray, got {type(times)}")
    if len(times) < 2:
        raise ValueError(f"Times array is too short: {len(times)} elements")
    if not (0 <= T_wait <= system.t_max):
        raise ValueError(f"T_wait={T_wait} must be between 0 and t_max={system.t_max}")

    # get the symmetric times, tau_coh, t_det
    tau_coh_vals, t_det_vals = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait=T_wait)

    # Check for empty arrays
    if len(tau_coh_vals) == 0 or len(t_det_vals) == 0:
        return np.array([0.0]), np.array([0.0]), np.zeros((1, 1), dtype=np.complex64)

    # initialize the time domain Spectroscopy data tr(Dip_op * rho_final(tau_coh, t_det))
    data = np.zeros(
        (len(tau_coh_vals), len(t_det_vals)), dtype=np.complex64
    )  # information about the first pulse
    t_peak_pulse0 = 0
    pulse_0 = (t_peak_pulse0, phi_0)
    pulse_seq_0 = PulseSequence.from_pulse_specs(
        system=system, pulse_specs=[(0, t_peak_pulse0, phi_0)]
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
        pulse_seq_1 = PulseSequence.from_pulse_specs(
            system=system,
            pulse_specs=[
                (0, t_peak_pulse0, phi_0),  # prev: pulse 0
                (1, t_peak_pulse1, phi_1),  # curr: pulse 1
            ],
        )
        data_1 = compute_pulse_evolution(rho_1, times_1, pulse_seq_1, system=system)

        idx_start_pulse2_in_times_1 = idx_start_pulse2 - idx_start_pulse1
        rho_2 = data_1.states[idx_start_pulse2_in_times_1]

        times_2 = times[idx_start_pulse2:]
        if times_2.size == 0:
            times_2 = [times[-1]]

        phi_2 = 0
        pulse_f = (t_peak_pulse2, phi_2)
        pulse_seq_f = PulseSequence.from_pulse_specs(
            system=system,
            pulse_specs=[
                (0, t_peak_pulse0, phi_0),  # preprev: pulse 0
                (1, t_peak_pulse1, phi_1),  # prev: pulse 1
                (2, t_peak_pulse2, phi_2),  # curr: pulse 2
            ],
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
            # 3. We have enough elements in our data arrays
            time_cut = kwargs.get("time_cut", np.inf)
            if (
                t_idx + tau_idx < len(tau_coh_vals)
                and actual_det_time < system.t_max  # TODO Not needed anymore?
                and actual_det_time < time_cut
                and len(times_2) > 0
                and len(data_f.states) > 0
            ):

                # Safety check for index bounds
                t_idx_in_times_2 = np.abs(times_2 - actual_det_time).argmin()
                if t_idx_in_times_2 >= len(data_f.states):
                    print(
                        f"WARNING: Index {t_idx_in_times_2} out of bounds for states array of length {len(data_f.states)}"
                    )
                    continue

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
    phi1: float,
    phi2: float,
    tau_coh: float,
    T_wait: float,
    times: np.ndarray,
    system: SystemParameters,
    **kwargs: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Process a single phase combination for IFT-based 1D execution.
    This function runs in a separate process for parallel phase cycling.

    Parameters
    ----------
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
        System parameters (already contains the correct omega_A_cm).
    kwargs : dict
        Additional arguments.

    Returns
    -------
    tuple or None
        (t_det_vals, data) for the computed 1D polarization data, or None if failed.
    """
    try:
        # Compute the 1D polarization for this specific phase combination
        t_det_vals, data = compute_1d_polarization(
            tau_coh=tau_coh,
            T_wait=T_wait,
            phi_0=phi1,
            phi_1=phi2,
            times=times,
            system=system,
            **kwargs,
        )

        return t_det_vals, data

    except Exception as e:
        print(f"Error in _process_single_1d_combination: {str(e)}")
        return None


def parallel_compute_1d_E_with_inhomogenity(
    n_freqs: int,  # Number of frequencies for inhomogeneous broadening
    n_phases: int,  # Number of phases for phase cycling
    tau_coh: float,
    T_wait: float,
    times: np.ndarray,
    system: SystemParameters,
    max_workers: int = None,
    **kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D REAL polarization with frequency loop and phase cycling for IFT processing.
    Loops over omega_ats sequentially, parallelizes phase combinations, then performs IFT
    and averages over frequencies.

    Parameters
    ----------
    n_freqs : int
        Number of frequencies to simulate.
    n_phases : int
        Number of phases for phase cycling (should be 4).
    tau_coh : float
        Coherence time.
    T_wait : float
        Waiting time.
    times : np.ndarray
        Time grid for simulation.
    system : SystemParameters
        System parameters object.
    max_workers : int, optional
        Number of workers for parallel processing (default: CPU count).    **kwargs : dict
        Additional keyword arguments for compute_1d_polarization.

    Returns
    -------
    tuple
        (t_det_vals, data_avg) where data_avg is averaged over frequencies after IFT.
    """

    # Set default max_workers
    if max_workers is None:
        max_workers = mp.cpu_count()

    # =============================
    # VALIDATE PHASE INPUT FOR IFT
    # =============================
    phases = [k * np.pi / 2 for k in range(n_phases)]  # [0, π/2, π, 3π/2]
    if n_phases != 4:
        print(f"Warning: Phases {phases} may not be optimal for IFT")

    print(
        f"Processing {n_freqs} frequencies with {n_phases}×{n_phases} phase combinations"
    )
    print(f"Using {max_workers} parallel workers for phase combinations")

    # =============================
    # INITIALIZE STORAGE FOR FREQUENCY AVERAGING
    # =============================
    omega_frequency_results = []  # Store IFT results for each frequency
    t_det_vals = None

    phase_combinations = []
    for phi1_idx, phi1 in enumerate(phases):
        for phi2_idx, phi2 in enumerate(phases):
            phase_combinations.append((phi1_idx, phi2_idx, phi1, phi2))

    # =============================
    # LOOP OVER FREQUENCIES SEQUENTIALLY
    # =============================
    for omega_idx in range(n_freqs):
        # Sample new frequencies
        new_omega_A = sample_from_sigma(1, system.Delta_cm, system.omega_A_cm).item()

        # Set sampled frequencies in the copied system
        system.omega_A_cm = new_omega_A

        if system.N_atoms == 1:
            print(
                f"\nProcessing frequency {omega_idx + 1}/{n_freqs}: ω_A = {new_omega_A:.2f} cm⁻¹"
            )

        elif system.N_atoms == 2:
            new_omega_B = sample_from_sigma(
                1, system.Delta_cm, system.omega_B_cm
            ).item()
            system.omega_B_cm = new_omega_B
            print(
                f"\nProcessing frequencies {omega_idx + 1}/{n_freqs}: "
                f"ω_A = {new_omega_A:.2f} cm⁻¹, ω_B = {new_omega_B:.2f} cm⁻¹"
            )
        # =============================
        # PARALLELIZE PHASE COMBINATIONS FOR THIS FREQUENCY
        # =============================
        # Initialize results matrix for this frequency: [phi1_idx, phi2_idx]
        results_matrix = np.zeros((n_phases, n_phases), dtype=object)

        # Process phase combinations in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit phase combination tasks
            futures = {
                executor.submit(
                    _process_single_1d_combination,
                    phi1=phi1,
                    phi2=phi2,
                    tau_coh=tau_coh,
                    T_wait=T_wait,
                    times=times,
                    system=system,
                    **kwargs,
                ): (phi1_idx, phi2_idx)
                for phi1_idx, phi2_idx, phi1, phi2 in phase_combinations
            }

            # Collect results as they complete
            completed_phase_count = 0
            for future in as_completed(futures):
                phi1_idx, phi2_idx = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        t_det_result, data_result = result

                        # Store in results matrix
                        results_matrix[phi1_idx, phi2_idx] = data_result

                        # Set t_det_vals on first successful result
                        if t_det_vals is None:
                            t_det_vals = t_det_result

                    completed_phase_count += 1
                    if completed_phase_count % 4 == 0:
                        print(
                            f"  Completed {completed_phase_count}/{len(phase_combinations)} phase combinations"
                        )

                except Exception as e:
                    print(
                        f"Error processing phase combination ({phi1_idx}, {phi2_idx}): {str(e)}"
                    )
                    continue

        # =============================
        # PERFORM INVERSE FOURIER TRANSFORM FOR THIS FREQUENCY
        # =============================
        print(f"  Performing IFT for frequency {omega_idx + 1}/{n_freqs}...")

        # Extract photon echo component P_{-1,1}(t) using discrete IFT
        # P_{-1,1}(t) = Σ_{m1=0}^3 Σ_{m2=0}^3 P_{m1,m2}(t) * exp(-i(-1*m1*π/2 + 1*m2*π/2))

        if t_det_vals is not None:  # TODO get rid of this t_det_vals dependance
            # Extract photon echo component P_{-1,1}(t) using our new function
            # with coefficients l=-1, m=1 for the photon echo signal
            """
            photon_echo_signal = (
                np.sum(results_matrix) / n_phases**2
            )  # Average over all phase combinations

            """
            photon_echo_signal = extract_ift_signal_component(
                results_matrix=results_matrix,
                phases=phases,
                l=-1,  # Coefficient for phi_1
                m=1,  # Coefficient for phi_2
            )

            omega_frequency_results.append(photon_echo_signal)

            omega_frequency_results.append(photon_echo_signal)
            print(f"  ✅ IFT completed for frequency {omega_idx + 1}")
        else:
            print(f"  ❌ No valid results for frequency {omega_idx + 1}")
            omega_frequency_results.append(None)

    # =============================
    # AVERAGE OVER FREQUENCIES
    # =============================
    print(f"\nAveraging over {len(omega_frequency_results)} frequencies...")

    valid_results = [res for res in omega_frequency_results if res is not None]

    if len(valid_results) > 0:
        # Average over all valid frequency results
        final_averaged_data = np.mean(valid_results, axis=0)
        print(
            f"✅ Successfully averaged {len(valid_results)}/{n_freqs} frequency results"
        )

        return t_det_vals, final_averaged_data * 1j  # because E ~ iP
    else:
        print("❌ No valid frequency results to average")
        return np.array([]), np.array([])


def _process_single_2d_combination(
    phi1: float,
    phi2: float,
    T_wait: float,
    times: np.ndarray,
    system: SystemParameters,
    **kwargs: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Process a single phase combination for IFT-based 2D execution.
    This function runs in a separate process for parallel phase cycling.

    Parameters
    ----------
    phi1 : float
        First phase value.
    phi2 : float
        Second phase value.
    T_wait : float
        Waiting time.
    times : np.ndarray
        Time grid.
    system : SystemParameters
        System parameters (already contains the correct omega_A_cm).
    kwargs : dict
        Additional arguments.

    Returns
    -------
    tuple or None
        (t_det_vals, tau_coh_vals, data) for the computed 2D polarization data, or None if failed.
    """
    try:  # Compute the 2D polarization for this specific phase combination
        # Get expected dimensions for this T_wait
        tau_coh_vals, t_det_vals = get_tau_cohs_and_t_dets_for_T_wait(
            times, T_wait=T_wait
        )

        if len(tau_coh_vals) == 0 or len(t_det_vals) == 0:
            print(
                f"DEBUG: Invalid dimensions - skipping computation for T_wait={T_wait}"
            )
            return None

        t_det_vals, tau_coh_vals, data = compute_2d_polarization(
            T_wait=T_wait,
            phi_0=phi1,
            phi_1=phi2,
            times=times,
            system=system,
            **kwargs,
        )

        return t_det_vals, tau_coh_vals, data

    except Exception as e:
        import traceback

        print(f"Error in _process_single_2d_combination: {str(e)}")
        print(f"Traceback for debugging:")
        traceback.print_exc()
        return None


def parallel_compute_2d_E_with_inhomogenity(
    n_freqs: int,  # Number of frequencies for inhomogeneous broadening
    n_phases: int,  # Number of phases for phase cycling
    times_T: np.ndarray,
    times: np.ndarray,
    system: SystemParameters,
    max_workers: int = None,
    **kwargs: dict,
) -> list:
    """
    Compute 2D REAL polarization with frequency loop and phase cycling for IFT processing.
    Loops over omega_ats sequentially, parallelizes phase combinations, then performs IFT
    and averages over frequencies for each T_wait.

    Parameters
    ----------
    n_freqs : int
        Number of frequencies to simulate.
    n_phases : int
        Number of phases for phase cycling (should be 4).
    times_T : np.ndarray
        Array of T_wait values.
    times : np.ndarray
        Time grid for simulation.
    system : SystemParameters
        System parameters object.
    max_workers : int, optional
        Number of workers for parallel processing (default: CPU count).    **kwargs : dict
        Additional keyword arguments for compute_2d_polarization.

    Returns
    -------
    list
        List of averaged 2D data arrays for each T_wait (after IFT processing).
    """

    # Set default max_workers
    if max_workers is None:
        max_workers = mp.cpu_count()

    print(f"Processing {len(times_T)} T_wait values")
    print(f"Using {max_workers} parallel workers for phase combinations")

    # =============================
    # VALIDATE PHASE INPUT FOR IFT
    # =============================
    phases = [k * np.pi / 2 for k in range(n_phases)]  # [0, π/2, π, 3π/2]

    if n_phases != 4:
        print(f"Warning: Phases {phases} may not be optimal for IFT")

    print(
        f"Processing {n_freqs} frequencies with {n_phases}×{n_phases} phase combinations"
    )

    phase_combinations = []
    for phi1_idx, phi1 in enumerate(phases):
        for phi2_idx, phi2 in enumerate(phases):
            phase_combinations.append((phi1_idx, phi2_idx, phi1, phi2))

    # =============================
    # PROCESS EACH T_WAIT VALUE
    # =============================
    all_results = []

    for T_wait_idx, T_wait in enumerate(times_T):
        if T_wait < 0 or T_wait > system.t_max:
            # Skip invalid T_wait values
            print(
                f"Skipping T_wait={T_wait} (must be >= 0 and <= t_max={system.t_max})"
            )
            all_results.append(None)
            continue

        # Get dimensions for this T_wait
        tau_coh_vals, t_det_vals = get_tau_cohs_and_t_dets_for_T_wait(
            times, T_wait=T_wait
        )

        if len(tau_coh_vals) == 0 or len(t_det_vals) == 0:
            print(f"  ❌ Invalid T_wait={T_wait}, skipping")
            all_results.append(None)
            continue

        print(f"\nProcessing T_wait {T_wait_idx + 1}/{len(times_T)}: {T_wait}")

        # =============================
        # INITIALIZE STORAGE FOR FREQUENCY AVERAGING
        # =============================
        omega_frequency_results = []  # Store IFT results for each frequency

        # =============================
        # LOOP OVER FREQUENCIES SEQUENTIALLY
        # =============================
        for omega_idx in range(n_freqs):
            # Sample new frequencies
            new_omega_A = sample_from_sigma(
                1, system.Delta_cm, system.omega_A_cm
            ).item()

            # Set sampled frequencies in the copied system
            system.omega_A_cm = new_omega_A

            if system.N_atoms == 1:
                print(
                    f"\nProcessing frequency {omega_idx + 1}/{n_freqs}: ω_A = {new_omega_A:.2f} cm⁻¹"
                )

            elif system.N_atoms == 2:
                new_omega_B = sample_from_sigma(
                    1, system.Delta_cm, system.omega_B_cm
                ).item()
                system.omega_B_cm = new_omega_B
                print(
                    f"\nProcessing frequencies {omega_idx + 1}/{n_freqs}: "
                    f"ω_A = {new_omega_A:.2f} cm⁻¹, ω_B = {new_omega_B:.2f} cm⁻¹"
                )

            # =============================
            # =============================
            # Initialize results matrix for this frequency: [phi1_idx, phi2_idx]
            results_matrix = np.zeros((n_phases, n_phases), dtype=object)

            # Process phase combinations in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit phase combination tasks
                futures = {
                    executor.submit(
                        _process_single_2d_combination,
                        phi1=phi1,
                        phi2=phi2,
                        T_wait=T_wait,
                        times=times,
                        system=system,
                        **kwargs,
                    ): (phi1_idx, phi2_idx)
                    for phi1_idx, phi2_idx, phi1, phi2 in phase_combinations
                }

                # Collect results as they complete
                completed_phase_count = 0
                for future in as_completed(futures):
                    phi1_idx, phi2_idx = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            t_det_result, tau_coh_result, data_result = result

                            # Store in results matrix
                            results_matrix[phi1_idx, phi2_idx] = data_result

                        completed_phase_count += 1
                        if completed_phase_count % 4 == 0:
                            print(
                                f"    Completed {completed_phase_count}/{len(phase_combinations)} phase combinations"
                            )

                    except Exception as e:
                        print(
                            f"    Error processing phase combination ({phi1_idx}, {phi2_idx}): {str(e)}"
                        )
                        continue

            # =============================
            # PERFORM INVERSE FOURIER TRANSFORM FOR THIS FREQUENCY
            # =============================
            print(f"    Performing IFT for frequency {omega_idx + 1}")

            # Extract photon echo component P_{-1,1}(t) using discrete IFT
            # P_{-1,1}(t) = Σ_{m1=0}^3 Σ_{m2=0}^3 P_{m1,m2}(t) * exp(-i(-1*m1*π/2 + 1*m2*π/2))

            # Check if we have valid results
            valid_results = [
                results_matrix[m1, m2]
                for m1 in range(n_phases)
                for m2 in range(n_phases)
                if results_matrix[m1, m2] is not None
            ]

            if len(valid_results) > 0:
                # Extract photon echo component using our new function
                photon_echo_signal = extract_ift_signal_component(
                    results_matrix=results_matrix,
                    phases=phases,
                    l=-1,  # Coefficient for phi_1
                    m=1,  # Coefficient for phi_2
                )

                omega_frequency_results.append(photon_echo_signal)
                print(f"    ✅ IFT completed for frequency {omega_idx + 1}")
            else:
                print(f"    ❌ No valid results for frequency {omega_idx + 1}")
                omega_frequency_results.append(None)

        # =============================
        # AVERAGE OVER FREQUENCIES FOR THIS T_WAIT
        # =============================
        print(
            f"  Averaging over {len(omega_frequency_results)} frequencies for T_wait={T_wait}"
        )

        valid_results = [res for res in omega_frequency_results if res is not None]

        if len(valid_results) > 0:
            # Average over all valid frequency results
            final_averaged_data = np.mean(valid_results, axis=0)
            print(
                f"  ✅ Successfully averaged {len(valid_results)}/{n_freqs} frequency results"
            )
            all_results.append(final_averaged_data * 1j)  # because E ~ iP
        else:
            print(f"  ❌ No valid frequency results to average for T_wait={T_wait}")
            all_results.append(None)

    return all_results


# ##########################
# Helper functions for IFT processing
# ##########################
def extract_ift_signal_component(
    results_matrix: np.ndarray, phases: list, l: int, m: int
) -> np.ndarray:
    """
    Extract a specific signal component using inverse Fourier transform (IFT)
    with custom phase coefficients. Works for both 1D and 2D signal arrays.

    Computes the IFT signal component:
    P_{l,m}(t) = Σ_{phi1} Σ_{phi2} P_{phi1,phi2}(t) * exp(-i(l*phi1 + m*phi2))

    Parameters
    ----------
    results_matrix : np.ndarray
        Matrix of results indexed by phase indices [phi1_idx, phi2_idx]
        Each element can be a 1D or 2D array
    phases : list
        List of phase values used (typically [0, π/2, π, 3π/2])
    l : int
        Coefficient for the first phase in the IFT
    m : int
        Coefficient for the second phase in the IFT

    Returns
    -------
    np.ndarray
        Extracted signal component after IFT (same shape as input data elements)
    """
    n_phases = len(phases)

    # Find first non-None result to determine output shape, should be the first one !
    first_valid_result = None
    for i in range(n_phases):
        for j in range(n_phases):
            if results_matrix[i, j] is not None:
                first_valid_result = results_matrix[i, j]
                break
        if first_valid_result is not None:
            break

    if first_valid_result is None:
        # No valid results found
        return None

    # Initialize output array based on the shape of the first valid result
    signal = np.zeros_like(first_valid_result, dtype=np.complex64)

    # Compute the IFT
    for phi1_idx, phi_1 in enumerate(phases):
        for phi2_idx, phi_2 in enumerate(phases):
            if results_matrix[phi1_idx, phi2_idx] is not None:
                # IFT phase factor with parameterized coefficients
                phase_factor = np.exp(-1j * (l * phi_1 + m * phi_2))  # n * phi_3 is 0!
                signal += results_matrix[phi1_idx, phi2_idx] * phase_factor

    # Normalize by number of phase combinations
    signal /= n_phases * n_phases

    return signal
