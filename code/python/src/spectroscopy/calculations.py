# -*- coding: utf-8 -*-
# TODO REALLY change the structure to match pulse_peak_time
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from qutip import Qobj, mesolve, brmesolve, expect, qutip
from qutip.core import QobjEvo
from src.core.system_parameters import SystemParameters
from src.core.pulse_functions import *
from src.core.pulse_sequences import PulseSequence
from src.core.solver_fcts import (
    matrix_ODE_paper,
    R_paper,
    apply_RWA_phase_factors,
    get_expect_vals_with_RWA,  # TODO where did I put this? and where should it go?
)
from src.spectroscopy.inhomogenity import sample_from_sigma
from src.core.hamiltonians import H_int


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
    t_det = tau_coh + T_wait

    # =============================
    # Ensure t_det does not exceed t_max due to floating point
    # =============================
    valid_idx = t_det <= t_max + 1e-10
    tau_coh = tau_coh[valid_idx]
    t_det = t_det[valid_idx]

    return tau_coh, t_det


def compute_pulse_evolution(
    psi_ini: Qobj,
    times: np.ndarray,
    pulse_seq: PulseSequence,
    system: SystemParameters = None,
) -> qutip.Result:
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


def compute_two_dimensional_polarization(
    T_wait: float,
    phi_0: float,
    phi_1: float,
    times: np.ndarray,  # ACTURALLY a list -> qutip needs a list ?
    system: SystemParameters,
    time_cut: float = np.inf,  # to avoid numerical issues
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

    plot_example = kwargs.get("plot_example", False)

    # get the symmetric times, tau_coh, t_det
    tau_coh_vals, t_det_vals = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait=T_wait)

    # initialize the time domain Spectroscopy data tr(Dip_op * rho_final(tau_coh, t_det))
    data = np.zeros((len(tau_coh_vals), len(t_det_vals)), dtype=np.float32)

    t_peak_pulse0 = 0
    idx_pulse1_max_peak = np.abs(
        times - (tau_coh_vals[-1])
    ).argmin()  # the last possible coherence time

    times_0 = times[: idx_pulse1_max_peak + 1]
    if times_0.size == 0:
        # TODO check if this case ever happens
        idx_end_pulse0 = np.abs(times - (system.Delta_ts[0])).argmin()
        times_0 = times[: idx_end_pulse0 + 1]

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

    for tau_idx, tau_coh in enumerate(tau_coh_vals):
        idx_start_pulse1 = np.abs(
            times - (tau_coh - system.Delta_ts[1])
        ).argmin()  # from which point to start the next pulse
        rho_1 = data_0.states[idx_start_pulse1]

        idx_start_pulse2 = np.abs(
            times - (tau_coh + T_wait - system.Delta_ts[2])
        ).argmin()
        idx_end_pulse2 = np.abs(
            times - (tau_coh + T_wait + system.Delta_ts[2])
        ).argmin()
        t_start_2 = times[idx_start_pulse2]
        t_peak_pulse2 = tau_coh + T_wait

        times_1 = times[idx_start_pulse1 : idx_start_pulse2 + 1]
        if times_1.size == 0:
            # TODO check if this case ever happens
            idx_end_pulse1 = np.abs(times - (tau_coh + system.Delta_ts[1])).argmin()
            times_1 = times[idx_start_pulse1 : idx_end_pulse1 + 1]

        t_peak_pulse1 = tau_coh
        pulse_1 = (t_peak_pulse1, phi_1)
        pulse_seq_1 = PulseSequence.from_args(
            system=system,
            curr=pulse_1,
            prev=pulse_0,
        )
        data_1 = compute_pulse_evolution(rho_1, times_1, pulse_seq_1, system=system)

        idx_start_pulse2_in_times_1 = np.abs(
            times_1 - t_start_2
        ).argmin()  # TODO Check this should be equal to (idx_start_pulse2 - idx_start_pulse1)!!!

        rho_2 = data_1.states[idx_start_pulse2_in_times_1]

        times_2 = times[idx_start_pulse2:]
        if times_2.size == 0:
            times_2 = times[idx_start_pulse2 : idx_end_pulse2 + 1]

        phi_2 = 0
        pulse_f = (t_peak_pulse2, phi_2)
        pulse_seq_f = PulseSequence.from_args(
            system=system,
            curr=pulse_f,
            prev=pulse_1,
            preprev=pulse_0,
        )
        data_f = compute_pulse_evolution(rho_2, times_2, pulse_seq_f, system=system)

        for t_idx, t_det in enumerate(t_det_vals):
            actual_det_time = t_peak_pulse2 + t_det

            if actual_det_time < system.t_max and actual_det_time < time_cut:
                t_idx_in_times_2 = np.abs(
                    times_2 - actual_det_time
                ).argmin()  # TODO this should be equal to (t_idx)

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
                data[tau_idx, t_idx] = np.real(
                    value
                )  # TODO check the dtype -> np.float32 or am i missing precission?

                if (
                    t_idx == 0
                    and tau_idx == 0  # len(tau_coh_vals) // 3
                    and plot_example
                ):
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

                    # Note: Plotting should be done in visualization layer, not calculation layer
                    # TODO: Move Plot_example_evo to a separate plotting script if needed
                    # Plot_example_evo(
                    #     times_0[: idx_start_pulse1 + 1],
                    #     times_1,
                    #     times_2,
                    #     data_expectations,
                    #     pulse_seq_f,
                    #     tau_coh,
                    #     T_wait,
                    #     system=system,
                    # )

    return (
        t_det_vals,
        tau_coh_vals,
        data,  # (polarization itself is real)
    )


# ##########################
# dependent of system
# ##########################
def compute_many_polarizations(
    T_wait: float,
    phi_0: float,
    phi_1: float,
    N: int,
    E0: float,
    Delta: float,
    times: np.ndarray,
    system: SystemParameters,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calls compute_two_dimensional_polarization N times for different omega_ats sampled from σ.

    Parameters
    ----------
    T_wait : float
        Delay between pulses.
    phi_0 : float
        Phase of the first pulse.
    phi_1 : float
        Phase of the second pulse.
    N : int
        Number of simulations (samples).
    E0 : float
        Center of the frequency distribution.
    Delta : float
        FWHM for sigma distribution.
    times : np.ndarray
        Time grid for simulation.
    **kwargs : dict
        Additional arguments for compute_two_dimensional_polarization.

    Returns
    -------
    tuple
        (t_det_vals, tau_coh_vals, data_avg, omega_ats)
    """
    # =============================
    # Sample omega_ats from sigma distribution
    # =============================
    omega_ats = sample_from_sigma(N, Delta, E0)
    results = []

    # =============================
    # Run simulations for each sampled omega
    # =============================
    for omega in omega_ats:
        system_new = copy.deepcopy(system)  # create a copy of the system
        system_new.omega_A_cm = omega
        # print("new omega_A", system_new.omega_A)
        data = compute_two_dimensional_polarization(
            T_wait=T_wait,
            phi_0=phi_0,
            phi_1=phi_1,
            times=times,
            system=system_new,
            **kwargs,
        )

        results.append(data[2])  # only store the data array

    data_avg = np.mean(results, axis=0)  # average over all samples
    t_det_vals = data[0]  # detection times from last run
    tau_coh_vals = data[1]  # coherence times from last run

    return t_det_vals, tau_coh_vals, data_avg, omega_ats


# ##########################
# functions for parallel processing
# ##########################


def batch_process_all_combinations_with_inhomogeneity(
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
            (len(tau_coh_vals), len(t_det_vals)), dtype=np.float32
        )
        total_count = 0

        # Process all combinations in parallel for this T_wait
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all combination tasks
            futures = [
                executor.submit(
                    _process_single_combination,
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


def _process_single_combination(
    omega_at: float,
    phi1: float,
    phi2: float,
    T_wait: float,
    times: np.ndarray,
    system: SystemParameters,
    kwargs: dict,
) -> np.ndarray:
    """
    Process a single parameter combination for parallel execution.
    This function runs in a separate process to avoid QUTIP thread safety issues.

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
            f"Error in _process_single_combination: omega={omega_at}, phi1={phi1}, phi2={phi2}: {str(e)}"
        )
        return None
