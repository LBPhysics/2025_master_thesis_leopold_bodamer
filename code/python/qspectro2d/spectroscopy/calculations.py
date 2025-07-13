# -*- coding: utf-8 -*-

# =============================
# STANDARD LIBRARY IMPORTS
# =============================
from cmath import polar
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Union, Tuple
import logging

# =============================
# THIRD-PARTY IMPORTS
# =============================
import numpy as np
from qutip import Qobj, Result, mesolve, brmesolve, ket2dm
from qutip.core import QobjEvo

# =============================
# LOCAL IMPORTS
# =============================
from qspectro2d.core.simulation_class import SimulationModuleOQS
from qspectro2d.core.laser_system.laser_class import (
    LaserPulseSequence,
    identify_non_zero_pulse_regions,
    split_by_active_regions,
)
from qspectro2d.spectroscopy.inhomogenity import sample_from_gaussian
from qspectro2d.core.functions_with_rwa import (
    apply_RWA_phase_factors,
    get_expect_vals_with_RWA,
)
from qspectro2d.spectroscopy.simulation.utils import (
    PHASE_CYCLING_PHASES,
    DEFAULT_SOLVER_OPTIONS,
    NEGATIVE_EIGVAL_THRESHOLD,
    TRACE_TOLERANCE,
    DETECTION_PHASE,
)


# =============================
# LOGGING CONFIGURATION
# =============================
logger = logging.getLogger(__name__)


def compute_pulse_evolution(
    sim_oqs: SimulationModuleOQS,
    **solver_options: dict,
) -> Result:
    """
    Compute the evolution of the system for a given pulse sequence.

    Parameters
    ----------
    psi_ini : Qobj
        Initial quantum state.
    times : np.ndarray
        Time array for the evolution.
    ops : SimulationModuleOQS contains simulation configuration, system, laser parameters and much more.
    pulse_seq : LaserPulseSequence
        LaserPulseSequence object defining the pulse sequence.
    system : AtomicSystem
        System parameters containing Hamiltonian and solver configuration.
    **solver_options : dict
        Additional solver options that override defaults.

    Returns
    -------
    Result
        QuTiP Result object containing evolution data.

    Raises
    ------
    ValueError
        If unknown ODE solver is specified.
    """
    # =============================
    # CONFIGURE SOLVER OPTIONS
    # =============================
    options = solver_options.copy() if solver_options else {}

    # Update options with defaults only if not already set
    for key, value in DEFAULT_SOLVER_OPTIONS.items():
        options.setdefault(key, value)

    all_states, all_times = [], []

    current_state = sim_oqs.system.psi_ini
    actual_times = sim_oqs.times_local
    pulse_seq = sim_oqs.laser
    decay_ops_list = sim_oqs.decay_channels

    # Find pulse regions and split time array
    pulse_regions = identify_non_zero_pulse_regions(actual_times, pulse_seq)
    split_times = split_by_active_regions(actual_times, pulse_regions)

    for i, curr_times in enumerate(split_times):
        # Extend curr_times by one point if not the last segment
        if i < len(split_times) - 1:
            next_times = split_times[i + 1]
            if len(next_times) > 0:
                curr_times = np.append(curr_times, next_times[0])

        # Find the indices in the original times array for this split
        start_idx = np.abs(actual_times - curr_times[0]).argmin()
        has_pulse = pulse_regions[start_idx]
        if has_pulse:
            EVO_obj = sim_oqs.Evo_obj_int
        else:
            EVO_obj = sim_oqs.Evo_obj_free
        # Execute evolution for this time segment
        result = _execute_single_evolution_segment(
            sim_oqs.simulation_config.ODE_Solver,
            EVO_obj,
            decay_ops_list,
            current_state,
            curr_times,
            options,
        )

        # Store results
        if hasattr(result, "states") and result.states:
            if i < len(split_times) - 1:
                all_states.extend(result.states[:-1])
                all_times.extend(result.times[:-1])
            else:
                all_states.extend(result.states)
                all_times.extend(result.times)
            current_state = result.states[-1]
        elif hasattr(result, "final_state"):
            current_state = result.final_state
        else:
            raise RuntimeError(
                "No valid state found in result for next evolution step."
            )

    # Create combined result object
    result.states = all_states if all_states else []
    result.times = np.array(all_times) if all_times else []

    return result


def _execute_single_evolution_segment(
    ODE_Solver: str,
    EVO_obj: Union[Qobj, QobjEvo],
    decay_ops_list: list,
    current_state: Qobj,
    times_: np.ndarray,
    options: dict,
) -> Result:
    """Execute evolution for a single time segment."""
    if ODE_Solver == "BR":
        return brmesolve(
            EVO_obj,
            current_state,
            times_,
            a_ops=decay_ops_list,
            options=options,
        )
    else:
        return mesolve(
            EVO_obj,
            current_state,
            times_,
            c_ops=decay_ops_list,
            options=options,
        )


def check_the_solver(sim_oqs: SimulationModuleOQS) -> tuple[Result, float]:
    """
    Checks the solver within the compute_pulse_evolution function
    with the provided psi_ini, times, and system.

    Parameters:
        sim_oqs (SimulationModuleOQS):  object containing all relevant parameters

    Returns:
    tuple of:
        result (Result): The Qotip result object.
        time_cut (float): The time after which the checks failed, or np.inf if all checks passed.
    """
    print(f"Checking '{sim_oqs.simulation_config.ODE_Solver}' solver ", flush=True)
    copy_sim_oqs = deepcopy(sim_oqs)
    t_max = 2 * copy_sim_oqs.simulation_config.t_max
    dt = 10 * copy_sim_oqs.simulation_config.dt
    t0 = -2 * copy_sim_oqs.laser.pulse_fwhms[0]
    times = np.linspace(t0, t_max, int((t_max - t0) / dt) + 1)
    copy_sim_oqs.times_local = times

    # =============================
    # INPUT VALIDATION
    # =============================
    if not isinstance(copy_sim_oqs.system.psi_ini, Qobj):
        raise TypeError("psi_ini must be a Qobj")
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be a numpy.ndarray")
    if not isinstance(copy_sim_oqs.observable_ops, list) or not all(
        isinstance(op, Qobj) for op in copy_sim_oqs.observable_ops
    ):
        raise TypeError("system.observable_ops must be a list of Qobj")
    if len(times) < 2:
        raise ValueError("times must have at least two elements")

    result = compute_pulse_evolution(copy_sim_oqs, **{"store_states": True})
    states = result.states
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
    time_cut = np.inf  # time after which the checks failed
    # Apply RWA phase factors if needed
    if getattr(copy_sim_oqs.simulation_config, "RWA_SL", False):
        N_atoms = copy_sim_oqs.system.N_atoms
        omega_laser = copy_sim_oqs.laser.omega_laser
        states = apply_RWA_phase_factors(states, times, N_atoms, omega_laser)
    for index, state in enumerate(states):
        time = times[index]
        if not state.isherm:
            strg += f"Density matrix is not Hermitian after t = {time}.\n"
            print(state, flush=True)
        eigvals = state.eigenenergies()
        if not np.all(eigvals >= NEGATIVE_EIGVAL_THRESHOLD):
            strg += f"Density matrix is not positive semidefinite after t = {time}: The lowest eigenvalue is {eigvals.min()}.\n"
            time_cut = time
        if not np.isclose(state.tr(), 1.0, atol=TRACE_TOLERANCE):
            strg += f"Density matrix is not trace-preserving after t = {time}: The trace is {state.tr()}.\n"
            time_cut = time
        if strg:
            strg += "Adjust your parameters!"
            print(strg, flush=True)
            break
    else:
        print(
            "Checks passed. DM remains Hermitian and positive.",
            flush=True,
        )

    return result, time_cut


def _compute_next_start_point(
    sim_oqs: SimulationModuleOQS,
    **kwargs,
) -> Qobj:
    """
    Compute the final state after a single pulse evolution.

    Parameters
    psi_initial : Qobj
        Initial quantum state before the pulse.
    times : np.ndarray
        Time array for the pulse evolution.
    pulse_specs : list[index: int, t_peak: float, phase: float]

    system : AtomicSystem
        System parameters object.
    Returns
    -------
    Qobj
        Final quantum state after pulse evolution.

    """

    # =============================
    # EVOLUTION COMPUTATION
    # =============================
    evolution_options = {
        "store_final_state": True,
        "store_states": False,  # Only need final state for efficiency
    }
    evolution_options.update(kwargs)  # Allow override of options

    evolution_data = compute_pulse_evolution(sim_oqs=sim_oqs, **evolution_options)

    return evolution_data.final_state


def compute_1d_polarization(
    sim_oqs: SimulationModuleOQS,
    **kwargs,
) -> list[np.complex64]:
    """
    Compute the data for a fixed t_coh and t_wait. AND NOW VARIABLE t_det_max
    """

    if kwargs.get("plot_example_evo", False):

        data = compute_pulse_evolution(sim_oqs=sim_oqs, store_states=True)
        states = data.states
        times = data.times
        Dip_op = sim_oqs.system.Dip_op
        N_atoms = sim_oqs.system.N_atoms
        e_ops = sim_oqs.observable_ops
        RWA_SL = sim_oqs.simulation_config.RWA_SL
        omega_laser = sim_oqs.laser.omega_laser
        datas = get_expect_vals_with_RWA(
            states,
            times,
            N_atoms=N_atoms,
            e_ops=e_ops,
            omega_laser=omega_laser,
            RWA_SL=RWA_SL,
            Dip_op=Dip_op,
        )
        return times, datas, sim_oqs

    # =============================
    # COMPUTE EVOLUTION STATES
    # =============================
    evolution_data = _compute_3_pulse_evolution(sim_oqs)

    # =============================
    # COMPUTE LINEAR SIGNALS
    # =============================
    linear_signals = _compute_3_linear_signals(sim_oqs)

    # =============================
    # EXTRACT AND PROCESS DETECTION DATA
    # =============================
    time_cut_val = kwargs.get("time_cut", np.inf)
    # Ensure time_cut is a proper float, not a Qobj
    if hasattr(time_cut_val, "full"):  # Check if it's a Qobj
        raise TypeError("time_cut must be a float, not a Qobj")
    elif not isinstance(time_cut_val, (int, float)):
        time_cut_val = float(time_cut_val)

    detection_data = _extract_detection_data(
        sim_oqs,
        evolution_data,
        linear_signals,
        time_cut_val,
        # maybe could also pass kwargs for plotting
    )

    # Return based on plotting flag
    if kwargs.get("plot_example_polarization", False):
        return detection_data["plot_polarization_data"]

    return detection_data["nonlinear_signal"]


def _compute_3_pulse_evolution(
    sim_oqs: SimulationModuleOQS,
) -> dict:
    """Compute the n-pulse evolution using segmented approach."""
    copy_sim_oqs = deepcopy(sim_oqs)
    t_coh = sim_oqs.simulation_config.t_coh
    t_wait = sim_oqs.simulation_config.t_wait
    detection_time = t_coh + t_wait
    times = sim_oqs.times_local
    fwhms = sim_oqs.laser.pulse_fwhms

    full_sequence = sim_oqs.laser
    n_pulses = len(full_sequence.pulses)

    if n_pulses < 2:
        raise ValueError("Need at least 2 pulses for segmented evolution")

    # Initialize variables for loop
    current_state = sim_oqs.system.psi_ini
    prev_pulse_start_idx = 0

    # Loop over first n_pulses - 1 segments
    for pulse_idx in range(n_pulses - 1):
        # Calculate pulse start index
        if pulse_idx == 0:
            # First pulse: start at t_coh - fwhm
            pulse_start_idx = np.abs(times - (t_coh - fwhms[pulse_idx + 1])).argmin()
            times_segment = _ensure_valid_times(times[: pulse_start_idx + 1], times)
        elif pulse_idx == 1:
            pulse_start_idx = np.abs(
                times - (detection_time - fwhms[pulse_idx + 1])
            ).argmin()
        else:
            raise ValueError("STILL TODO implement general n-pulse evolution")

        times_segment = times[prev_pulse_start_idx : pulse_start_idx + 1]
        times_segment = _ensure_valid_times(times_segment, times, prev_pulse_start_idx)

        # Update simulation parameters for this segment
        copy_sim_oqs.times_local = times_segment
        copy_sim_oqs.laser = LaserPulseSequence(
            pulses=full_sequence.pulses[: pulse_idx + 1]
        )
        copy_sim_oqs.system.psi_ini = current_state

        # Compute evolution for this segment
        current_state = _compute_next_start_point(sim_oqs=copy_sim_oqs)

        # Update for next iteration
        prev_pulse_start_idx = pulse_start_idx

    # Final segment: evolution with detection (last pulse)
    times_final = _ensure_valid_times(
        times[prev_pulse_start_idx:], times, prev_pulse_start_idx
    )
    copy_sim_oqs.times_local = times_final

    copy_sim_oqs.laser = full_sequence
    copy_sim_oqs.system.psi_ini = current_state

    data_final = compute_pulse_evolution(sim_oqs=copy_sim_oqs, store_states=True)

    detection_length = len(sim_oqs.times_det)
    final_states = data_final.states[-detection_length:]

    return final_states


def _compute_3_linear_signals(
    sim_oqs: SimulationModuleOQS,
) -> dict:
    """Compute all linear signal contributions."""
    laser = sim_oqs.laser
    detection_length = len(sim_oqs.times_det)
    linear_data_states = {}
    copy_sim_oqs = deepcopy(sim_oqs)
    for i, pulse in enumerate(laser):
        single_seq = LaserPulseSequence(pulses=[pulse])
        copy_sim_oqs.laser = single_seq  # Update the laser sequence for each pulse
        data = compute_pulse_evolution(sim_oqs=copy_sim_oqs, store_states=True)
        linear_data_states[f"pulse{i}"] = data.states[-detection_length:]

    return linear_data_states


def _ensure_valid_times(
    times_segment: np.ndarray, full_times: np.ndarray, fallback_idx: int = 0
) -> np.ndarray:
    """
    Ensure time segment is not empty to prevent computation errors.

    Parameters
    ----------
    times_segment : np.ndarray
        The time segment that might be empty
    full_times : np.ndarray
        The complete time array as backup
    fallback_idx : int
        Index to use if times_segment is empty

    Returns
    -------
    np.ndarray
        Valid time array (either original or fallback)
    """
    # Check if the time segment has no elements
    if times_segment.size == 0:
        # First fallback: try to use a single time point at fallback_idx
        if fallback_idx < len(full_times):
            print(
                f"Warning: times_segment is empty. Using fallback time at index {fallback_idx}: {full_times[fallback_idx]}",
                flush=True,
            )
            return np.array([full_times[fallback_idx]])
        print(
            f"Warning: times_segment is empty. Using first two time points from full array: {full_times[:2]}",
            flush=True,
        )
        # Second fallback: use first two time points from full array
        return full_times[:2]

    # Time segment is valid, return as-is
    return times_segment


def _extract_detection_data(
    sim_oqs: SimulationModuleOQS,
    evolution_data: List[Qobj],
    linear_signals: dict[List[Qobj]],
    time_cut: float,
    # maybe could also pass kwargs for plotting
) -> dict:
    """Extract and process detection time data."""
    actual_det_times = sim_oqs.times_det_actual
    # Apply RWA phase factors if needed
    if sim_oqs.simulation_config.RWA_SL:
        N_atoms = sim_oqs.system.N_atoms
        omega_laser = sim_oqs.laser.omega_laser
        evolution_data = apply_RWA_phase_factors(
            evolution_data, actual_det_times, N_atoms, omega_laser
        )
        for key in linear_signals:
            linear_signals[key] = apply_RWA_phase_factors(
                linear_signals[key], actual_det_times, N_atoms, omega_laser
            )

    # Calculate polarizations
    Dip_op = sim_oqs.system.Dip_op
    polarizations = {}
    polarizations_full = complex_polarization(Dip_op, evolution_data)
    for key in linear_signals:
        polarizations[key] = complex_polarization(Dip_op, linear_signals[key])

    # Calculate nonlinear signal - generalized for n pulses
    nonlinear_signal = polarizations_full
    for pulse_key in linear_signals.keys():
        nonlinear_signal -= polarizations[pulse_key]

    detection_times = sim_oqs.times_det
    # padd the data to match the length of detection_times
    nonlinear_signal = nonlinear_signal[actual_det_times < time_cut]
    # print(nonlinear_signal, flush=True)
    if len(nonlinear_signal) < len(detection_times):
        print(
            "The data will be padded with zeros to match the detection times.",
            flush=True,
        )
        zeros_to_add = len(detection_times) - len(nonlinear_signal)
        nonlinear_signal = np.concatenate([nonlinear_signal, np.zeros(zeros_to_add)])

    plot_polarization_data = [polarizations_full]
    for key in linear_signals.keys():
        pol = polarizations[key]
        plot_polarization_data.append(pol)

    return {
        "nonlinear_signal": nonlinear_signal,
        "plot_polarization_data": tuple(plot_polarization_data),
    }


# ##########################
# parallel processing 1d and 2d data
# ##########################
def _process_single_1d_combination(
    sim_oqs: SimulationModuleOQS,
    new_freqs: np.ndarray,
    phi1: float,
    phi2: float,
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
    t_coh : float
        Coherence time.
    t_wait : float
        Waiting time.

    system : AtomicSystem
        System parameters (already contains the correct omega_A_cm).
    kwargs : dict
        Additional arguments.

    Returns
    -------
    tuple or None
        (t_det_vals, data) for the computed 1D polarization data, or None if failed.
    """
    try:
        # CRITICAL FIX: Make a deep copy to avoid modifying the shared object
        local_sim_oqs = deepcopy(sim_oqs)

        # DEBUG: Check phases before and after update
        # print(f"Before update: {local_sim_oqs.laser.pulse_phases}", flush=True)

        local_sim_oqs.laser.update_phases(
            phases=[phi1, phi2, DETECTION_PHASE]
        )  # Update the laser phases in the local copy

        # print(f"After update: {local_sim_oqs.laser.pulse_phases}", flush=True)
        # TODO THE problem likely happens because the frequencies are not (really) updated in the local copy

        local_sim_oqs.system.freqs_cm = (
            new_freqs  # Update frequencies in the local copy
        )

        data = compute_1d_polarization(
            sim_oqs=local_sim_oqs,
            **kwargs,
        )

        return data

    except Exception as e:
        print(f"Error in _process_single_1d_combination: {str(e)}", flush=True)
        return None


def parallel_compute_1d_E_with_inhomogenity(
    sim_oqs: SimulationModuleOQS,
    **kwargs: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D COMPLEX polarization with frequency loop and phase cycling for IFT processing.
    Parallelizes over all frequency and phase combinations, averages, then performs IFT.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments for compute_1d_polarization.

    Returns
    -------
    tuple
        (t_det_vals, photon_echo_signal) where signal is averaged and IFT-processed.
    """
    # Configure phase cycling
    n_phases = sim_oqs.simulation_config.n_phases
    n_freqs = sim_oqs.simulation_config.n_freqs
    max_workers = sim_oqs.simulation_config.max_workers
    phases = PHASE_CYCLING_PHASES[:n_phases]  # Use predefined phases
    if n_phases != 4:
        logger.warning(
            f"Phase cycling with {n_phases} phases may not be optimal for IFT"
        )

    # Sample frequency offsets for inhomogeneous broadening
    Delta_cm = sim_oqs.system.Delta_cm
    freqs_cm = sim_oqs.system.freqs_cm

    # Each row = one realization, each column = atom index
    # Shape: (n_freqs, N_atoms)
    all_freq_sets = np.stack(
        [sample_from_gaussian(n_freqs, Delta_cm, freq) for freq in freqs_cm], axis=1
    )
    # print(f"Using frequency samples ={all_freq_sets}", flush=True)

    # Prepare all jobs: one per (omega_idx, phi1_idx, phi2_idx)
    combinations = []
    for omega_idx in range(n_freqs):
        new_freqs = all_freq_sets[omega_idx]
        for phi1_idx, phi1 in enumerate(phases):
            for phi2_idx, phi2 in enumerate(phases):
                combinations.append(
                    (omega_idx, phi1_idx, phi2_idx, new_freqs, phi1, phi2)
                )

    # Execute all jobs in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_1d_combination,
                sim_oqs=sim_oqs,
                new_freqs=new_freqs,
                phi1=phi1,
                phi2=phi2,
                **kwargs,
            ): (omega_idx, phi1_idx, phi2_idx)
            for (omega_idx, phi1_idx, phi2_idx, new_freqs, phi1, phi2) in combinations
        }

        for future in as_completed(futures):
            omega_idx, phi1_idx, phi2_idx = futures[future]
            try:
                data = future.result()
                results[(omega_idx, phi1_idx, phi2_idx)] = data
            except Exception as exc:
                logger.error(
                    f"Combination ({omega_idx},{phi1_idx},{phi2_idx}) failed: {exc}"
                )
                # You might want to handle this more gracefully, e.g., using default values

    # Fill 3D result array
    results_cube = np.zeros((n_freqs, n_phases, n_phases), dtype=object)
    for (omega_idx, phi1_idx, phi2_idx), data in results.items():
        if data is None:
            continue
        results_cube[omega_idx, phi1_idx, phi2_idx] = data * 1j  # E ~ iP
    # Average over frequencies to get 2D result array before IFT or phase-average
    results_matrix_avg = np.mean(results_cube, axis=0)

    print("BEFORE IFT", results_matrix_avg, flush=True)
    # Final IFT extraction for the specified component
    ift_component = sim_oqs.simulation_config.IFT_component
    photon_echo_signal = extract_ift_signal_component(
        results_matrix=results_matrix_avg, phases=phases, component=ift_component
    )
    print("AFTER IFT", photon_echo_signal, flush=True)

    return photon_echo_signal


# ##########################
# Helper functions for IFT processing
# ##########################
def extract_ift_signal_component(
    results_matrix: np.ndarray, phases: list, component: list[int, int, int]
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
        Each element can be a 1D or 2D array.
    phases : list
        List of phase values used (typically [0, π/2, π, 3π/2])
    component : list[l: int, m: int, int]
        Coefficients for the phases in the IFT

    Returns
    -------
    np.ndarray
        Extracted signal component after IFT (same shape as input data elements).

    Examples
    --------
    >>> signal = extract_ift_signal_component(results_matrix, phases, [-1, 1, 0])
    >>> print(signal)
    """
    l, m, n = component
    n_phases = len(phases)

    # Find first non-None result to determine output shape
    first_valid_result = next(
        (
            results_matrix[i, j]
            for i in range(n_phases)
            for j in range(n_phases)
            if results_matrix[i, j] is not None
        ),
        None,
    )

    if first_valid_result is None:
        print("No valid results found in the results matrix.", flush=True)
        return None

    signal = np.zeros_like(first_valid_result, dtype=np.complex64)

    for phi1_idx, phi_1 in enumerate(phases):
        for phi2_idx, phi_2 in enumerate(phases):
            if results_matrix[phi1_idx, phi2_idx] is not None:
                phase_factor = np.exp(
                    -1j * (l * phi_1 + m * phi_2 + n * DETECTION_PHASE)
                )
                print(f"{phi1_idx},{phi2_idx} phase_factor={phase_factor}")
                signal += results_matrix[phi1_idx, phi2_idx] * phase_factor

    signal /= n_phases * n_phases

    return signal


# =============================
# POLARIZATION CALCULATIONS
# =============================
def complex_polarization(
    dip_op: Qobj, state: Union[Qobj, List[Qobj]]
) -> Union[complex, np.ndarray]:
    """
    Calculate the complex polarization for state(s) using the dipole operator.
    The polarization is defined as one part of the expectation value of the dipole operator
    with the given quantum state(s) or density matrix(es).

    Parameters
    ----------
    dip_op : Qobj
        Dipole operator for the system
    state : Union[Qobj, List[Qobj]]
        A single quantum state/density matrix or list of states.

    Returns
    -------
    Union[complex, np.ndarray]
        Complex polarization value(s). Returns complex for single state,
        complex array for multiple states.

    Raises
    ------
    TypeError
        If state is not a Qobj or list of Qobj.

    Examples
    --------
    >>> pol = complex_polarization(dip_op, rho)  # Single density matrix
    >>> pols = complex_polarization(dip_op, [rho1, rho2])  # Multiple states
    """
    if isinstance(state, Qobj):
        return _single_qobj_polarization(dip_op, state)

    if isinstance(state, list):
        return np.array(
            [_single_qobj_polarization(dip_op, s) for s in state],
            dtype=np.complex64,  # Use higher precision
        )

    raise TypeError(f"State must be a Qobj or list of Qobj, got {type(state)}")


def _single_qobj_polarization(dip_op: Qobj, state: Qobj) -> complex:
    """
    Calculate polarization for a single quantum state or density matrix.

    Parameters
    ----------
    dip_op : Qobj
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
    if not (state.isket or state.isoper):
        raise TypeError("State must be a ket or density matrix")

    # General approach that works for any system size
    polarization = 0j

    if state.isket:
        # Convert ket to density matrix for consistent handling
        state = ket2dm(state)

    # For any system size, calculate polarization as sum of off-diagonal elements
    # This works for any number of atoms without needing special cases TODO BUT NOT IN SINGLE EXCITATION SUBSPACE?
    for i in range(dip_op.shape[0]):
        for j in range(i):
            if i != j and abs(dip_op[i, j]) != 0:
                polarization += dip_op[i, j] * state[j, i]

    return polarization
