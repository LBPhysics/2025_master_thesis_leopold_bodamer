# -*- coding: utf-8 -*-

# =============================
# STANDARD LIBRARY IMPORTS
# =============================
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from qspectro2d.core.simulation_class import SimClassOQS
from qspectro2d.core.laser_system.laser_class import (
    LaserPulseSystem,
    identify_non_zero_pulse_regions,
    split_by_active_regions,
)
from qspectro2d.spectroscopy.inhomogenity import sample_from_gaussian
from qspectro2d.core.functions_with_rwa import (
    apply_RWA_phase_factors,
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
    sim_oqs: SimClassOQS,
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
    ops : SimClassOQS contains simulation configuration, system, laser parameters and much more.
    pulse_seq : LaserPulseSystem
        LaserPulseSystem object defining the pulse sequence.
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

    return _execute_segmented_evolution(sim_oqs, options)


def _execute_segmented_evolution(
    sim_oqs: SimClassOQS,
    options: dict,
) -> Result:
    """
    Execute evolution split by pulse regions.

    Parameters
    ----------
    sim_oqs : SimClassOQS
        Simulation class object containing system, laser, and time configurations.
    options : dict
        Solver options for the evolution.

    Returns
    -------
    Result
        QuTiP Result object containing combined evolution data.

    Examples
    --------
    >>> result = _execute_segmented_evolution(sim_oqs, options)
    >>> print(result.states)
    """
    all_states, all_times = [], []

    current_state = sim_oqs.system.psi_ini
    # Find pulse regions and split time array
    pulse_regions = identify_non_zero_pulse_regions(sim_oqs.times, sim_oqs.laser)
    split_times = split_by_active_regions(sim_oqs.times, pulse_regions)

    for i, curr_times in enumerate(split_times):
        # Extend curr_times by one point if not the last segment
        if i < len(split_times) - 1:
            next_times = split_times[i + 1]
            if len(next_times) > 0:
                curr_times = np.append(curr_times, next_times[0])

        # Find the indices in the original times array for this split
        start_idx = np.abs(sim_oqs.times - curr_times[0]).argmin()
        has_pulse = pulse_regions[start_idx]
        decay_ops_list = sim_oqs.decay_channels
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


def check_the_solver(sim_oqs: SimClassOQS) -> tuple[Result, float]:
    """
    Checks the solver within the compute_pulse_evolution function
    with the provided psi_ini, times, and system.

    Parameters:
        sim_oqs (SimClassOQS):  object containing all relevant parameters

    Returns:
    tuple of:
        result (Result): The Qotip result object.
        time_cut (float): The time after which the checks failed, or np.inf if all checks passed.
    """
    t_max = 2 * sim_oqs.t_max
    dt = 10 * sim_oqs.dt
    t0 = -sim_oqs.t0
    times = np.linspace(t0, t_max, int((t_max - t0) / dt) + 1)

    print(f"Checking '{sim_oqs.simulation_config.ODE_Solver}' solver ", flush=True)

    # =============================
    # INPUT VALIDATION
    # =============================
    if not isinstance(sim_oqs.system.psi_ini, Qobj):
        raise TypeError("psi_ini must be a Qobj")
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be a numpy.ndarray")
    if not isinstance(sim_oqs.observable_ops, list) or not all(
        isinstance(op, Qobj) for op in sim_oqs.observable_ops
    ):
        raise TypeError("system.observable_ops must be a list of Qobj")
    if len(times) < 2:
        raise ValueError("times must have at least two elements")

    result = compute_pulse_evolution(sim_oqs, **{"store_states": True})
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
    if getattr(sim_oqs.simulation_config, "RWA_SL", False):
        N_atoms = sim_oqs.system.N_atoms
        omega_laser = sim_oqs.laser.omega_laser
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
    sim_oqs: SimClassOQS,
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
    sim_oqs: SimClassOQS,
    **kwargs,
) -> np.ndarray:
    """
    Compute the data for a fixed tau_coh and T_wait. AND NOW VARIABLE t_det_max
    """
    time_cut = kwargs.get("time_cut", np.inf)

    # =============================
    # COMPUTE EVOLUTION STATES
    # =============================
    evolution_data = _compute_three_pulse_evolution(sim_oqs)

    # =============================
    # COMPUTE LINEAR SIGNALS
    # =============================
    linear_signals = _compute_linear_signals(sim_oqs)

    # =============================
    # EXTRACT AND PROCESS DETECTION DATA
    # =============================
    detection_data = _extract_detection_data(
        sim_oqs,
        evolution_data,
        linear_signals,
        time_cut,
        # maybe could also pass kwargs for plotting
    )

    # Return based on plotting flag
    if kwargs.get("plot_example_polarization", False):
        return detection_data["plot_data"]

    return detection_data["nonlinear_signal"]


def _compute_three_pulse_evolution(
    sim_oqs: SimClassOQS,
) -> dict:
    """Compute the three-pulse evolution using segmented approach."""
    # Segment 1: First pulse evolution
    tau_coh = sim_oqs.simulation_config.tau_coh
    t_wait = sim_oqs.simulation_config.t_wait
    times = sim_oqs.times
    fwhms = sim_oqs.laser.pulse_fwhms

    full_sequence = sim_oqs.laser

    pulse1_start_idx = np.abs(times - (tau_coh - fwhms[0])).argmin()
    times_0 = _ensure_valid_times(times[: pulse1_start_idx + 1], times)

    sim_oqs.times = times_0  # Update times in sim_class_oqs for the first segment
    sim_oqs.laser = LaserPulseSystem(pulses=[full_sequence.pulses[0]])
    rho_1 = _compute_next_start_point(
        sim_oqs=sim_oqs,
    )

    # Segment 2: Second pulse evolution
    pulse2_start_idx = np.abs(times - (tau_coh + t_wait - fwhms[1])).argmin()
    times_1 = times[pulse1_start_idx : pulse2_start_idx + 1]
    times_1 = _ensure_valid_times(times_1, times, pulse1_start_idx)

    sim_oqs.times = times_1  # Update times in sim_class_oqs for the first segment
    sim_oqs.laser = LaserPulseSystem(pulses=[full_sequence.pulses[0:2]])
    sim_oqs.system.psi_ini = rho_1  # Set initial state for next segment
    rho_2 = _compute_next_start_point(
        sim_oqs=sim_oqs,
    )

    # Segment 3: Final evolution with detection
    times_2 = _ensure_valid_times(times[pulse2_start_idx:], times, pulse2_start_idx)
    sim_oqs.times = times_2  # Update times in sim_class_oqs for the first segment
    sim_oqs.laser = LaserPulseSystem(pulses=[full_sequence])
    sim_oqs.system.psi_ini = rho_2  # Set initial state for next segment
    data_final = compute_pulse_evolution(sim_oqs=sim_oqs, store_states=True)

    return {
        "final_data": data_final,
        "times_2": times_2,
        "detection_start_idx": pulse2_start_idx,
    }


def _compute_linear_signals(
    sim_oqs: SimClassOQS,
) -> dict:
    """Compute all linear signal contributions."""
    times = sim_oqs.times
    system = sim_oqs.system
    laser = sim_oqs.laser
    detection_time: float = sim_oqs.simulation_config.detection_time

    linear_data = {}
    detection_idx = np.abs(times - detection_time).argmin()

    for i, pulse in enumerate(laser):
        single_seq = LaserPulseSystem(pulses=[pulse])
        data = compute_pulse_evolution(
            system.psi_ini, times, single_seq, system=system, store_states=True
        )
        linear_data[f"pulse{i}"] = data.states[detection_idx:]

    return linear_data


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
            return np.array([full_times[fallback_idx]])
        # Second fallback: use first two time points from full array
        return full_times[:2]

    # Time segment is valid, return as-is
    return times_segment


def _calculate_all_polarizations(
    states_full: list,
    linear_signals: dict,
    times: np.ndarray,
    time_cut: float,
    Dip_op: Qobj,
) -> dict:
    """Calculate polarizations for all signal components."""
    valid_indices = [i for i, t in enumerate(times) if t < time_cut]

    if not valid_indices:
        return {
            key: np.zeros(len(times), dtype=np.complex64)
            for key in ["full", "pulse0", "pulse1", "pulse2"]
        }

    # Extract valid states
    valid_states = {
        "full": [states_full[i] for i in valid_indices],
        "pulse0": [linear_signals["pulse0"][i] for i in valid_indices],
        "pulse1": [linear_signals["pulse1"][i] for i in valid_indices],
        "pulse2": [linear_signals["pulse2"][i] for i in valid_indices],
    }

    # Calculate polarizations
    polarizations = {}
    for key, states in valid_states.items():
        P_array = np.zeros(len(times), dtype=np.complex64)
        P_values = complex_polarization(Dip_op, states)

        # Handle scalar vs array results
        if np.isscalar(P_values):
            P_array[valid_indices[0]] = P_values
        else:
            for idx, orig_idx in enumerate(valid_indices):
                P_array[orig_idx] = P_values[idx]

        polarizations[key] = P_array

    return polarizations


def _extract_detection_data(
    sim_class_oqs: SimClassOQS,
    evolution_data: dict,
    linear_signals: dict,
    time_cut: float,
    # maybe could also pass kwargs for plotting
) -> dict:
    """Extract and process detection time data."""
    final_data = evolution_data["final_data"]
    detection_time = (
        sim_class_oqs.simulation_config.tau_coh + sim_class_oqs.simulation_config.t_wait
    )
    detection_start_idx = np.abs(final_data.times - detection_time).argmin()
    actual_det_times = final_data.times[detection_start_idx:]
    states_full = final_data.states[detection_start_idx:]

    # Apply RWA phase factors if needed
    if sim_class_oqs.simulation_config.RWA_SL:
        N_atoms = sim_class_oqs.system.N_atoms
        omega_laser = sim_class_oqs.laser.omega_laser
        states_full = apply_RWA_phase_factors(
            states_full, actual_det_times, N_atoms, omega_laser
        )
        for key in linear_signals:
            linear_signals[key] = apply_RWA_phase_factors(
                linear_signals[key], actual_det_times, N_atoms, omega_laser
            )

    # Calculate polarizations
    Dip_op = sim_class_oqs.system.Dip_op
    polarizations = _calculate_all_polarizations(
        states_full, linear_signals, actual_det_times, time_cut, Dip_op
    )

    # Calculate nonlinear signal
    nonlinear_signal = (
        polarizations["full"]
        - polarizations["pulse0"]
        - polarizations["pulse1"]
        - polarizations["pulse2"]
    )

    # if len(actual_det_times) != len(t_det_values):
    #     print(
    #         f"Warning: interpolated t_det length {len(actual_det_times)} ≠ {len(t_det_values)}"
    #     )
    # cap the nonlinear_signal to the sim_class_oqs.times_det length!
    t_det_values = np.linspace(
        actual_det_times[0], actual_det_times[-1], len(sim_class_oqs.times_det)
    )
    # 1D linear interpolation to match the canonical grid
    data_capped = np.interp(
        t_det_values, actual_det_times - actual_det_times[0], nonlinear_signal
    )

    return {
        "nonlinear_signal": data_capped,
        "plot_data": (
            actual_det_times - actual_det_times[0],
            polarizations["full"],
            polarizations["pulse0"],
            polarizations["pulse1"],
            polarizations["pulse2"],
        ),
    }


# ##########################
# parallel processing 1d and 2d data
# ##########################
def _process_single_1d_combination(
    sim_class_oqs: SimClassOQS,
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
    tau_coh : float
        Coherence time.
    T_wait : float
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
        sim_class_oqs.laser.update_first_two_pulse_phases(
            phase1=phi1, phase2=phi2
        )  # Update the laser phases in the simulation class
        # Compute the 1D polarization for this specific phase combination
        sim_class_oqs.system.freqs_cm = new_freqs  # Update frequencies in the system
        data = compute_1d_polarization(
            sim_class_oqs=sim_class_oqs,
            **kwargs,
        )

        return data

    except Exception as e:
        print(f"Error in _process_single_1d_combination: {str(e)}", flush=True)
        return None


def parallel_compute_1d_E_with_inhomogenity(
    sim_class_oqs: SimClassOQS,
    ift_component: tuple = (
        -1,
        1,
        0,
    ),  # could also go into sim_class_oqs.simulation_config
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
    n_phases = sim_class_oqs.simulation_config.n_phases
    n_freqs = sim_class_oqs.simulation_config.n_freqs
    max_workers = sim_class_oqs.simulation_config.max_workers
    phases = PHASE_CYCLING_PHASES[:n_phases]  # Use predefined phases
    if n_phases != 4:
        logger.warning(
            f"Phase cycling with {n_phases} phases may not be optimal for IFT"
        )

    # Sample frequency offsets for inhomogeneous broadening
    Delta_cm = sim_class_oqs.system.Delta_cm
    freqs_cm = sim_class_oqs.system.freqs_cm

    # Each row = one realization, each column = atom index
    # Shape: (n_freqs, N_atoms)
    all_freq_sets = np.stack(
        [sample_from_gaussian(n_freqs, Delta_cm, freq) for freq in freqs_cm], axis=1
    )
    print(f"Using frequency samples ={all_freq_sets}", flush=True)

    # Prepare all jobs: one per (omega_idx, phi1_idx, phi2_idx)
    combinations = []
    for omega_idx in range(n_freqs):
        new_freqs = all_freq_sets[omega_idx]
        sim_class_oqs.system.update_freqs_cm(new_freqs)

    for phi1_idx, phi1 in enumerate(phases):
        for phi2_idx, phi2 in enumerate(phases):
            combinations.append((omega_idx, phi1_idx, phi2_idx, new_freqs, phi1, phi2))

    # Execute all jobs in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_1d_combination,
                sim_class_oqs=sim_class_oqs,
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
        results_cube[omega_idx, phi1_idx, phi2_idx] = data * 1j  # E ~ iP

    # Average over frequencies to get 2D result array before IFT or phase-average
    results_matrix_avg = np.mean(results_cube, axis=0)

    apply_ift = sim_class_oqs.simulation_config.apply_ift
    if apply_ift:
        # Final IFT extraction for the specified component
        photon_echo_signal = extract_ift_signal_component(
            results_matrix=results_matrix_avg, phases=phases, component=ift_component
        )
    else:
        # Phase-averaged raw signal E = i*P
        phase_signals = []
        for phi1_idx in range(n_phases):
            for phi2_idx in range(n_phases):
                if results_matrix_avg[phi1_idx, phi2_idx] is not None:
                    phase_signals.append(results_matrix_avg[phi1_idx, phi2_idx])
        if phase_signals:
            photon_echo_signal = np.mean(np.array(phase_signals), axis=0)
        else:
            photon_echo_signal = (
                np.zeros_like(results_matrix_avg[0, 0])
                if results_matrix_avg[0, 0] is not None
                else np.array([])
            )
            print(
                "Warning: No valid phase signals found, returning empty signal.",
                flush=True,
            )

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
        return None

    signal = np.zeros_like(first_valid_result, dtype=np.complex64)

    for phi1_idx, phi_1 in enumerate(phases):
        for phi2_idx, phi_2 in enumerate(phases):
            if results_matrix[phi1_idx, phi2_idx] is not None:
                phase_factor = np.exp(
                    -1j * (l * phi_1 + m * phi_2 + n * DETECTION_PHASE)
                )
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
            dtype=np.complex128,  # Use higher precision
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
