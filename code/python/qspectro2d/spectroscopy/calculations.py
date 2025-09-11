# -*- coding: utf-8 -*-
"""
Quantum Spectroscopy Calculations Module

This module contains functions for computing quantum evolution, polarization signals,
and 2D spectroscopy data using QuTiP. It handles multi-pulse sequences, phase cycling,
and parallel processing for efficient computation.

Main Functions:
- compute_pulse_evolution: Core evolution computation
- compute_1d_polarization: 1D spectroscopy calculations
- parallel_compute_1d_E_with_inhomogenity: Parallel 1D with inhomogeneity
- check_the_solver: Solver validation and diagnostics
"""

# STANDARD LIBRARY IMPORTS
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Union, Tuple
import numpy as np
from qutip import Qobj, QobjEvo, Result, mesolve, brmesolve

# LOCAL IMPORTS
from qspectro2d.core.simulation import SimulationModuleOQS
from qspectro2d.core.laser_system.laser_class import (
    LaserPulseSequence,
    identify_non_zero_pulse_regions,
    split_by_active_regions,
)
from qspectro2d.spectroscopy.inhomogenity import sample_from_gaussian
from qspectro2d.utils import apply_RWA_phase_factors, get_expect_vals_with_RWA
from qspectro2d.spectroscopy.polarization import complex_polarization
from project_config.logging_setup import get_logger


# LOGGING CONFIGURATION
logger = get_logger(__name__, level=30)  # 30 = logging.WARNING

# TODO put into config module defaults_params.py


def _validate_simulation_input(sim_oqs: SimulationModuleOQS) -> None:
    """Validate simulation input parameters."""
    if not isinstance(sim_oqs.system.psi_ini, Qobj):
        raise TypeError("psi_ini must be a Qobj")
    if not isinstance(sim_oqs.times_local, np.ndarray):
        raise TypeError("times_local must be a numpy.ndarray")
    if not isinstance(sim_oqs.observable_ops, list) or not all(
        isinstance(op, Qobj) for op in sim_oqs.observable_ops
    ):
        raise TypeError("observable_ops must be a list of Qobj")
    if len(sim_oqs.times_local) < 2:
        raise ValueError("times_local must have at least two elements")


def _log_system_diagnostics(sim_oqs: SimulationModuleOQS) -> None:
    """Log diagnostic information about the quantum system."""
    logger.info("=== SYSTEM DIAGNOSTICS ===")
    psi_ini = sim_oqs.system.psi_ini
    logger.info(f"Initial state type: {type(psi_ini)}")
    logger.info(f"Initial state shape: {psi_ini.shape}")
    logger.info(f"Initial state is Hermitian: {psi_ini.isherm}")
    logger.info(f"Initial state trace: {psi_ini.tr():.6f}")

    if psi_ini.type == "oper":  # density matrix
        ini_eigvals = psi_ini.eigenenergies()
        logger.info(
            f"Initial eigenvalues range: [{ini_eigvals.min():.6f}, {ini_eigvals.max():.6f}]"
        )
        logger.info(f"Initial min eigenvalue: {ini_eigvals.min():.10f}")

    # System Hamiltonian diagnostics
    try:
        if hasattr(sim_oqs, "evo_obj_free") and sim_oqs.evo_obj_free is not None:
            H_free = sim_oqs.evo_obj_free
            if hasattr(H_free, "dims"):
                logger.info(f"Free Hamiltonian dims: {H_free.dims}")
            logger.info(f"Free Hamiltonian type: {type(H_free)}")

        if hasattr(sim_oqs, "decay_channels") and sim_oqs.decay_channels:
            logger.info(f"Number of decay channels: {len(sim_oqs.decay_channels)}")
    except Exception as e:
        logger.warning(f"Could not analyze Hamiltonian: {e}")


def _check_density_matrix_properties(
    states: List[Qobj], times: np.ndarray
) -> tuple[List[str], float]:
    """
    Check density matrix properties for numerical stability.

    Returns:
        tuple: (error_messages, time_cut)
    """
    error_messages = []
    time_cut = np.inf
    from qspectro2d.config.default_simulation_params import (
        NEGATIVE_EIGVAL_THRESHOLD,
        TRACE_TOLERANCE,
    )

    check_interval = max(1, len(states) // 10)  # Check every 10% of states

    for index, state in enumerate(states):
        time = times[index]

        # Sample state analysis
        if index % check_interval == 0 or index < 5:
            logger.info(
                f"State {index} (t={time:.3f}): trace={state.tr():.6f}, Hermitian={state.isherm}"
            )

        # Check Hermiticity
        if not state.isherm:
            error_messages.append(f"Density matrix is not Hermitian after t = {time}")
            logger.error(f"Non-Hermitian density matrix at t = {time}")
            logger.error(f"  State details: trace={state.tr():.6f}, shape={state.shape}")

        # Check positive semidefiniteness
        eigvals = state.eigenenergies()
        min_eigval = eigvals.min()

        if not np.all(eigvals >= NEGATIVE_EIGVAL_THRESHOLD):
            error_messages.append(
                f"Density matrix is not positive semidefinite after t = {time}: "
                f"The lowest eigenvalue is {min_eigval}"
            )
            logger.error(f"NEGATIVE EIGENVALUE DETECTED:")
            logger.error(f"  Time: {time:.6f}")
            logger.error(f"  Min eigenvalue: {min_eigval:.12f}")
            logger.error(f"  Threshold: {NEGATIVE_EIGVAL_THRESHOLD}")
            logger.error(f"  All eigenvalues: {eigvals[:5]}...")
            logger.error(f"  State trace: {state.tr():.10f}")
            logger.error(f"  State index: {index}/{len(states)}")

            if index > 0:
                prev_state = states[index - 1]
                prev_eigvals = prev_state.eigenenergies()
                logger.error(f"  Previous state min eigval: {prev_eigvals.min():.12f}")
                logger.error(f"  Eigenvalue change: {min_eigval - prev_eigvals.min():.12f}")

            time_cut = time

        # Check trace preservation
        trace_val = state.tr()
        if not np.isclose(trace_val, 1.0, atol=TRACE_TOLERANCE):
            error_messages.append(
                f"Density matrix is not trace-preserving after t = {time}: "
                f"The trace is {trace_val}"
            )
            logger.error(f"TRACE VIOLATION:")
            logger.error(f"  Time: {time:.6f}")
            logger.error(f"  Trace: {trace_val:.10f}")
            logger.error(f"  Deviation from 1: {abs(trace_val - 1.0):.10f}")
            logger.error(f"  Tolerance: {TRACE_TOLERANCE}")

            time_cut = min(time_cut, time)

        # Break on first error for detailed analysis
        if error_messages:
            logger.error("=== FIRST ERROR ANALYSIS ===")
            logger.error(f"Stopping analysis at first error (state {index}, t={time:.6f})")
            logger.error("Density matrix validation failed: " + "; ".join(error_messages))
            break

    return error_messages, time_cut


def check_the_solver(sim_oqs: SimulationModuleOQS) -> tuple[Result, float]:
    """
    Validate the quantum solver by running a test evolution and checking density matrix properties.

    This function performs a comprehensive validation of the solver by:
    1. Running a test evolution with extended time range
    2. Checking density matrix properties (Hermiticity, trace preservation, positive semidefiniteness)
    3. Applying RWA phase factors if needed
    4. Logging detailed diagnostics throughout the process

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Simulation object containing system parameters, laser pulses, and configuration.
        A deep copy is made internally to avoid modifying the original object.

    Returns
    -------
    tuple[Result, float]
        result : Result
            QuTiP Result object from the test evolution.
        time_cut : float
            Time after which numerical instabilities were detected, or np.inf if all checks passed.

    Notes
    -----
    The function uses extended time parameters (2x t_max, 10x dt) to stress-test the solver.
    It checks for common numerical issues in quantum simulations:
    - Non-Hermitian density matrices
    - Negative eigenvalues (non-physical states)
    - Trace deviation from 1.0

    If RWA (Rotating Wave Approximation) is enabled, phase factors are applied to the states
    before validation to account for the rotating frame transformation.
    """
    logger.info(f"Checking '{sim_oqs.simulation_config.ode_solver}' solver")
    copy_sim_oqs = deepcopy(sim_oqs)
    t_max = 2 * copy_sim_oqs.simulation_config.t_max
    dt = 10 * copy_sim_oqs.simulation_config.dt
    t0 = -2 * copy_sim_oqs.laser.pulse_fwhms[0]
    times = np.linspace(t0, t_max, int((t_max - t0) / dt) + 1)
    copy_sim_oqs.times_local = times

    # DETAILED SYSTEM DIAGNOSTICS

    logger.info(f"=== SOLVER DIAGNOSTICS ===")
    logger.info(f"Solver: {copy_sim_oqs.simulation_config.ode_solver}")
    logger.info(f"Time range: t0={t0:.3f}, t_max={t_max:.3f}, dt={dt:.6f}")
    logger.info(f"Number of time points: {len(times)}")
    logger.info(f"RWA enabled: {getattr(copy_sim_oqs.simulation_config, 'rwa_sl', False)}")

    _log_system_diagnostics(copy_sim_oqs)

    # INPUT VALIDATION
    _validate_simulation_input(copy_sim_oqs)

    result = compute_pulse_evolution(copy_sim_oqs, **{"store_states": True})
    states = result.states

    # CHECK THE RESULT
    if not isinstance(result, Result):
        raise TypeError("Result must be a Result object")
    if list(result.times) != list(times):
        raise ValueError("Result times do not match input times")
    if len(result.states) != len(times):
        raise ValueError("Number of output states does not match number of time points")

    # CHECK DENSITY MATRIX PROPERTIES
    # Apply RWA phase factors if needed
    if getattr(copy_sim_oqs.simulation_config, "rwa_sl", False):
        n_atoms = copy_sim_oqs.system.n_atoms
        omega_laser = copy_sim_oqs.laser._carrier_freq_fs
        logger.info(f"Applying RWA phase factors: n_atoms={n_atoms}, omega_laser={omega_laser}")
        states = apply_RWA_phase_factors(states, times, n_atoms, omega_laser)

    ### Enhanced state checking with more diagnostics
    logger.info("=== STATE-BY-STATE ANALYSIS ===")
    error_messages, time_cut = _check_density_matrix_properties(states, times)

    if not error_messages:
        logger.info("✅ Checks passed. DM remains Hermitian and positive.")
        logger.info(f"Final state trace: {states[-1].tr():.6f}")
        logger.info(f"Final state min eigenvalue: {states[-1].eigenenergies().min():.10f}")

    return result, time_cut


def _compute_total_evolution(sim_oqs: SimulationModuleOQS, options: dict) -> Result:
    """Compute evolution over the full time grid without segmentation.

    Uses the interactive evolution object ``evo_obj_int`` for the complete
    ``times_local`` array. Appropriate when segmentation by pulse activity
    is unnecessary (e.g., single broad pulse, debugging, or benchmarking).
    """
    current_state = sim_oqs.system.psi_ini
    actual_times = sim_oqs.times_local
    decay_ops_list = sim_oqs.decay_channels
    evo_obj = sim_oqs.evo_obj_int
    return _execute_single_evolution_segment(
        sim_oqs.simulation_config.ode_solver,
        evo_obj,
        decay_ops_list,
        current_state,
        actual_times,
        options,
    )


def compute_pulse_evolution(
    sim_oqs: SimulationModuleOQS,
    segmentation: bool = True,
    **solver_options: dict,
) -> Result:
    """
    Compute the evolution of the quantum system for a given pulse sequence.

    This function handles multi-pulse quantum evolution by segmenting the time array
    based on pulse regions and using appropriate evolution operators for each segment.
    It supports both interactive (with pulses) and free evolution periods.

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Prepared simulation object containing:
        - system: AtomicSystem with initial state, dipole operators, etc.
        - laser: LaserPulseSequence with pulse definitions
        - times_local: Time array for evolution
        - evo_obj_int: Evolution operator for interactive periods
        - evo_obj_free: Evolution operator for free evolution periods
        - simulation_config: Solver settings and tolerances
        - decay_channels: List of decay operators for dissipation
    **solver_options : dict
        User overrides for solver options (highest precedence).
        Examples: store_states=True, atol=1e-8, rtol=1e-6

    Parameters
    ----------
    segmentation : bool, default True
        If True, perform region-based segmented evolution (interactive vs free).
        If False, evolve once over the entire ``times_local`` using ``evo_obj_int``.

    Returns
    -------
    Result
        QuTiP Result object containing:
        - states: List of density matrices at each time point (if store_states=True)
        - times: Array of time points corresponding to states
        - final_state: Final density matrix (if store_final_state=True)

    Notes
    -----
    The function automatically segments the time evolution based on pulse activity:
    - Uses evo_obj_int during pulse periods (when laser field is active)
    - Uses evo_obj_free during free evolution periods (no laser field)
    - Handles overlapping time segments by extending segments by one point
    - Combines results from all segments into a single Result object

    The evolution is computed using either mesolve (master equation) or brmesolve
    (Bloch-Redfield master equation) depending on the ode_solver setting.
    """
    from qspectro2d.config.default_simulation_params import SOLVER_OPTIONS

    options: dict = SOLVER_OPTIONS
    if solver_options:
        options.update(solver_options)

    # Fast path: no segmentation requested
    if not segmentation:
        return _compute_total_evolution(sim_oqs, options)

    # --- Segmented evolution path ---
    all_states: list = []
    all_times: list = []

    current_state = sim_oqs.system.psi_ini
    actual_times = sim_oqs.times_local
    pulse_seq = sim_oqs.laser
    decay_ops_list = sim_oqs.decay_channels

    # Determine active regions and split
    pulse_regions = identify_non_zero_pulse_regions(actual_times, pulse_seq)
    split_times = split_by_active_regions(actual_times, pulse_regions)

    for i, curr_times in enumerate(split_times):
        # Ensure continuity by appending the first point of the next segment (except last)
        if i < len(split_times) - 1:
            next_times = split_times[i + 1]
            if len(next_times) > 0:
                curr_times = np.append(curr_times, next_times[0])

        # Decide on interactive vs free evo object
        start_idx = np.abs(actual_times - curr_times[0]).argmin()
        has_pulse = pulse_regions[start_idx]
        evo_obj = sim_oqs.evo_obj_int if has_pulse else sim_oqs.evo_obj_free

        segment_result = _execute_single_evolution_segment(
            sim_oqs.simulation_config.ode_solver,
            evo_obj,
            decay_ops_list,
            current_state,
            curr_times,
            options,
        )

        # Collect states
        if hasattr(segment_result, "states") and segment_result.states:
            if i < len(split_times) - 1:
                all_states.extend(segment_result.states[:-1])
                all_times.extend(segment_result.times[:-1])
            else:
                all_states.extend(segment_result.states)
                all_times.extend(segment_result.times)
            current_state = segment_result.states[-1]
        elif hasattr(segment_result, "final_state"):
            current_state = segment_result.final_state
        else:
            raise RuntimeError(
                "Solver must return either 'states' or 'final_state'. "
                "Check solver options: use 'store_states' or 'store_final_state'."
            )

    # Build consolidated Result (reuse last segment_result object for metadata safety)
    segment_result.states = all_states if all_states else []
    segment_result.times = np.array(all_times) if all_times else []
    return segment_result


def _execute_single_evolution_segment(
    ode_solver: str,
    evo_obj: Union[Qobj, QobjEvo],
    decay_ops_list: List[Qobj],
    current_state: Qobj,
    times_: np.ndarray,
    options: dict,
) -> Result:
    """
    Execute quantum evolution for a single time segment.

    This helper function chooses the appropriate QuTiP solver based on the
    ode_solver parameter and executes the evolution for the given time segment.

    Parameters
    ----------
    ode_solver : str
        Solver type: "ME" for standard master equation,
        "BR" for Bloch-Redfield master equation.
    evo_obj : Union[Qobj, QobjEvo]
        Evolution operator (Hamiltonian or Liouvillian).
        Can be a static Qobj or time-dependent QobjEvo.
    decay_ops_list : List[Qobj]
        List of collapse operators for dissipative dynamics.
    current_state : Qobj
        Initial density matrix for this time segment.
    times_ : np.ndarray
        Time points for this evolution segment.
    options : dict
        Solver options (atol, rtol, store_states, etc.).

    Returns
    -------
    Result
        QuTiP Result object with evolution data for this segment.
    """
    if ode_solver == "BR":
        # Optional Bloch-Redfield secular cutoff passthrough
        sec_cutoff = options.get("sec_cutoff", None)
        return brmesolve(
            evo_obj,
            current_state,
            times_,
            a_ops=decay_ops_list,
            options=options,
            sec_cutoff=sec_cutoff,
        )
    else:
        return mesolve(
            evo_obj,
            current_state,
            times_,
            c_ops=decay_ops_list,
            options=options,
        )


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

    # EVOLUTION COMPUTATION

    evolution_options = {
        "store_final_state": True,
        "store_states": False,  # Only need final state for efficiency
    }
    # evolution_options.update(kwargs)  # Allow override of options

    evolution_data = compute_pulse_evolution(sim_oqs=sim_oqs, **evolution_options)

    return evolution_data.final_state


def compute_1d_polarization(
    sim_oqs: SimulationModuleOQS,
    **kwargs,
) -> Union[Tuple[np.ndarray, np.ndarray, SimulationModuleOQS], Tuple[np.ndarray], np.ndarray]:
    """
    This function calculates the total nonlinear polarization response
    for a given coherence time and waiting time. It supports different output
    modes for plotting and analysis.

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Simulation configuration object containing:
        - System parameters (Hamiltonians, dipole operators, initial state)
        - Laser pulse sequence (multiple pulses for nonlinear spectroscopy)
        - Time arrays (local times, detection times)
        - Observable operators for signal detection
    **kwargs : dict
        Additional keyword arguments:
        - plot_example_evo : bool
            - Optional dipole_op: bool, False)  # only if this is set true we compute expectation value of the dipole op
            If True, return evolution data for plotting expectation values.
            Returns: (times, expectation_values, sim_oqs)
        - plot_example_polarization : bool
            If True, return raw polarization data for plotting.
            Returns: (polarization_plot_data,)
        - time_cut : float
            Time cutoff for detection (default: np.inf).
            Filters out data points after this time to avoid numerical artifacts.

    Returns
    -------
    Union[Tuple, np.ndarray]
        - If plot_example_evo=True:

            (times, expectation_values, sim_oqs)
            - times: np.ndarray, time points for evolution
            - expectation_values: np.ndarray, expectation values of observables
            - sim_oqs: SimulationModuleOQS, simulation object
        - If plot_example_polarization=True:
            (polarization_plot_data,)
            - polarization_plot_data: tuple of np.ndarray, raw polarization signals
        - Otherwise:
            nonlinear_signal: np.ndarray
            Complex nonlinear polarization signal as function of detection time

    Notes
    -----
    The function computes the nonlinear signal using the perturbative approach:
    P^(3)(t) = P_total(t) - Σ P_linear(t)

    where P_total is the full evolution with all pulses, and P_linear are the
    individual pulse contributions. This isolates the third-order nonlinear response.

    For multi-pulse sequences, the function uses segmented evolution to handle
    different time periods efficiently.
    """

    if kwargs.get("plot_example_evo", False):
        dip = kwargs.get("dipole_op", False)

        sys = sim_oqs.system
        data = compute_pulse_evolution(sim_oqs=sim_oqs, store_states=True)
        states = data.states
        times = data.times
        n_atoms = sys.n_atoms
        e_ops = sim_oqs.observable_ops
        rwa_sl = sim_oqs.simulation_config.rwa_sl
        omega_laser = sim_oqs.laser._carrier_freq_fs

        if dip:
            dipole_op = sys.dipole_op
            datas = get_expect_vals_with_RWA(
                states,
                times,
                n_atoms=n_atoms,
                e_ops=e_ops,
                omega_laser=omega_laser,
                rwa_sl=rwa_sl,
                dipole_op=sys.to_eigenbasis(dipole_op),
            )
        else:
            datas = get_expect_vals_with_RWA(
                states,
                times,
                n_atoms=n_atoms,
                e_ops=e_ops,
                omega_laser=omega_laser,
                rwa_sl=rwa_sl,
            )
        return times, datas, sim_oqs

    # COMPUTE EVOLUTION STATES

    evolution_data = _compute_n_pulse_evolution(sim_oqs)

    # COMPUTE LINEAR SIGNALS

    linear_signals = _compute_n_linear_signals(sim_oqs)

    # EXTRACT AND PROCESS DETECTION DATA

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


def _compute_n_pulse_evolution(
    sim_oqs: SimulationModuleOQS,
) -> List[Qobj]:
    """Compute the n-pulse evolution using segmented approach."""
    laser = sim_oqs.laser
    sys = sim_oqs.system
    peak_times = laser.pulse_peak_times
    fwhms = laser.pulse_fwhms
    psi_ini = sys.psi_ini
    times = sim_oqs.times_local

    n_pulses = len(laser.pulses)

    logger.info(f"  times_local: {len(times)} points from {times[0]:.2f} to {times[-1]:.2f} fs")
    logger.info(f"  times_det: {len(sim_oqs.times_det)} points")
    logger.info(f"  fwhms: {fwhms}")
    logger.info(f"  n_pulses: {n_pulses}")

    prev_pulse_start_idx = 0  # first -> include the time range before the first pulse
    current_state = psi_ini
    # Loop over pulses to calculate new start point n_pulses - 1
    for pulse_idx in range(1, n_pulses):
        # Calculate pulse start index
        next_pulse_start_time = peak_times[pulse_idx] - fwhms[pulse_idx]
        next_pulse_start_idx = np.abs(times - next_pulse_start_time).argmin()
        times_segment = times[prev_pulse_start_idx : next_pulse_start_idx + 1]
        logger.info(
            f"current pulses at {peak_times[pulse_idx - 1]} fs (idx {prev_pulse_start_idx}) and next at {peak_times[pulse_idx]} fs (idx {next_pulse_start_idx})"
        )
        times_segment = _ensure_valid_times(times_segment, times, prev_pulse_start_idx)
        logger.info(
            f"  Segment {pulse_idx}: {len(times_segment)} points from {times_segment[0]:.2f} to {times_segment[-1]:.2f} fs"
        )

        # Update simulation parameters for this segment
        sim_oqs.times_local = times_segment

        # Compute evolution for this segment
        current_state = _compute_next_start_point(sim_oqs=sim_oqs)
        sys.psi_ini = current_state

        # Update for next iteration
        prev_pulse_start_idx = next_pulse_start_idx

    times_final = _ensure_valid_times(times[prev_pulse_start_idx:], times, prev_pulse_start_idx)
    logger.info(
        f"  Final times: {len(times_final)} points from {times_final[0]:.2f} to {times_final[-1]:.2f} fs"
    )

    sim_oqs.times_local = times_final

    data_final = compute_pulse_evolution(sim_oqs=sim_oqs, store_states=True)

    detection_length = len(sim_oqs.times_det)

    # Ensure we have enough states
    if len(data_final.states) < detection_length:
        logger.warning(
            f"Not enough states in n pulse evo: got {len(data_final.states)}, need {detection_length}\n"
            f"  Final times: {len(times_final)} points from {times_final[0]:.2f} to {times_final[-1]:.2f} fs"
        )
        # Pad with the last state
        final_states = data_final.states.copy()
        while len(final_states) < detection_length:
            final_states.append(final_states[-1] if final_states else sim_oqs.system.psi_ini)
    else:
        final_states = data_final.states[-detection_length:]

    # Debug: Check the states being returned
    logger.debug(f"Total states computed: {len(data_final.states)}")
    logger.debug(f"Detection length expected: {detection_length}")
    logger.debug(f"Final states extracted: {len(final_states)}")

    sim_oqs.reset_times_local()  # restore original times
    sys.psi_ini = psi_ini  # restore initial state

    return final_states


def _compute_n_linear_signals(
    sim_oqs: SimulationModuleOQS,
) -> dict[str, List[Qobj]]:
    """Compute all linear signal contributions."""
    laser = sim_oqs.laser
    detection_length = len(sim_oqs.times_det)
    linear_data_states = {}
    for i, pulse in enumerate(laser):
        single_seq = LaserPulseSequence(pulses=[pulse])
        sim_oqs.laser = single_seq  # Update the laser sequence for each pulse
        data = compute_pulse_evolution(sim_oqs=sim_oqs, store_states=True)

        # Ensure we have enough states
        if len(data.states) < detection_length:
            logger.warning(
                f"Pulse {i}: Not enough states: got {len(data.states)}, need {detection_length}"
            )
            # Pad with the last state
            extracted_states = data.states.copy()
            while len(extracted_states) < detection_length:
                extracted_states.append(
                    extracted_states[-1] if extracted_states else sim_oqs.system.psi_ini
                )
        else:
            extracted_states = data.states[-detection_length:]

        linear_data_states[f"pulse{i}"] = extracted_states

        # Debug: Check linear signal states
        logger.debug(
            f"Pulse {i}: Total states: {len(data.states)}, extracted: {len(extracted_states)}"
        )

    sim_oqs.laser = laser  # Restore the original laser sequence
    # THE initial state is modified!!!
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
    # Ensure we return at least two time points when possible to satisfy solvers
    if times_segment.size == 0:
        n_full = len(full_times)
        if n_full == 0:
            logger.error("full_times is empty; returning empty time segment")
            return times_segment

        # Prefer [fallback_idx, fallback_idx+1] if available
        if 0 <= fallback_idx < n_full - 1:
            logger.warning(
                f"times_segment empty. Using fallback times at indices {fallback_idx}:{fallback_idx+2}"
            )
            return full_times[fallback_idx : fallback_idx + 2]

        # If fallback_idx is last index, use the last two points if available
        if n_full >= 2:
            logger.warning("times_segment empty. Using last two time points as fallback")
            return full_times[-2:]

        # Only one time point exists in full_times; return it and let caller guard
        logger.warning(
            "times_segment empty and only one time in full_times; returning single point"
        )
        return full_times[:1]

    # Time segment is valid, return as-is
    return times_segment


def _extract_detection_data(
    sim_oqs: SimulationModuleOQS,
    evolution_data: List[Qobj],
    linear_signals: dict[str, List[Qobj]],
    time_cut: float,
    # maybe could also pass kwargs for plotting
) -> dict:
    """Extract and process detection time data."""
    actual_det_times = sim_oqs.times_det_actual
    detection_times = sim_oqs.times_det

    # Debug: Print shapes for debugging
    logger.debug(f"evolution_data length: {len(evolution_data) if evolution_data else 'None'}")
    logger.debug(
        f"actual_det_times shape: {actual_det_times.shape if hasattr(actual_det_times, 'shape') else len(actual_det_times)}"
    )
    logger.debug(
        f"detection_times shape: {detection_times.shape if hasattr(detection_times, 'shape') else len(detection_times)}"
    )

    # Apply RWA phase factors if needed
    sys = sim_oqs.system
    if sim_oqs.simulation_config.rwa_sl:
        n_atoms = sys.n_atoms
        omega_laser = sim_oqs.laser._carrier_freq_fs
        evolution_data = apply_RWA_phase_factors(
            evolution_data, actual_det_times, n_atoms, omega_laser
        )
        for key in linear_signals:
            linear_signals[key] = apply_RWA_phase_factors(
                linear_signals[key], actual_det_times, n_atoms, omega_laser
            )

    # Calculate polarizations
    dipole_op = sys.to_eigenbasis(sys.dipole_op)
    polarizations = {}
    polarizations_full = complex_polarization(dipole_op, evolution_data)
    logger.debug(
        f"polarizations_full shape: {polarizations_full.shape if hasattr(polarizations_full, 'shape') else type(polarizations_full)}"
    )

    for key in linear_signals:
        polarizations[key] = complex_polarization(dipole_op, linear_signals[key])
        logger.debug(
            f"polarizations[{key}] shape: {polarizations[key].shape if hasattr(polarizations[key], 'shape') else type(polarizations[key])}"
        )

    # Calculate nonlinear signal - generalized for n pulses
    nonlinear_signal = polarizations_full.copy()  # Make explicit copy
    logger.debug(
        f"nonlinear_signal initial shape: {nonlinear_signal.shape if hasattr(nonlinear_signal, 'shape') else type(nonlinear_signal)}"
    )

    for pulse_key in linear_signals.keys():
        nonlinear_signal -= polarizations[pulse_key]

    logger.debug(
        f"nonlinear_signal after subtraction shape: {nonlinear_signal.shape if hasattr(nonlinear_signal, 'shape') else type(nonlinear_signal)}"
    )

    # Apply time cutoff - be careful about array shapes
    if time_cut < np.inf:
        # Create a mask for valid times
        valid_time_mask = actual_det_times < time_cut
        logger.debug(
            f"valid_time_mask shape: {valid_time_mask.shape}, sum: {np.sum(valid_time_mask)}"
        )

        if np.any(valid_time_mask):
            # Check if shapes are compatible for indexing
            if hasattr(nonlinear_signal, "shape") and len(nonlinear_signal.shape) > 0:
                if nonlinear_signal.shape[0] == len(valid_time_mask):
                    nonlinear_signal = nonlinear_signal[valid_time_mask]
                else:
                    logger.error(
                        f"Shape mismatch: nonlinear_signal shape {nonlinear_signal.shape}, mask length {len(valid_time_mask)}"
                    )
                    # Fallback: create zeros with correct shape
                    nonlinear_signal = np.zeros(np.sum(valid_time_mask), dtype=np.complex64)
            else:
                # If nonlinear_signal is scalar, create array
                logger.warning(
                    f"nonlinear_signal is scalar, creating array of length {np.sum(valid_time_mask)}"
                )
                nonlinear_signal = np.full(
                    np.sum(valid_time_mask), nonlinear_signal, dtype=np.complex64
                )
        else:
            # If no valid times, create a single zero
            logger.warning(f"time_cut={time_cut} filters out all data. Using zeros.")
            nonlinear_signal = np.zeros(1, dtype=np.complex64)

    # Ensure signal length matches expected detection times
    expected_length = len(detection_times)
    current_length = len(nonlinear_signal) if hasattr(nonlinear_signal, "__len__") else 1

    logger.debug(f"Expected length: {expected_length}, current length: {current_length}")

    if current_length < expected_length:
        logger.info(f"Padding signal from {current_length} to {expected_length} points")
        zeros_to_add = expected_length - current_length
        if hasattr(nonlinear_signal, "shape"):
            nonlinear_signal = np.concatenate(
                [nonlinear_signal, np.zeros(zeros_to_add, dtype=nonlinear_signal.dtype)]
            )
        else:
            # If scalar, create array
            nonlinear_signal = np.concatenate(
                [
                    np.array([nonlinear_signal], dtype=np.complex64),
                    np.zeros(zeros_to_add, dtype=np.complex64),
                ]
            )
    elif current_length > expected_length:
        logger.warning(f"Truncating signal from {current_length} to {expected_length} points")
        nonlinear_signal = nonlinear_signal[:expected_length]

    plot_polarization_data = [polarizations_full]
    for key in linear_signals.keys():
        pol = polarizations[key]
        plot_polarization_data.append(pol)

    return {
        "nonlinear_signal": nonlinear_signal,
        "plot_polarization_data": tuple(plot_polarization_data),
    }


# parallel processing 1d and 2d data
def _process_single_1d_combination(
    sim_oqs: SimulationModuleOQS,
    new_freqs: np.ndarray,
    phi1: float,
    phi2: float,
    **kwargs: dict,
) -> np.ndarray | None:
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
        System parameters (already contains the correct atomic frequencies).
    kwargs : dict
        Additional arguments.

    Returns
    -------
    np.ndarray or None
        Computed 1D polarization data array, or None if failed.
    """
    try:
        # CRITICAL FIX: Make a deep copy to avoid modifying the shared object
        local_sim_oqs = deepcopy(sim_oqs)

        local_sim_oqs.laser.update_phases(
            phases=[phi1, phi2, 0.0]
        )  # Update the laser phases in the local copy

        local_sim_oqs.system.frequencies_cm = new_freqs  # Update frequencies in the local copy

        data = compute_1d_polarization(
            sim_oqs=local_sim_oqs,
            **kwargs,
        )

        # Ensure we return the correct type and shape
        if data is None:
            logger.error("compute_1d_polarization returned None")
            return None

        if not isinstance(data, np.ndarray):
            logger.error(f"compute_1d_polarization returned {type(data)}, expected np.ndarray")
            return None

        # Additional shape validation
        expected_length = len(local_sim_oqs.times_det)
        if data.shape != (expected_length,):
            logger.error(f"Data shape {data.shape} doesn't match expected ({expected_length},)")
            return None

        return data

    except Exception as e:
        import traceback

        logger.error(f"Error in _process_single_1d_combination: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def compute_results_matrix_avg_1d(
    sim_oqs: SimulationModuleOQS,
    parallel: bool = True,
    **kwargs: dict,
) -> np.ndarray:
    """Compute the phase-resolved polarization matrix averaged over inhomogeneous realizations.

    This performs all expensive solver calls (optionally in parallel) and returns a
    3D array: results_matrix_avg[phi1_idx, phi2_idx, t] representing the averaged
    nonlinear polarization (still containing all phase information).

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Fully configured simulation object.
    parallel : bool, default=True
        Use process based parallelism for the (freq, phi1, phi2) combinations.
    **kwargs : dict
        Forwarded to compute_1d_polarization.

    Returns
    -------
    np.ndarray
        Averaged results matrix of shape (n_phases, n_phases, n_times_det) with dtype complex64.
    """
    # Phase cycling configuration
    n_phases = sim_oqs.simulation_config.n_phases
    n_inhomogen = sim_oqs.simulation_config.n_inhomogen
    max_workers = sim_oqs.simulation_config.max_workers
    from qspectro2d.config.default_simulation_params import PHASE_CYCLING_PHASES

    phases = PHASE_CYCLING_PHASES[:n_phases]
    if n_phases != 4:
        logger.warning(f"Phase cycling with {n_phases} phases may not be optimal for IFT")

    # Inhomogeneous broadening samples
    sys = sim_oqs.system
    delta_cm = sys.delta_cm
    frequencies_cm = sys.frequencies_cm

    # Each row = one realization, each column = atom index
    # Shape: (n_inhomogen, n_atoms)
    all_freq_sets = np.stack(
        [sample_from_gaussian(n_inhomogen, delta_cm, freq) for freq in frequencies_cm], axis=1
    )
    logger.debug(
        f"Using frequency samples: shape={all_freq_sets.shape}, dtype={all_freq_sets.dtype}"
    )

    # Prepare jobs
    combinations: list[tuple[int, int, int, np.ndarray, float, float]] = []
    for omega_idx in range(n_inhomogen):
        new_freqs = all_freq_sets[omega_idx]
        for phi1_idx, phi1 in enumerate(phases):
            for phi2_idx, phi2 in enumerate(phases):
                combinations.append((omega_idx, phi1_idx, phi2_idx, new_freqs, phi1, phi2))

    data_length = len(sim_oqs.times_det)
    results_array = np.empty((n_inhomogen, n_phases, n_phases, data_length), dtype=np.complex64)

    if not parallel:
        for omega_idx in range(n_inhomogen):
            new_freqs = all_freq_sets[omega_idx]
            for phi1_idx, phi1 in enumerate(phases):
                for phi2_idx, phi2 in enumerate(phases):
                    try:
                        data = _process_single_1d_combination(
                            sim_oqs=sim_oqs,
                            new_freqs=new_freqs,
                            phi1=phi1,
                            phi2=phi2,
                            **kwargs,
                        )
                        if data is None or not isinstance(data, np.ndarray):
                            data = np.full(data_length, np.nan, dtype=np.complex64)
                        elif data.shape != (data_length,):
                            if data.size == data_length:
                                data = data.reshape(data_length)
                            elif data.size < data_length:
                                padded_data = np.zeros(data_length, dtype=np.complex64)
                                padded_data[: data.size] = data.flatten()
                                data = padded_data
                            else:
                                data = data.flatten()[:data_length]
                        results_array[omega_idx, phi1_idx, phi2_idx, :] = data
                    except Exception as exc:
                        logger.error(
                            f"Serial combination ({omega_idx},{phi1_idx},{phi2_idx}) failed: {exc}"
                        )
                        results_array[omega_idx, phi1_idx, phi2_idx, :] = np.nan

    else:
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

                    # Validate data shape and type
                    if data is None:
                        logger.warning(
                            f"Combination ({omega_idx},{phi1_idx},{phi2_idx}) returned None"
                        )
                        results_array[omega_idx, phi1_idx, phi2_idx, :] = np.nan
                        continue

                    # Ensure data is a numpy array
                    if not isinstance(data, np.ndarray):
                        logger.error(
                            f"Expected ndarray, got {type(data)} for combination ({omega_idx},{phi1_idx},{phi2_idx})"
                        )
                        results_array[omega_idx, phi1_idx, phi2_idx, :] = np.nan
                        continue

                    # Check shape compatibility
                    if data.shape != (data_length,):
                        logger.error(
                            f"Shape mismatch: expected ({data_length},), got {data.shape} for combination ({omega_idx},{phi1_idx},{phi2_idx})"
                        )
                        # Try to handle shape mismatch gracefully
                        if data.size == data_length:
                            data = data.reshape(data_length)
                        elif data.size < data_length:
                            # Pad with zeros
                            padded_data = np.zeros(data_length, dtype=np.complex64)
                            padded_data[: data.size] = data.flatten()
                            data = padded_data
                        else:
                            # Truncate
                            data = data.flatten()[:data_length]

                    results_array[omega_idx, phi1_idx, phi2_idx, :] = data

                except Exception as exc:
                    logger.error(f"Combination ({omega_idx},{phi1_idx},{phi2_idx}) failed: {exc}")
                    results_array[omega_idx, phi1_idx, phi2_idx, :] = np.nan

    results_matrix_avg = np.mean(results_array, axis=0)
    logger.debug(f"results_matrix_avg shape={results_matrix_avg.shape}")
    return results_matrix_avg


def extract_signal_components_1d(
    results_matrix_avg: np.ndarray, sim_oqs: SimulationModuleOQS
) -> dict[str, np.ndarray]:
    """Extract phase-cycled signal components from the averaged results matrix.

    Parameters
    ----------
    results_matrix_avg : np.ndarray
        Array of shape (n_phases, n_phases, n_times) from compute_results_matrix_avg_1d.
    sim_oqs : SimulationModuleOQS
        Simulation object (used for signal_types and phase list).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from signal type (e.g. 'rephasing') to extracted polarization P_kS(t).
    """
    from qspectro2d.config.default_simulation_params import PHASE_CYCLING_PHASES, COMPONENT_MAP

    n_phases = sim_oqs.simulation_config.n_phases
    phases = PHASE_CYCLING_PHASES[:n_phases]
    signal_types = sim_oqs.simulation_config.signal_types

    extracted: dict[str, np.ndarray] = {}
    for sig in signal_types:
        comp = COMPONENT_MAP.get(sig)
        if comp is None:
            logger.warning(f"Unknown signal type '{sig}' – skipping")
            continue
        extracted[sig] = extract_ift_signal_component(
            results_matrix=results_matrix_avg, phases=phases, component=comp
        )
    return extracted


def compute_detected_fields_from_polarization(
    polarization_components: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert polarization components P_kS(t) to detected electric fields E_kS(t).

    Uses simple proportionality E_kS = i * P_kS (overall scaling factors can be
    applied externally if needed).

    Parameters
    ----------
    polarization_components : dict[str, np.ndarray]
        Mapping from signal type to polarization arrays.

    Returns
    -------
    dict[str, np.ndarray]
        Same keys with complex field arrays.
    """
    return {k: 1j * v for k, v in polarization_components.items()}


def parallel_compute_1d_E_with_inhomogenity(
    sim_oqs: SimulationModuleOQS,
    parallel: bool = True,
    **kwargs: dict,
) -> List[np.ndarray]:
    """Backward-compatible wrapper returning polarization components list.

    This preserves the original public API while internally using the new
    modular functions:
      1. compute_results_matrix_avg_1d
      2. extract_signal_components_1d

    Returns list in the order of sim_oqs.simulation_config.signal_types.
    """
    results_matrix_avg = compute_results_matrix_avg_1d(sim_oqs=sim_oqs, parallel=parallel, **kwargs)
    components_dict = extract_signal_components_1d(results_matrix_avg, sim_oqs)
    ordered = [
        components_dict[s] for s in sim_oqs.simulation_config.signal_types if s in components_dict
    ]
    return ordered


# Helper functions for IFT processing
def extract_ift_signal_component(
    results_matrix: np.ndarray, phases: list, component: list[int, int, int]
) -> np.ndarray:
    """
    Extract a specific signal component using inverse Fourier transform (IFT).

    Computes the IFT signal component:
    P_{l,m}(t) = Σ_{phi1} Σ_{phi2} P_{phi1,phi2}(t) * exp(-i(l*phi1 + m*phi2))

    Parameters
    ----------
    results_matrix : np.ndarray
        2D matrix of polarization results indexed by [phi1_idx, phi2_idx].
        Each element is a 1D or 2D array representing the signal at those phases.
        Shape: (n_phases, n_phases, ...)
    phases : list
        List of phase values used in the experiment (typically [0, π/2, π, 3π/2]).
        Length should match the first two dimensions of results_matrix.
    component : list[int, int, int]
        IFT coefficients [l, m, n] for the phase factors:
        - l: coefficient for first pulse phase (φ₁)
        - m: coefficient for second pulse phase (φ₂)
        - n: coefficient for detection phase (φ_det), typically 0 or 1

    Returns
    -------
    np.ndarray
        Extracted signal component with the same shape as individual matrix elements.
        The signal is computed as:
        P_{l,m}(t) = Σ_{i,j} P_{φ₁ᵢ,φ₂ⱼ}(t) * exp(-i(l*φ₁ᵢ + m*φ₂ⱼ + n*φ_det))

    Examples
    --------
    Extract rephasing signal component:
    >>> rephasing = extract_ift_signal_component(results_matrix, phases, [-1, 1, 1])

    Extract nonrephasing signal component:
    >>> nonrephasing = extract_ift_signal_component(results_matrix, phases, [1, -1, 1])

    Notes
    -----
    The function handles missing data gracefully by skipping None values in the
    results matrix. The output shape matches the first valid (non-None) element
    found in the matrix.

    Common phase cycling components for 2D spectroscopy:
    - Rephasing: [-1, 1, 1] - extracts signals with (-φ₁ + φ₂) phase evolution
    - Nonrephasing: [1, -1, 1] - extracts signals with (φ₁ - φ₂) phase evolution
    """
    l, m, n = component
    n_phases = len(phases)

    # Validate input dimensions
    if results_matrix.shape[0] != n_phases or results_matrix.shape[1] != n_phases:
        raise ValueError("results_matrix dimensions must match number of phases in both axes")

    # Determine the shape of individual signal elements
    sample_shape = None
    for i in range(n_phases):
        for j in range(n_phases):
            if results_matrix[i, j] is not None:
                sample_shape = results_matrix[i, j].shape
                break
        if sample_shape is not None:
            break

    if sample_shape is None:
        raise ValueError("All entries in results_matrix are None")

    # Initialize the output array with zeros
    signal_component = np.zeros(sample_shape, dtype=np.complex64)

    # Perform the IFT summation
    for i, phi1 in enumerate(phases):
        for j, phi2 in enumerate(phases):
            P_phi1_phi2 = results_matrix[i, j]
            if P_phi1_phi2 is not None:
                phase_factor = np.exp(-1j * (l * phi1 + m * phi2))
                signal_component += P_phi1_phi2 * phase_factor

    return signal_component
