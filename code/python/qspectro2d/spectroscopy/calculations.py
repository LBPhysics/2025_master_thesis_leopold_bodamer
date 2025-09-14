# -*- coding: utf-8 -*-
"""
Quantum Spectroscopy Calculations Module

This module contains functions for computing quantum evolution, polarization signals,
and 2D spectroscopy data using QuTiP. It handles multi-pulse sequences, phase cycling,
and parallel processing for efficient computation.

Main Functions:
- compute_seq_evolution: Core evolution computation
- compute_1d_polarization: 1D spectroscopy calculations
- parallel_compute_1d_E_with_inhomogenity: Parallel 1D with inhomogeneity
"""

# STANDARD LIBRARY IMPORTS
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Union, Tuple, final
from matplotlib.pylab import f
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


def _execute_single_evolution_segment(
    ode_solver: str,
    evo_obj: Union[Qobj, QobjEvo],
    decay_ops_list: List[Qobj],
    current_state: Qobj,
    tlist: np.ndarray,
    options: dict,
) -> Result:
    """
    Execute quantum evolution for a single time segment or either activated or deactivated pulse.

    Parameters
    ----------
    ode_solver : str
        Solver type: "ME" for standard master equation / "BR" for Bloch-Redfield master equation.
    evo_obj : Union[Qobj, QobjEvo]
        Evolution operator (Hamiltonian or Liouvillian).
        Can be a static Qobj or time-dependent QobjEvo.
    decay_ops_list : List[Qobj]
        List of collapse operators for dissipative dynamics.
    current_state : Qobj
        Initial density matrix for this time segment.
    tlist : np.ndarray
        Time points for this evolution segment.
    options : dict
        Solver options (atol, rtol, store_states, etc.).

    Returns
    -------
    Result
        QuTiP Result object with evolution data for this segment.
    """
    if ode_solver == "BR":
        # Optional Bloch-Redfield secular cutoff passthrough TODO add to options
        sec_cutoff = options.get("sec_cutoff", None)
        res = brmesolve(
            H=evo_obj,
            psi0=current_state,
            tlist=tlist,
            a_ops=decay_ops_list,
            options=options,
            sec_cutoff=sec_cutoff,
        )
    else:
        res = mesolve(
            H=evo_obj,
            rho0=current_state,
            tlist=tlist,
            c_ops=decay_ops_list,
            options=options,
        )
    return res


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


def compute_seq_evolution(
    sim_oqs: SimulationModuleOQS,
    segmentation: bool = False,
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

    evolution_options = {
        "store_final_state": True,
        "store_states": False,  # Only need final state for efficiency
    }
    evolution_options.update(**kwargs)  # Allow override of options

    # results for detection time containing the complex polarization
    result_P = compute_seq_evolution(sim_oqs=sim_oqs, **evolution_options)

    return result_P.final_state


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
            P_3_data: np.ndarray
            P^(3)(t) = P_total(t) - Σ P_linear(t)
            where P_total is the full evolution with all pulses, and P_linear are the
            individual pulse contributions. This isolates the third-order nonlinear response.
    """

    if kwargs.get("plot_example_evo", False):  # TODO somehow add this to the general flow
        dip = kwargs.get("dipole_op", False)

        sys = sim_oqs.system
        data = compute_seq_evolution(sim_oqs=sim_oqs, store_states=True)
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

    # total
    states_P = _compute_n_pulse_det_states(sim_oqs)

    # COMPUTE LINEAR SIGNALS

    list_states_P = _compute_n_linear_det_states(sim_oqs)

    # EXTRACT AND PROCESS DETECTION DATA

    time_cut_val = kwargs.get("time_cut", np.inf)
    # Ensure time_cut is a proper float, not a Qobj
    if hasattr(time_cut_val, "full"):  # Check if it's a Qobj
        raise TypeError("time_cut must be a float, not a Qobj")
    elif not isinstance(time_cut_val, (int, float)):
        time_cut_val = float(time_cut_val)

    detection_data = _extract_detection_data(
        sim_oqs,
        states_P,
        list_states_P,
        time_cut_val,
        # maybe could also pass kwargs for plotting
    )

    # Return based on plotting flag
    if kwargs.get("plot_example_polarization", False):
        return detection_data["plot_polarization_data"]

    return detection_data["P_3_data"]


def _compute_n_pulse_det_states(
    sim_oqs: SimulationModuleOQS,
) -> List[Qobj]:
    """Compute n-pulse evolution by stitching segments up to each next pulse's active start."""

    def _clone_sim_for_segment(
        base_sim: SimulationModuleOQS,
        *,
        times_local: np.ndarray | None = None,
        psi_ini_override: Qobj | None = None,
        laser_override: LaserPulseSequence | None = None,
    ) -> SimulationModuleOQS:
        """Create an isolated simulation object for a specific segment without mutating the original.

        Only the provided fields are overridden on a deep-copied simulation.
        """
        local_sim = deepcopy(base_sim)
        if times_local is not None:
            local_sim.times_local = times_local
        if psi_ini_override is not None:
            local_sim.system.psi_ini = psi_ini_override
        if laser_override is not None:
            local_sim.laser = laser_override
        return local_sim

    laser = sim_oqs.laser
    n_pulses = len(laser.pulses)
    sys = sim_oqs.system
    peak_times = laser.pulse_peak_times
    psi_ini = sys.psi_ini
    times = sim_oqs.times_local

    logger.info(f"  times_local: {len(times)} points from {times[0]:.2f} to {times[-1]:.2f} fs")
    logger.info(f"  times_det: {len(sim_oqs.times_det)} points")
    logger.info(f"  n_pulses: {n_pulses}")

    prev_start_idx = 0  # first -> include the time range before the first pulse
    current_state = psi_ini

    # Build intermediate segments up to the start of each next pulse's active region
    for pulse_idx in range(1, n_pulses):
        # Calculate the beginning of the next pulse's ACTIVE region to avoid dropping pulse tails
        next_start_time = laser.pulses[pulse_idx].active_time_range[0]
        next_start_idx = int(np.searchsorted(times, next_start_time, side="left"))

        # Ensure forward progress and include boundary point
        if next_start_idx <= prev_start_idx:
            next_start_idx = min(prev_start_idx + 1, len(times) - 1)

        seg_times = times[prev_start_idx : next_start_idx + 1]
        seg_times = _ensure_valid_times(seg_times, times, prev_start_idx)

        logger.debug(
            f"Segment {pulse_idx}: idx[{prev_start_idx}:{next_start_idx}] "
            f"-> t[{seg_times[0]:.2f},{seg_times[-1]:.2f}] fs ({len(seg_times)} pts)"
        )

        seg_sim = _clone_sim_for_segment(
            sim_oqs,
            times_local=seg_times,
            psi_ini_override=current_state,
        )

        # Compute evolution for this segment (final state becomes next start state)
        current_state = _compute_next_start_point(sim_oqs=seg_sim)
        prev_start_idx = next_start_idx

    final_times = _ensure_valid_times(times[prev_start_idx:], times, prev_start_idx)
    logger.debug(
        f"Final segment: idx[{prev_start_idx}:{len(times)-1}] "
        f"-> t[{final_times[0]:.2f},{final_times[-1]:.2f}] fs ({len(final_times)} pts)"
    )

    # Final segment uses the accumulated current_state as initial state
    final_sim = _clone_sim_for_segment(
        sim_oqs, times_local=final_times, psi_ini_override=current_state
    )
    final_result = compute_seq_evolution(sim_oqs=final_sim, store_states=True)

    # Ensure we have enough states
    detection_length = len(sim_oqs.times_det)
    if len(final_result.states) < detection_length:
        logger.warning(
            f"Not enough states in final segment: got {len(final_result.states)}, need {detection_length}; padding last state."
        )
        # Pad with the last state
        final_states = final_result.states
        while len(final_states) < detection_length:
            final_states.append(final_states[-1] if final_states else sim_oqs.system.psi_ini)
    else:
        final_states = final_result.states[-detection_length:]

    # Debug: Check the states being returned
    logger.debug(f"Total final states: {len(final_result.states)}; extracted: {len(final_states)}")

    return final_states


def _compute_n_linear_det_states(
    sim_oqs: SimulationModuleOQS,
) -> List[List[Qobj]]:
    """Compute linear contributions per pulse.

    Returns a list ordered by pulse index: [states_from_pulse0, states_from_pulse1, ...].
    Each element is a list of Qobj density matrices aligned to `times_det`.
    """
    laser = sim_oqs.laser
    detection_length = len(sim_oqs.times_det)
    linear_states_by_pulse: list[list[Qobj]] = []

    for i, pulse in enumerate(laser):
        single_seq = LaserPulseSequence(pulses=[pulse])
        # Use an isolated simulation with a single-pulse sequence
        local_sim = deepcopy(sim_oqs)
        local_sim.laser = single_seq
        result_det = compute_seq_evolution(sim_oqs=local_sim, store_states=True)

        # Ensure we have enough states
        if len(result_det.states) < detection_length:
            logger.warning(
                f"Pulse {i}: Not enough states: got {len(result_det.states)}, need {detection_length}"
            )
            # Pad with the last state
            extracted_states = result_det.states
            while len(extracted_states) < detection_length:
                extracted_states.append(
                    extracted_states[-1] if extracted_states else sim_oqs.system.psi_ini
                )
        else:
            extracted_states = result_det.states[-detection_length:]

        linear_states_by_pulse.append(extracted_states)

        # Debug: Check linear signal states
        logger.debug(
            f"Pulse {i}: Total states: {len(result_det.states)}, extracted: {len(extracted_states)}"
        )

    return linear_states_by_pulse


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
    full_signal_states_P: List[Qobj],
    linear_signals_states_P: List[Qobj],
    time_cut: float,
    # maybe could also pass kwargs for plotting
) -> dict:
    """Extract and process detection time data."""
    actual_det_times = sim_oqs.times_det_actual
    detection_times = sim_oqs.times_det

    # Debug: Print shapes for debugging
    logger.debug(
        f"full_signal_states_P length: {len(full_signal_states_P) if full_signal_states_P else 'None'}"
    )
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

        # apply to the full signal
        omega_laser = sim_oqs.laser._carrier_freq_fs
        full_signal_states_P = apply_RWA_phase_factors(
            full_signal_states_P, actual_det_times, n_atoms, omega_laser
        )

        # Apply RWA to each linear contribution
        linear_signals_states_P = [
            apply_RWA_phase_factors(
                states, actual_det_times, sys.n_atoms, sim_oqs.laser._carrier_freq_fs
            )
            for states in linear_signals_states_P
        ]

    # Calculate polarizations
    dipole_op = sys.to_eigenbasis(sys.dipole_op)
    full_P_data = complex_polarization(dipole_op, full_signal_states_P)
    logger.debug(
        f"full_P_data shape: {full_P_data.shape if hasattr(full_P_data, 'shape') else type(full_P_data)}"
    )

    linear_pols_data = []
    for idx, states in enumerate(linear_signals_states_P):
        pol = complex_polarization(dipole_op, states)
        linear_pols_data.append(pol)
        logger.debug(
            f"linear_pols_data[{idx}] shape: {pol.shape if hasattr(pol, 'shape') else type(pol)}"
        )

    # Calculate nonlinear signal - generalized for n pulses
    P_3_data = full_P_data.copy()  # Make explicit copy
    logger.debug(
        f"P_3_data initial shape: {P_3_data.shape if hasattr(P_3_data, 'shape') else type(P_3_data)}"
    )

    for pol in linear_pols_data:
        P_3_data -= pol

    logger.debug(
        f"P_3_data after subtraction shape: {P_3_data.shape if hasattr(P_3_data, 'shape') else type(P_3_data)}"
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
            if P_3_data.shape[0] == len(valid_time_mask):
                P_3_data = P_3_data[valid_time_mask]
            else:
                logger.error(
                    f"Shape mismatch: P_3_data shape {P_3_data.shape}, mask length {len(valid_time_mask)}"
                )
                # Fallback: create zeros with correct shape
                P_3_data = np.zeros(np.sum(valid_time_mask), dtype=np.complex64)
        else:
            # If no valid times, create a single zero
            logger.warning(f"time_cut={time_cut} filters out all data. Using zeros.")

    # Ensure signal length matches expected detection times
    expected_length = len(detection_times)
    current_length = len(P_3_data) if hasattr(P_3_data, "__len__") else 1

    logger.debug(f"Expected length: {expected_length}, current length: {current_length}")

    if current_length < expected_length:
        logger.info(f"Padding signal from {current_length} to {expected_length} points")
        zeros_to_add = expected_length - current_length
        P_3_data = np.concatenate([P_3_data, np.zeros(zeros_to_add, dtype=P_3_data.dtype)])
    elif current_length > expected_length:
        logger.warning(f"Truncating signal from {current_length} to {expected_length} points")
        P_3_data = P_3_data[:expected_length]

    plot_polarization_data = [full_P_data]
    plot_polarization_data.extend(linear_pols_data)

    return {
        "P_3_data": P_3_data,
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
        from qspectro2d.config.default_simulation_params import DETECTION_PHASE

        local_sim_oqs.laser.update_phases(
            phases=[phi1, phi2, DETECTION_PHASE]
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


def compute_1d_pol_matrix_phases(
    sim_oqs: SimulationModuleOQS,
    parallel: bool = True,
    **kwargs: dict,
) -> np.ndarray:
    """Compute the phase-resolved polarization matrix averaged over inhomogeneous realizations.

    This performs all expensive solver calls (optionally in parallel) and returns a
    3D array: Pol_matrix_phases[phi1_idx, phi2_idx, t] representing the averaged
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
    if n_phases < 4:
        logger.warning(f"Phase cycling with {n_phases} phases may be too small for IFT")

    # Inhomogeneous broadening samples
    sys = sim_oqs.system
    delta_cm = sys.delta_cm
    frequencies_cm = sys.frequencies_cm

    # Each row = one realization, each column = atom index
    # Shape: (n_inhomogen, n_atoms)
    all_freq_sets = np.stack(
        [sample_from_gaussian(n_inhomogen, delta_cm, freq) for freq in frequencies_cm], axis=1
    )
    # Use logger instead of stdout for debug info
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

    Pol_matrix_phases = np.mean(results_array, axis=0)
    logger.debug(f"Pol_matrix_phases shape={Pol_matrix_phases.shape}")
    return Pol_matrix_phases


def extract_P_lmn_1d(
    Pol_matrix_phases: np.ndarray, sim_oqs: SimulationModuleOQS
) -> dict[str, np.ndarray]:
    """Extract phase-cycled signal components from the averaged results matrix.

    Parameters
    ----------
    Pol_matrix_phases : np.ndarray
        Array of shape (n_phases, n_phases, n_times) from compute_1d_pol_matrix_phases.
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
            results_matrix=Pol_matrix_phases, phases=phases, component=comp
        )
    return extracted


def compute_e_from_p(
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
    """Compute electric field components E_kS(t) with inhomogeneous averaging.

    Internally:
      1. compute_1d_pol_matrix_phases -> phase-resolved polarization matrix
      2. extract_P_lmn_1d -> polarization components P_kS(t) per signal type
      3. compute_e_from_p -> convert to electric fields E_kS(t) = i P_kS(t)

    Returns list in the order of sim_oqs.simulation_config.signal_types.
    """
    Pol_matrix_phases = compute_1d_pol_matrix_phases(sim_oqs=sim_oqs, parallel=parallel, **kwargs)
    components_dict = extract_P_lmn_1d(Pol_matrix_phases, sim_oqs)
    e_components_dict = compute_e_from_p(components_dict)
    ordered_E = [
        e_components_dict[s]
        for s in sim_oqs.simulation_config.signal_types
        if s in e_components_dict
    ]
    return ordered_E


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
    from qspectro2d.config.default_simulation_params import DETECTION_PHASE

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
                phase_factor = np.exp(-1j * (l * phi1 + m * phi2 + n * DETECTION_PHASE))
                signal_component += P_phi1_phi2 * phase_factor

    return signal_component
