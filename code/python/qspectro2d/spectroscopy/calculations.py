# -*- coding: utf-8 -*-

# =============================
# STANDARD LIBRARY IMPORTS
# =============================
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from re import S
from typing import List, Union, Tuple
import logging
import numpy as np
from qutip import Qobj, QobjEvo, Result, mesolve, brmesolve

# =============================
# LOCAL IMPORTS
# =============================
from qspectro2d.core.simulation import SimulationModuleOQS
from qspectro2d.core.laser_system.laser_class import (
    LaserPulseSequence,
    identify_non_zero_pulse_regions,
    split_by_active_regions,
)
from qspectro2d.spectroscopy.inhomogenity import sample_from_gaussian
from qspectro2d.utils import apply_RWA_phase_factors, get_expect_vals_with_RWA
from qspectro2d.spectroscopy.polarization import complex_polarization


# =============================
# LOGGING CONFIGURATION
# =============================
logger = logging.getLogger(__name__)
# Set to DEBUG level to see all debug messages
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)  # Changed to WARNING to reduce verbosity

# Remove CONFIG usage throughout; introduce module-level defaults container
DEFAULT_SOLVER_THRESHOLDS = {
    "negative_eigval_threshold": -1e-3,
    "trace_tolerance": 1e-6,
}


def compute_pulse_evolution(
    sim_oqs: SimulationModuleOQS,
    solver_defaults: dict | None = None,
    **solver_options: dict,
) -> Result:
    """
    Compute the evolution of the system for a given pulse sequence.

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Prepared simulation object.
    solver_defaults : dict | None
        Default solver options (previously CONFIG.solver.solver_options).
    **solver_options : dict
        User overrides (highest precedence).

    Returns
    -------
    Result
        QuTiP Result object containing evolution data.
    """
    # =============================
    # CONFIGURE SOLVER OPTIONS
    # =============================
    options = {}
    if solver_defaults:
        options.update(solver_defaults)
    if solver_options:
        options.update(solver_options)

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
            evo_obj = sim_oqs.evo_obj_int
        else:
            evo_obj = sim_oqs.evo_obj_free
        # Execute evolution for this time segment
        result = _execute_single_evolution_segment(
            sim_oqs.simulation_config.ode_solver,
            evo_obj,
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
                "Use either 'store_states' or 'store_final_state' in options."
            )

    # Create combined result object
    result.states = all_states if all_states else []
    result.times = np.array(all_times) if all_times else []

    return result


def _execute_single_evolution_segment(
    ode_solver: str,
    evo_obj: Union[Qobj, QobjEvo],
    decay_ops_list: List[Qobj],
    current_state: Qobj,
    times_: np.ndarray,
    options: dict,
) -> Result:
    """Execute evolution for a single time segment."""
    if ode_solver == "BR":
        return brmesolve(
            evo_obj,
            current_state,
            times_,
            a_ops=decay_ops_list,
            options=options,
        )
    else:
        return mesolve(
            evo_obj,
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
    logger.info(f"Checking '{sim_oqs.simulation_config.ode_solver}' solver")
    copy_sim_oqs = deepcopy(sim_oqs)
    t_max = 2 * copy_sim_oqs.simulation_config.t_max
    dt = 10 * copy_sim_oqs.simulation_config.dt
    t0 = -2 * copy_sim_oqs.laser.pulse_fwhms[0]
    times = np.linspace(t0, t_max, int((t_max - t0) / dt) + 1)
    copy_sim_oqs.times_local = times

    # =============================
    # DETAILED SYSTEM DIAGNOSTICS
    # =============================
    logger.info(f"=== SOLVER DIAGNOSTICS ===")
    logger.info(f"Solver: {copy_sim_oqs.simulation_config.ode_solver}")
    logger.info(f"Time range: t0={t0:.3f}, t_max={t_max:.3f}, dt={dt:.6f}")
    logger.info(f"Number of time points: {len(times)}")
    logger.info(
        f"RWA enabled: {getattr(copy_sim_oqs.simulation_config, 'rwa_sl', False)}"
    )

    ### Initial state diagnostics
    psi_ini = copy_sim_oqs.system.psi_ini
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

    ### System Hamiltonian diagnostics
    try:
        if (
            hasattr(copy_sim_oqs, "evo_obj_free")
            and copy_sim_oqs.evo_obj_free is not None
        ):
            H_free = copy_sim_oqs.evo_obj_free
            if hasattr(H_free, "dims"):
                logger.info(f"Free Hamiltonian dims: {H_free.dims}")
            logger.info(f"Free Hamiltonian type: {type(H_free)}")

        if hasattr(copy_sim_oqs, "decay_channels") and copy_sim_oqs.decay_channels:
            logger.info(f"Number of decay channels: {len(copy_sim_oqs.decay_channels)}")

    except Exception as e:
        logger.warning(f"Could not analyze Hamiltonian: {e}")

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

    ### Check solver tolerances
    solver_opts = {}  # no global defaults after CONFIG removal
    logger.info(f"Solver options: {solver_opts}")
    neg_thr = DEFAULT_SOLVER_THRESHOLDS["negative_eigval_threshold"]
    tr_tol = DEFAULT_SOLVER_THRESHOLDS["trace_tolerance"]
    logger.info(f"Negative eigenvalue threshold: {neg_thr}")
    logger.info(f"Trace tolerance: {tr_tol}")

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
    error_messages = []
    time_cut = np.inf  # time after which the checks failed

    # Apply RWA phase factors if needed
    if getattr(copy_sim_oqs.simulation_config, "rwa_sl", False):
        n_atoms = copy_sim_oqs.system.n_atoms
        omega_laser = copy_sim_oqs.laser._carrier_freq_fs
        logger.info(
            f"Applying RWA phase factors: n_atoms={n_atoms}, omega_laser={omega_laser}"
        )
        states = apply_RWA_phase_factors(states, times, n_atoms, omega_laser)

    ### Enhanced state checking with more diagnostics
    logger.info("=== STATE-BY-STATE ANALYSIS ===")
    check_interval = max(1, len(states) // 10)  # Check every 10% of states

    for index, state in enumerate(states):
        time = times[index]

        ### Detailed analysis for problematic or sample states
        if index % check_interval == 0 or index < 5:  # First 5 states + samples
            logger.info(
                f"State {index} (t={time:.3f}): trace={state.tr():.6f}, Hermitian={state.isherm}"
            )

        ### Check Hermiticity
        if not state.isherm:
            error_messages.append(f"Density matrix is not Hermitian after t = {time}")
            logger.error(f"Non-Hermitian density matrix at t = {time}")
            logger.error(
                f"  State details: trace={state.tr():.6f}, shape={state.shape}"
            )

        ### Check positive semidefiniteness
        eigvals = state.eigenenergies()
        min_eigval = eigvals.min()

        if not np.all(eigvals >= neg_thr):
            error_messages.append(
                f"Density matrix is not positive semidefinite after t = {time}: The lowest eigenvalue is {min_eigval}"
            )
            logger.error(f"NEGATIVE EIGENVALUE DETECTED:")
            logger.error(f"  Time: {time:.6f}")
            logger.error(f"  Min eigenvalue: {min_eigval:.12f}")
            logger.error(f"  Threshold: {neg_thr}")
            logger.error(f"  All eigenvalues: {eigvals[:5]}...")  # Show first 5
            logger.error(f"  State trace: {state.tr():.10f}")
            logger.error(f"  State index: {index}/{len(states)}")

            ### Additional diagnostics for the failing state
            if index > 0:
                prev_state = states[index - 1]
                prev_eigvals = prev_state.eigenenergies()
                logger.error(
                    f"  Previous state (t={times[index-1]:.6f}) min eigval: {prev_eigvals.min():.12f}"
                )
                logger.error(
                    f"  Eigenvalue change: {min_eigval - prev_eigvals.min():.12f}"
                )

            time_cut = time

        ### Check trace preservation
        trace_val = state.tr()
        if not np.isclose(trace_val, 1.0, atol=tr_tol):
            error_messages.append(
                f"Density matrix is not trace-preserving after t = {time}: The trace is {trace_val}"
            )
            logger.error(f"TRACE VIOLATION:")
            logger.error(f"  Time: {time:.6f}")
            logger.error(f"  Trace: {trace_val:.10f}")
            logger.error(f"  Deviation from 1: {abs(trace_val - 1.0):.10f}")
            logger.error(f"  Tolerance: {tr_tol}")

            time_cut = min(time_cut, time)

        ### Break on first error for detailed analysis
        if error_messages:
            logger.error("=== FIRST ERROR ANALYSIS ===")
            logger.error(
                f"Stopping analysis at first error (state {index}, t={time:.6f})"
            )
            logger.error(
                "Density matrix validation failed: " + "; ".join(error_messages)
            )
            break

    if not error_messages:
        logger.info("✅ Checks passed. DM remains Hermitian and positive.")
        logger.info(f"Final state trace: {states[-1].tr():.6f}")
        logger.info(
            f"Final state min eigenvalue: {states[-1].eigenenergies().min():.10f}"
        )

    return result, time_cut


def diagnose_solver_instability(sim_oqs: SimulationModuleOQS) -> dict:
    """
    Comprehensive diagnostic function to identify the source of solver instabilities.

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Simulation configuration object

    Returns
    -------
    dict
        Diagnostic information including potential issues and recommendations
    """
    diagnostics = {
        "issues_found": [],
        "recommendations": [],
        "system_info": {},
        "solver_info": {},
    }

    ### Analyze initial state
    psi_ini = sim_oqs.system.psi_ini
    diagnostics["system_info"]["initial_state"] = {
        "type": psi_ini.type,
        "shape": psi_ini.shape,
        "is_hermitian": psi_ini.isherm,
        "trace": complex(psi_ini.tr()),
        "is_normalized": abs(psi_ini.tr() - 1.0) < 1e-10,
    }

    if psi_ini.type == "oper":
        eigvals = psi_ini.eigenenergies()
        diagnostics["system_info"]["initial_state"]["min_eigenvalue"] = float(
            eigvals.min()
        )
        diagnostics["system_info"]["initial_state"]["max_eigenvalue"] = float(
            eigvals.max()
        )

        if eigvals.min() < 0:
            diagnostics["issues_found"].append("Initial state has negative eigenvalues")
            diagnostics["recommendations"].append(
                "Check initial state construction/normalization"
            )

    ### Analyze time parameters
    t_max = sim_oqs.simulation_config.t_max
    dt = sim_oqs.simulation_config.dt
    solver_type = sim_oqs.simulation_config.ode_solver

    diagnostics["solver_info"] = {
        "solver_type": solver_type,
        "t_max": t_max,
        "dt": dt,
        "time_steps": int(t_max / dt),
        "solver_options": {},
    }

    ### Check if time step is too large
    if dt > 0.1:  # Arbitrary threshold, adjust based on your system
        diagnostics["issues_found"].append(f"Large time step: dt={dt}")
        diagnostics["recommendations"].append(
            "Try reducing dt for better numerical stability"
        )

    ### Analyze system Hamiltonian
    try:
        if hasattr(sim_oqs, "evo_obj_free"):
            H = sim_oqs.evo_obj_free
            if hasattr(H, "eigenenergies"):
                H_eigvals = H.eigenenergies()
                energy_scale = np.max(np.abs(H_eigvals))
                diagnostics["system_info"]["hamiltonian"] = {
                    "energy_scale": float(energy_scale),
                    "max_eigenvalue": float(H_eigvals.max()),
                    "min_eigenvalue": float(H_eigvals.min()),
                }

                ### Check if dt is appropriate for energy scale
                if dt * energy_scale > 1.0:
                    diagnostics["issues_found"].append(
                        f"Time step too large for energy scale: dt*E_max = {dt * energy_scale:.3f}"
                    )
                    diagnostics["recommendations"].append(
                        f"Reduce dt below {1.0/energy_scale:.6f} for this energy scale"
                    )
    except Exception as e:
        diagnostics["issues_found"].append(f"Could not analyze Hamiltonian: {e}")

    ### Check decay operators
    if hasattr(sim_oqs, "decay_channels") and sim_oqs.decay_channels:
        diagnostics["system_info"]["decay_channels"] = len(sim_oqs.decay_channels)

        ### Basic sanity check on collapse operators
        try:
            for i, c_op in enumerate(sim_oqs.decay_channels):
                if not isinstance(c_op, Qobj):
                    diagnostics["issues_found"].append(
                        f"Decay channel {i} is not a Qobj"
                    )
        except Exception as e:
            diagnostics["issues_found"].append(f"Error checking decay channels: {e}")

    ### Solver-specific recommendations
    if solver_type == "BR":
        diagnostics["recommendations"].append(
            "BR solver: Ensure your system is in the Born-Markov regime"
        )
        diagnostics["recommendations"].append(
            "BR solver: Check that secular approximation is valid"
        )
    elif solver_type in ["me", "mesolve"]:
        diagnostics["recommendations"].append(
            "Mesolve: Consider using adaptive time stepping"
        )

    ### Check RWA settings
    if getattr(sim_oqs.simulation_config, "rwa_sl", False):
        diagnostics["system_info"]["rwa_enabled"] = True
        diagnostics["recommendations"].append(
            "RWA enabled: Verify that rotating wave approximation is appropriate"
        )

    return diagnostics


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
    # evolution_options.update(kwargs)  # Allow override of options

    evolution_data = compute_pulse_evolution(sim_oqs=sim_oqs, **evolution_options)

    return evolution_data.final_state


def compute_1d_polarization(
    sim_oqs: SimulationModuleOQS,
    **kwargs,
) -> Union[
    Tuple[np.ndarray, np.ndarray, SimulationModuleOQS], Tuple[np.ndarray], np.ndarray
]:
    """
    Compute the data for a fixed t_coh and t_wait.

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Simulation configuration object containing all parameters
    **kwargs : dict
        Additional keyword arguments:
        - plot_example_evo : bool, return evolution data for plotting
        - plot_example_polarization : bool, return polarization plot data
        - time_cut : float, time cutoff for detection

    Returns
    -------
    Union[Tuple, np.ndarray]
        - If plot_example_evo=True: (times, expectation_values, sim_oqs)
        - If plot_example_polarization=True: (polarization_plot_data,)
        - Otherwise: nonlinear_signal array
    """

    if kwargs.get("plot_example_evo", False):

        data = compute_pulse_evolution(sim_oqs=sim_oqs, store_states=True)
        states = data.states
        times = data.times
        dipole_op = sim_oqs.system.dipole_op
        n_atoms = sim_oqs.system.n_atoms
        e_ops = sim_oqs.observable_ops
        rwa_sl = sim_oqs.simulation_config.rwa_sl
        omega_laser = sim_oqs.laser._carrier_freq_fs
        datas = get_expect_vals_with_RWA(
            states,
            times,
            n_atoms=n_atoms,
            e_ops=e_ops,
            omega_laser=omega_laser,
            rwa_sl=rwa_sl,
            dipole_op=dipole_op,
        )
        return times, datas, sim_oqs

    # =============================
    # COMPUTE EVOLUTION STATES
    # =============================
    evolution_data = _compute_n_pulse_evolution(sim_oqs)

    # =============================
    # COMPUTE LINEAR SIGNALS
    # =============================
    linear_signals = _compute_n_linear_signals(sim_oqs)

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


def _compute_n_pulse_evolution(
    sim_oqs: SimulationModuleOQS,
) -> List[Qobj]:
    """Compute the n-pulse evolution using segmented approach."""
    laser = sim_oqs.laser
    peak_times = laser.pulse_peak_times
    fwhms = laser.pulse_fwhms
    psi_ini = sim_oqs.system.psi_ini
    times = sim_oqs.times_local

    n_pulses = len(laser.pulses)

    logger.info(
        f"  times_local: {len(times)} points from {times[0]:.2f} to {times[-1]:.2f} fs"
    )
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
        times_segment = _ensure_valid_times(times_segment, times, prev_pulse_start_idx)
        logger.info(
            f"  Segment {pulse_idx}: {len(times_segment)} points from {times_segment[0]:.2f} to {times_segment[-1]:.2f} fs"
        )

        # Update simulation parameters for this segment
        sim_oqs.times_local = times_segment

        # Compute evolution for this segment
        current_state = _compute_next_start_point(sim_oqs=sim_oqs)
        sim_oqs.system.psi_ini = current_state

        # Update for next iteration
        prev_pulse_start_idx = next_pulse_start_idx

    times_final = _ensure_valid_times(
        times[prev_pulse_start_idx:], times, prev_pulse_start_idx
    )
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
            final_states.append(
                final_states[-1] if final_states else sim_oqs.system.psi_ini
            )
    else:
        final_states = data_final.states[-detection_length:]

    # Debug: Check the states being returned
    logger.debug(f"Total states computed: {len(data_final.states)}")
    logger.debug(f"Detection length expected: {detection_length}")
    logger.debug(f"Final states extracted: {len(final_states)}")

    sim_oqs.reset_times_local()  # restore original times
    sim_oqs.system.psi_ini = psi_ini  # restore initial state

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
    # Check if the time segment has no elements
    if times_segment.size == 0:
        # First fallback: try to use a single time point at fallback_idx
        if fallback_idx < len(full_times):
            logger.warning(
                f"times_segment is empty. Using fallback time at index {fallback_idx}: {full_times[fallback_idx]}"
            )
            return np.array([full_times[fallback_idx]])
        logger.warning(
            f"times_segment is empty. Using first two time points from full array: {full_times[:2]}"
        )
        # Second fallback: use first two time points from full array
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
    logger.debug(
        f"evolution_data length: {len(evolution_data) if evolution_data else 'None'}"
    )
    logger.debug(
        f"actual_det_times shape: {actual_det_times.shape if hasattr(actual_det_times, 'shape') else len(actual_det_times)}"
    )
    logger.debug(
        f"detection_times shape: {detection_times.shape if hasattr(detection_times, 'shape') else len(detection_times)}"
    )

    # Apply RWA phase factors if needed
    if sim_oqs.simulation_config.rwa_sl:
        n_atoms = sim_oqs.system.n_atoms
        omega_laser = sim_oqs.laser._carrier_freq_fs
        evolution_data = apply_RWA_phase_factors(
            evolution_data, actual_det_times, n_atoms, omega_laser
        )
        for key in linear_signals:
            linear_signals[key] = apply_RWA_phase_factors(
                linear_signals[key], actual_det_times, n_atoms, omega_laser
            )

    # Calculate polarizations
    dipole_op = sim_oqs.system.dipole_op
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
                    nonlinear_signal = np.zeros(
                        np.sum(valid_time_mask), dtype=np.complex64
                    )
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
    current_length = (
        len(nonlinear_signal) if hasattr(nonlinear_signal, "__len__") else 1
    )

    logger.debug(
        f"Expected length: {expected_length}, current length: {current_length}"
    )

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
        logger.warning(
            f"Truncating signal from {current_length} to {expected_length} points"
        )
        nonlinear_signal = nonlinear_signal[:expected_length]

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

        local_sim_oqs.system.frequencies_cm = (
            new_freqs  # Update frequencies in the local copy
        )

        data = compute_1d_polarization(
            sim_oqs=local_sim_oqs,
            **kwargs,
        )

        # Ensure we return the correct type and shape
        if data is None:
            logger.error("compute_1d_polarization returned None")
            return None

        if not isinstance(data, np.ndarray):
            logger.error(
                f"compute_1d_polarization returned {type(data)}, expected np.ndarray"
            )
            return None

        # Additional shape validation
        expected_length = len(local_sim_oqs.times_det)
        if data.shape != (expected_length,):
            logger.error(
                f"Data shape {data.shape} doesn't match expected ({expected_length},)"
            )
            return None

        return data

    except Exception as e:
        import traceback

        logger.error(f"Error in _process_single_1d_combination: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def parallel_compute_1d_E_with_inhomogenity(
    sim_oqs: SimulationModuleOQS,
    parallel: bool = True,
    **kwargs: dict,
) -> np.ndarray:
    """
    Compute 1D COMPLEX polarization with frequency loop and phase cycling for IFT processing.
    Parallelizes over all frequency and phase combinations, averages, then performs IFT.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments for compute_1d_polarization.

    Returns
    -------
    np.ndarray
        photon_echo_signal where signal is averaged and IFT-processed.
    """
    # Configure phase cycling
    n_phases = sim_oqs.simulation_config.n_phases
    n_freqs = sim_oqs.simulation_config.n_freqs
    max_workers = sim_oqs.simulation_config.max_workers
    phases = np.linspace(0, 1, n_phases)  # placeholder simple phase grid
    if n_phases != 4:
        logger.warning(
            f"Phase cycling with {n_phases} phases may not be optimal for IFT"
        )

    # Sample frequency offsets for inhomogeneous broadening
    delta_cm = sim_oqs.system.delta_cm
    frequencies_cm = sim_oqs.system.frequencies_cm

    # Each row = one realization, each column = atom index
    # Shape: (n_freqs, n_atoms)
    all_freq_sets = np.stack(
        [sample_from_gaussian(n_freqs, delta_cm, freq) for freq in frequencies_cm],
        axis=1,
    )
    print(f"Using frequency samples ={all_freq_sets}", flush=True)

    # Prepare all jobs: one per (omega_idx, phi1_idx, phi2_idx)
    combinations = []
    for omega_idx in range(n_freqs):
        new_freqs = all_freq_sets[omega_idx]
        for phi1_idx, phi1 in enumerate(phases):
            for phi2_idx, phi2 in enumerate(phases):
                combinations.append(
                    (omega_idx, phi1_idx, phi2_idx, new_freqs, phi1, phi2)
                )

    data_length = len(sim_oqs.times_det)
    results_array = np.empty(
        (n_freqs, n_phases, n_phases, data_length), dtype=np.complex64
    )

    if not parallel:
        for omega_idx in range(n_freqs):
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
                for (
                    omega_idx,
                    phi1_idx,
                    phi2_idx,
                    new_freqs,
                    phi1,
                    phi2,
                ) in combinations
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
                    logger.error(
                        f"Combination ({omega_idx},{phi1_idx},{phi2_idx}) failed: {exc}"
                    )
                    results_array[omega_idx, phi1_idx, phi2_idx, :] = np.nan

    # Average over frequencies to get 2D result array before IFT or phase-average
    results_matrix_avg = np.mean(results_array, axis=0)

    logger.debug(f"Results matrix before IFT: {results_matrix_avg}")
    # Final IFT extraction for the specified component
    signal_type = sim_oqs.simulation_config.signal_type
    if signal_type == "rephasing":
        components = [(-1, 1, 1)]
    elif signal_type == "non-rephasing":
        components = [(1, -1, 1)]
    elif signal_type == "absorptive":
        components = [(-1, 1, 1), (1, -1, 1)]
    else:
        raise ValueError(f"Unsupported signal_type: {signal_type}")

    extracted = [
        extract_ift_signal_component(
            results_matrix=results_matrix_avg, phases=phases, component=c
        )
        for c in components
    ]
    if signal_type == "absorptive":
        rephasing_sig, nonrephasing_sig = extracted
        logger.debug(
            f"Final signals after IFT (absorptive request): rephasing shape={rephasing_sig.shape}, non-rephasing shape={nonrephasing_sig.shape}"
        )
        return rephasing_sig, nonrephasing_sig
    else:
        photon_echo_signal = extracted[0]
    logger.debug(f"Final signal after IFT ({signal_type}): {photon_echo_signal}")

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
        logger.error("No valid results found in the results matrix")
        return None

    signal = np.zeros_like(first_valid_result, dtype=np.complex64)

    for phi1_idx, phi_1 in enumerate(phases):
        for phi2_idx, phi_2 in enumerate(phases):
            if results_matrix[phi1_idx, phi2_idx] is not None:
                phase_factor = np.exp(-1j * (l * phi_1 + m * phi_2 + n * 0.0))
                signal += results_matrix[phi1_idx, phi2_idx] * phase_factor

    # signal /= n_phases * n_phases # TODO to get more prominent signal leave out

    return signal
