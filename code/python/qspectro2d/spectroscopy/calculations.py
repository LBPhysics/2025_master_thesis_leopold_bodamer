# -*- coding: utf-8 -*-

# =============================
# STANDARD LIBRARY IMPORTS
# =============================
from copy import deepcopy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Union, Tuple
import logging

# =============================
# THIRD-PARTY IMPORTS
# =============================
import numpy as np
from qutip import Qobj, Result, liouvillian, mesolve, brmesolve, expect
from qutip.core import QobjEvo

# =============================
# LOCAL IMPORTS
# =============================
from qspectro2d.core.system_parameters import SystemParameters
from qspectro2d.core.pulse_sequences import PulseSequence
from qspectro2d.core.pulse_functions import (
    identify_non_zero_pulse_regions,
    split_by_active_regions,
)
from qspectro2d.core.solver_fcts import (
    matrix_ODE_paper,
    R_paper,
)
from qspectro2d.spectroscopy.inhomogenity import sample_from_sigma
from qspectro2d.core.functions_with_rwa import (
    H_int,
    apply_RWA_phase_factors,
)


# =============================
# CONSTANTS AND CONFIGURATION
# =============================
DEFAULT_SOLVER_OPTIONS = {
    "nsteps": 200000,
    "atol": 1e-6,
    "rtol": 1e-4,
}

SUPPORTED_SOLVERS = ["ME", "BR", "Paper_eqs", "Paper_BR"]
PHASE_CYCLING_PHASES = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
IFT_TOLERANCE = 1e-3

# =============================
# LOGGING CONFIGURATION
# =============================
logger = logging.getLogger(__name__)


# =============================
# VALIDATION HELPERS
# =============================
def _validate_computation_inputs(
    times: np.ndarray,
    system: SystemParameters,
    n_freqs: int = None,
    n_phases: int = None,
) -> None:
    """Validate common inputs for computation functions."""
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be a numpy array")

    if len(times) < 2:
        raise ValueError("times array must have at least 2 elements")

    if not hasattr(system, "ODE_Solver"):
        raise AttributeError("system must have ODE_Solver attribute")

    if n_freqs is not None and n_freqs <= 0:
        raise ValueError("n_freqs must be positive")

    if n_phases is not None and n_phases not in [2, 4, 8]:
        logger.warning(
            f"n_phases={n_phases} may not be optimal. Consider using 2, 4, or 8."
        )


def _validate_system_state(system: SystemParameters) -> None:
    """Validate system parameters for consistency."""
    required_attrs = ["psi_ini", "observable_ops", "H0_diagonalized"]

    for attr in required_attrs:
        if not hasattr(system, attr):
            raise AttributeError(f"system missing required attribute: {attr}")

    if not isinstance(system.psi_ini, Qobj):
        raise TypeError("psi_ini must be a Qobj")


# =============================
# POLARIZATION CALCULATIONS
# =============================
def complex_polarization(
    system: SystemParameters, state: Union[Qobj, List[Qobj]]
) -> Union[complex, np.ndarray]:
    """
    Calculate the complex polarization for state(s) using the dipole operator.
    The polarization is defined as one part of the expectation value of the dipole operator
    with the given quantum state(s) or density matrix(es).

    Parameters
    ----------
    system : SystemParameters
        System parameters containing dipole operator information.
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
    >>> pol = complex_polarization(system, rho)  # Single density matrix
    >>> pols = complex_polarization(system, [rho1, rho2])  # Multiple states
    """
    if isinstance(state, Qobj):
        return _single_qobj_polarization(system, state)

    if isinstance(state, list):
        return np.array(
            [_single_qobj_polarization(system, s) for s in state],
            dtype=np.complex128,  # Use higher precision
        )

    raise TypeError(f"State must be a Qobj or list of Qobj, got {type(state)}")


def _single_qobj_polarization(system: SystemParameters, state: Qobj) -> complex:
    """
    Calculate polarization for a single quantum state or density matrix.

    Parameters
    ----------
    system : SystemParameters
        System parameters containing dipole operator.
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
    ValueError
        If number of atoms is not supported.
    """
    if not (state.isket or state.isoper):
        raise TypeError("State must be a ket or density matrix")

    dip_op = system.Dip_op
    n_atoms = system.N_atoms

    # Dispatch to specialized methods for better performance
    if n_atoms == 1:
        return _single_atom_polarization(dip_op, state)
    elif n_atoms == 2:
        return _two_atom_polarization(dip_op, state)
    else:
        return _multi_atom_polarization(dip_op, state)


def _single_atom_polarization(dip_op: Qobj, state: Qobj) -> complex:
    """Calculate polarization for single atom system."""
    return dip_op[1, 0] * state[0, 1]


def _two_atom_polarization(dip_op: Qobj, state: Qobj) -> complex:
    """Calculate polarization for two-atom system."""
    return (
        dip_op[1, 0] * state[0, 1]
        + dip_op[2, 0] * state[0, 2]
        + dip_op[3, 1] * state[1, 3]
        + dip_op[3, 2] * state[2, 3]
    )


def _multi_atom_polarization(dip_op: Qobj, state: Qobj) -> complex:
    """Calculate polarization for multi-atom system (N > 2)."""
    polarization = 0j
    dim = state.shape[0]

    for i in range(dim):
        for j in range(dim):
            if i != j and abs(dip_op[i, j]) > 0:  # Fixed: use 0 not IFT_TOLERANCE
                polarization += dip_op[i, j] * state[j, i]

    return polarization


def compute_pulse_evolution(
    psi_ini: Qobj,
    times: np.ndarray,
    pulse_seq: PulseSequence,
    system: SystemParameters,
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
    pulse_seq : PulseSequence
        PulseSequence object defining the pulse sequence.
    system : SystemParameters
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
    options = _configure_solver_options(solver_options)

    # =============================
    # VALIDATE SOLVER TYPE
    # =============================
    if system.ODE_Solver not in SUPPORTED_SOLVERS:
        raise ValueError(
            f"Unknown ODE solver: {system.ODE_Solver}. "
            f"Supported solvers: {SUPPORTED_SOLVERS}"
        )

    # =============================
    # INITIALIZE EVOLUTION COMPONENTS
    # =============================
    current_state = psi_ini

    # Build Hamiltonian components that already include the RWA if present
    H_free = system.H0_diagonalized
    H_int_evo = QobjEvo(lambda t, args=None: H_free + H_int(t, pulse_seq, system))

    # =============================
    # CONFIGURE SOLVER-SPECIFIC PARAMETERS
    # =============================
    if system.ODE_Solver == "Paper_eqs":
        _configure_paper_equations_solver(system)

    # =============================
    # SPLIT EVOLUTION BY PULSE REGIONS
    # =============================
    return _execute_segmented_evolution(
        times, pulse_seq, system, current_state, H_free, H_int_evo, options
    )


def _execute_segmented_evolution(  # TODO during "final_state" solving -> take coarser time steps
    times: np.ndarray,
    pulse_seq: PulseSequence,
    system: SystemParameters,
    current_state: Qobj,
    H_free: Qobj,
    H_int_evo: QobjEvo,
    options: dict,
) -> Result:
    """Execute evolution split by pulse regions."""
    all_states, all_times = [], []

    # Find pulse regions and split time array
    pulse_regions = identify_non_zero_pulse_regions(times, pulse_seq)
    split_times = split_by_active_regions(times, pulse_regions)

    for i, times_ in enumerate(split_times):
        # Extend times_ by one point if not the last segment
        if i < len(split_times) - 1:
            next_times = split_times[i + 1]
            if len(next_times) > 0:
                times_ = np.append(times_, next_times[0])

        # Find the indices in the original times array for this split
        start_idx = np.abs(times - times_[0]).argmin()
        has_pulse = pulse_regions[start_idx]

        # Configure evolution object and collapse operators
        EVO_obj, c_ops_list = _configure_evolution_objects(
            system, has_pulse, H_free, H_int_evo, pulse_seq
        )

        # Execute evolution for this time segment
        result = _execute_single_evolution_segment(
            system, EVO_obj, c_ops_list, current_state, times_, H_int_evo, options
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


def _configure_evolution_objects(
    system: SystemParameters,
    has_pulse: bool,
    H_free: Qobj,
    H_int_evo: QobjEvo,
    pulse_seq: PulseSequence,
) -> tuple[Union[Qobj, QobjEvo], list]:
    """Configure evolution objects and collapse operators based on solver type.

    Returns
    -------
    tuple[Union[Qobj, QobjEvo], list]
        (evolution_object, collapse_operators_list)
    """
    if system.ODE_Solver == "ME":
        if has_pulse:
            return (
                H_int_evo,
                system.me_decay_channels,
            )
        else:
            return (
                H_free,
                system.me_decay_channels,
            )
    elif system.ODE_Solver == "BR":
        # BR solver handles a_ops, it is handled in _execute_single_evolution_segment
        return (
            H_int_evo,
            [],
        )
    elif system.ODE_Solver == "Paper_BR":
        # Paper BR combines Liouvillian with custom R operator
        EVO_obj = liouvillian(H_int_evo) + R_paper(
            system
        )  # TODO THIS IS WRONG UNFORTUNATELY
        return EVO_obj, []
    elif system.ODE_Solver == "Paper_eqs":
        # Paper equations use custom matrix ODE
        EVO_obj = QobjEvo(lambda t, args=None: matrix_ODE_paper(t, pulse_seq, system))
        return EVO_obj, []


def _execute_single_evolution_segment(
    system: SystemParameters,
    EVO_obj: Union[Qobj, QobjEvo],
    c_ops_list: list,
    current_state: Qobj,
    times_: np.ndarray,
    H_int_evo: QobjEvo,
    options: dict,
) -> Result:
    """Execute evolution for a single time segment."""
    if system.ODE_Solver == "BR":
        return brmesolve(
            H_int_evo,
            current_state,
            times_,
            a_ops=system.br_decay_channels,
            options=options,
        )
    else:
        return mesolve(
            EVO_obj,
            current_state,
            times_,
            c_ops=c_ops_list,
            options=options,
        )


def check_the_solver(system: SystemParameters) -> tuple[Result, float]:
    """
    Checks the solver within the compute_pulse_evolution function
    with the provided psi_ini, times, and system.

    Parameters:
        system (System): System object containing all relevant parameters, including observable_ops.
        PulseSequence (type): The PulseSequence class to construct pulse sequences.

    Returns:
    tuple of:
        result (Result): The result object from compute_pulse_evolution.
        time_cut (float): The time after which the checks failed, or np.inf if all checks passed.
    """
    t_max = 2 * system.t_max
    dt = 10 * system.dt
    t0 = -system.fwhms[0]
    times = np.arange(t0, t_max, dt)

    print(f"Checking '{system.ODE_Solver}' solver ", flush=True)

    # =============================
    # INPUT VALIDATION
    # =============================
    if not hasattr(system, "ODE_Solver"):
        raise AttributeError("system must have attribute 'ODE_Solver'")
    if not hasattr(system, "observable_ops"):
        raise AttributeError("system must have attribute 'observable_ops'")
    if not isinstance(system.psi_ini, Qobj):
        raise TypeError("psi_ini must be a Qobj")
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be a numpy.ndarray")
    if not isinstance(system.observable_ops, list) or not all(
        isinstance(op, Qobj) for op in system.observable_ops
    ):
        raise TypeError("system.observable_ops must be a list of Qobj")
    if len(times) < 2:
        raise ValueError("times must have at least two elements")

    # =============================
    # CONSTRUCT PULSE SEQUENCE (refactored)
    # =============================
    # Define pulse parameters
    phi_0 = np.pi / 2
    phi_1 = np.pi / 4
    DETECTION_PHASE = 0
    pulse1_t_peak = times[-1] / 2
    pulse2_t_peak = times[-1] / 1.1

    # Use the from_pulse_specs static method to construct the sequence
    pulse_seq = PulseSequence.from_pulse_specs(
        system=system,
        pulse_specs=[
            (0, 0, phi_0),  # pulse 0 at t=0
            (1, pulse1_t_peak, phi_1),  # pulse 1 at middle time
            (2, pulse2_t_peak, DETECTION_PHASE),  # pulse 2 at end time
        ],
    )

    result = compute_pulse_evolution(
        system.psi_ini, times, pulse_seq, system=system, **{"store_states": True}
    )
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
    if getattr(system, "RWA_laser", False):
        states = apply_RWA_phase_factors(states, times, system)
    for index, state in enumerate(states):
        time = times[index]
        if not state.isherm:
            strg += f"Density matrix is not Hermitian after t = {time}.\n"
            print(state, flush=True)
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
            print(strg, flush=True)
            break
    else:
        print(
            "Checks passed. DM remains Hermitian and positive.",
            flush=True,
        )

    return result, time_cut


def _compute_next_start_point(
    psi_initial: Qobj,
    times: np.ndarray,
    pulse_specs: List,
    system: SystemParameters,
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

    system : SystemParameters
        System parameters object.
    Returns
    -------
    Qobj
        Final quantum state after pulse evolution.

    """

    if times.size == 0:
        raise ValueError("Times array cannot be empty")

    # =============================
    # PULSE SEQUENCE CREATION
    # =============================
    pulse_seq = PulseSequence.from_pulse_specs(system=system, pulse_specs=pulse_specs)

    # =============================
    # EVOLUTION COMPUTATION
    # =============================
    evolution_options = {
        "store_final_state": True,
        "store_states": False,  # Only need final state for efficiency
    }
    evolution_options.update(kwargs)  # Allow override of options

    evolution_data = compute_pulse_evolution(
        psi_initial, times, pulse_seq, system=system, **evolution_options
    )

    return evolution_data.final_state


def _create_single_pulse_sequence(system, pulse_idx, t_peak, phase):
    """Create a pulse sequence with a single pulse."""
    return PulseSequence.from_pulse_specs(
        system=system, pulse_specs=[(pulse_idx, t_peak, phase)]
    )


def compute_1d_polarization(
    tau_coh: float,
    T_wait: float,
    phi_0: float,
    phi_1: float,
    t_det_max: np.ndarray,
    system: SystemParameters,  # contains frequency omega
    **kwargs,
) -> tuple:
    """
    Compute the data for a fixed tau_coh and T_wait. AND NOW VARIABLE t_det_max
    """
    time_cut = kwargs.get("time_cut", np.inf)

    dt = system.dt
    t0 = -system.fwhms[0]
    times = np.arange(t0, tau_coh + T_wait + t_det_max, dt)
    # Force 1d times to a canonical grid
    t_det_values = np.arange(0, t_det_max, system.dt)

    # =============================
    # PULSE TIMING AND SEQUENCES
    # =============================
    pulse_timings = _get_pulse_timings(tau_coh, T_wait)
    pulse_sequences = _create_pulse_sequences(system, phi_0, phi_1, pulse_timings)

    # =============================
    # COMPUTE EVOLUTION STATES
    # =============================
    evolution_data = _compute_three_pulse_evolution(
        times, system, pulse_timings, pulse_sequences["full"]
    )

    # =============================
    # COMPUTE LINEAR SIGNALS
    # =============================
    linear_signals = _compute_linear_signals(
        times, system, pulse_sequences["individual"], pulse_timings["detection"]
    )

    # =============================
    # EXTRACT AND PROCESS DETECTION DATA
    # =============================
    detection_data = _extract_detection_data(
        evolution_data,
        linear_signals,
        pulse_timings["detection"],
        time_cut,
        t_det_values,
        system,
    )

    # Return based on plotting flag
    if kwargs.get("plot_example_polarization", False):
        return detection_data["plot_data"]

    return detection_data["times"], detection_data["nonlinear_signal"]


def _get_pulse_timings(tau_coh: float, T_wait: float) -> dict:
    """Extract pulse timing information."""
    return {
        "pulse0": 0,
        "pulse1": tau_coh,
        "detection": tau_coh + T_wait,
    }


def _create_pulse_sequences(
    system: SystemParameters, phi_0: float, phi_1: float, timings: dict
) -> dict:
    """Create all required pulse sequences."""
    DETECTION_PHASE = 0  # Fixed phase for detection pulse

    # Full three-pulse sequence specs
    full_specs = [
        (0, timings["pulse0"], phi_0),
        (1, timings["pulse1"], phi_1),
        (2, timings["detection"], DETECTION_PHASE),
    ]

    # Individual pulse sequences for linear signal subtraction
    individual_sequences = {
        "pulse0": _create_single_pulse_sequence(system, 0, timings["pulse0"], phi_0),
        "pulse1": _create_single_pulse_sequence(system, 1, timings["pulse1"], phi_1),
        "pulse2": _create_single_pulse_sequence(
            system, 2, timings["detection"], DETECTION_PHASE
        ),
    }

    return {
        "full": PulseSequence.from_pulse_specs(system=system, pulse_specs=full_specs),
        "individual": individual_sequences,
    }


def _compute_three_pulse_evolution(
    times: np.ndarray,
    system: SystemParameters,
    timings: dict,
    full_sequence: PulseSequence,
) -> dict:
    """Compute the three-pulse evolution using segmented approach."""
    # Segment 1: First pulse evolution
    pulse1_start_idx = np.abs(times - (timings["pulse1"] - system.fwhms[1])).argmin()
    times_0 = _ensure_valid_times(times[: pulse1_start_idx + 1], times)

    pulse0_specs = (0, timings["pulse0"], full_sequence.pulse_specs[0][2])
    rho_1 = _compute_next_start_point(
        psi_initial=system.psi_ini,
        times=times_0,
        pulse_specs=[pulse0_specs],
        system=system,
    )

    # Segment 2: Second pulse evolution
    pulse2_start_idx = np.abs(times - (timings["detection"] - system.fwhms[2])).argmin()
    times_1 = times[pulse1_start_idx : pulse2_start_idx + 1]
    times_1 = _ensure_valid_times(times_1, times, pulse1_start_idx)

    pulse1_specs = (1, timings["pulse1"], full_sequence.pulse_specs[1][2])
    rho_2 = _compute_next_start_point(
        psi_initial=rho_1,
        times=times_1,
        pulse_specs=[pulse0_specs, pulse1_specs],
        system=system,
    )

    # Segment 3: Final evolution with detection
    times_2 = _ensure_valid_times(times[pulse2_start_idx:], times, pulse2_start_idx)
    data_final = compute_pulse_evolution(
        rho_2, times_2, full_sequence, system=system, store_states=True
    )

    return {
        "final_data": data_final,
        "times_2": times_2,
        "detection_start_idx": pulse2_start_idx,
    }


def _compute_linear_signals(
    times: np.ndarray,
    system: SystemParameters,
    pulse_sequences: dict,
    detection_time: float,
) -> dict:
    """Compute all linear signal contributions."""
    linear_data = {}
    detection_idx = np.abs(times - detection_time).argmin()

    for pulse_name, sequence in pulse_sequences.items():
        data = compute_pulse_evolution(
            system.psi_ini, times, sequence, system=system, store_states=True
        )
        linear_data[pulse_name] = data.states[detection_idx:]

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
    system: SystemParameters,
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
        P_values = complex_polarization(system, states)

        # Handle scalar vs array results
        if np.isscalar(P_values):
            P_array[valid_indices[0]] = P_values
        else:
            for idx, orig_idx in enumerate(valid_indices):
                P_array[orig_idx] = P_values[idx]

        polarizations[key] = P_array

    return polarizations


def _extract_detection_data(
    evolution_data: dict,
    linear_signals: dict,
    detection_time: float,
    time_cut: float,
    t_det_values: np.ndarray,
    system: SystemParameters,
) -> dict:
    """Extract and process detection time data."""
    # =============================
    # DEBUG PRINTS FOR ARRAY LENGTHS
    # =============================
    final_data = evolution_data["final_data"]
    detection_start_idx = np.abs(final_data.times - detection_time).argmin()
    actual_det_times = final_data.times[detection_start_idx:]
    states_full = final_data.states[detection_start_idx:]

    # Apply RWA phase factors if needed
    if system.RWA_laser:
        states_full = apply_RWA_phase_factors(states_full, actual_det_times, system)
        for key in linear_signals:
            linear_signals[key] = apply_RWA_phase_factors(
                linear_signals[key], actual_det_times, system
            )

    # Calculate polarizations
    polarizations = _calculate_all_polarizations(
        states_full, linear_signals, actual_det_times, time_cut, system
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
    # 1D linear interpolation to match the canonical grid
    data_interp = np.interp(
        t_det_values, actual_det_times - actual_det_times[0], nonlinear_signal.real
    ) + 1j * np.interp(
        t_det_values, actual_det_times - actual_det_times[0], nonlinear_signal.imag
    )

    return {
        "times": t_det_values,
        "nonlinear_signal": data_interp,
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
    phi1: float,
    phi2: float,
    tau_coh: float,
    T_wait: float,
    t_det_max: np.ndarray,
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
            t_det_max=t_det_max,
            system=system,
            **kwargs,
        )

        return t_det_vals, data

    except Exception as e:
        print(f"Error in _process_single_1d_combination: {str(e)}", flush=True)
        return None


def parallel_compute_1d_E_with_inhomogenity(
    n_freqs: int,
    n_phases: int,
    tau_coh: float,
    T_wait: float,
    t_det_max: np.ndarray,
    system,
    max_workers: int = None,
    apply_ift: bool = True,
    ift_component: tuple = (-1, 1, 0),
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D COMPLEX polarization with frequency loop and phase cycling for IFT processing.
    Parallelizes over all frequency and phase combinations, averages, then performs IFT.

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
    t_det_max : np.ndarray
        Detection time values.
    system : SystemParameters
        System parameters object.
    max_workers : int, optional
        Number of workers for parallel processing.
    **kwargs : dict
        Additional keyword arguments for compute_1d_polarization.

    Returns
    -------
    tuple
        (t_det_vals, photon_echo_signal) where signal is averaged and IFT-processed.
    """
    # Configure parallel processing
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Limit to 8 to avoid memory issues

    # Configure phase cycling
    phases = PHASE_CYCLING_PHASES[:n_phases]  # Use predefined phases
    if n_phases != 4:
        logger.warning(
            f"Phase cycling with {n_phases} phases may not be optimal for IFT"
        )

    logger.info(
        f"Processing {n_freqs} frequencies with {n_phases}×{n_phases} phase combinations"
    )
    logger.info(f"Using {max_workers} parallel workers")

    # Sample frequency offsets for inhomogeneous broadening
    N_atoms = system.N_atoms
    if N_atoms == 1:
        freq_samples = sample_from_sigma(n_freqs, system.Delta_cm, system.omega_A_cm)
    elif N_atoms == 2:
        freq_samples = (
            sample_from_sigma(n_freqs, system.Delta_cm, system.omega_A_cm),
            sample_from_sigma(n_freqs, system.Delta_cm, system.omega_B_cm),
        )
    else:
        raise ValueError("Unsupported number of atoms")
    print(f"Using frequency samples ={freq_samples}")

    # Prepare all jobs: one per (omega_idx, phi1_idx, phi2_idx)
    combinations = []
    for omega_idx in range(n_freqs):
        if N_atoms == 1:
            new_freqs = freq_samples[omega_idx]
        elif N_atoms == 2:
            new_freqs = (freq_samples[0][omega_idx], freq_samples[1][omega_idx])

        for phi1_idx, phi1 in enumerate(phases):
            for phi2_idx, phi2 in enumerate(phases):
                sys_copy = deepcopy(system)
                if N_atoms == 1:
                    sys_copy.omega_A_cm = new_freqs
                elif N_atoms == 2:
                    sys_copy.omega_A_cm = new_freqs[0]
                    sys_copy.omega_B_cm = new_freqs[1]
                combinations.append(
                    (omega_idx, phi1_idx, phi2_idx, sys_copy, phi1, phi2)
                )

    # Execute all jobs in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_1d_combination,
                phi1=phi1,
                phi2=phi2,
                tau_coh=tau_coh,
                T_wait=T_wait,
                t_det_max=t_det_max,
                system=sys_copy,
                **kwargs,
            ): (omega_idx, phi1_idx, phi2_idx)
            for (omega_idx, phi1_idx, phi2_idx, sys_copy, phi1, phi2) in combinations
        }

        t_det_vals = None
        for future in as_completed(futures):
            omega_idx, phi1_idx, phi2_idx = futures[future]
            try:
                t_det_vals, data = future.result()
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

    # Average over frequencies before IFT or phase-average
    results_matrix_avg = np.mean(results_cube, axis=0)

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

    return t_det_vals, photon_echo_signal


def parallel_compute_2d_E_with_inhomogenity(
    n_freqs: int,
    n_phases: int,
    T_wait: float,
    t_det_max: np.ndarray,
    system,
    max_workers: int = None,
    apply_ift: bool = True,
    ift_component: tuple = (-1, 1, 0),
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D COMPLEX polarization signal with inhomogeneous broadening and phase cycling.

    Parameters
    ----------
    n_freqs : int
        Number of frequency samples for inhomogeneous averaging.
    n_phases : int
        Number of phases for phase cycling (typically 4).
    T_wait : float
        Waiting time between coherence and detection periods.
    t_det_max : int
        maximal time for detection. == maximal tau_coh
    system : SystemParameters
        Simulation parameters.
    max_workers : int, optional
        Number of workers for parallel processing (default: CPU count).
    apply_ift : bool, optional
        Whether to apply IFT signal extraction (default: True).
        If True, returns the specific signal component via extract_ift_signal_component.
        If False, returns the raw phase-averaged signal E = i*P.
    ift_component : tuple, optional
        Component to extract when apply_ift=True (default: (-1, 1, 0)).
    **kwargs : dict
        Additional arguments passed to the pulse sequence simulation function.

    Returns
    -------
    tuple
        tau_coh_vals : np.ndarray
            Array of coherence times τ.
        t_det_vals : np.ndarray
            Detection time axis.
        data_avg_2d : np.ndarray
            2D signal array of shape (len(tau_coh), len(t_det)), complex-valued.
            If apply_ift=True: specific IFT signal component.
            If apply_ift=False: phase-averaged raw signal E = i*P.
    """
    dt = system.dt
    tau_coh_vals = np.arange(0, t_det_max, dt)
    t_det_vals = None

    if max_workers is None:
        max_workers = mp.cpu_count()

    phases = [k * np.pi / 2 for k in range(n_phases)]
    if n_phases != 4:
        print(
            f"Warning: Phase cycling with {n_phases} phases is nonstandard", flush=True
        )

    print(
        f"Computing 2D signal for {n_freqs} frequencies × {len(tau_coh_vals)} τ × {n_phases}² phase combinations",
        flush=True,
    )
    print(f"Using {max_workers} parallel workers", flush=True)

    # --- Frequency samples ---
    N_atoms = system.N_atoms
    if N_atoms == 1:
        freq_samples = sample_from_sigma(n_freqs, system.Delta_cm, system.omega_A_cm)
    elif N_atoms == 2:
        freq_samples = (
            sample_from_sigma(n_freqs, system.Delta_cm, system.omega_A_cm),
            sample_from_sigma(n_freqs, system.Delta_cm, system.omega_B_cm),
        )
        print(f"Using frequency samples ={freq_samples}")
    else:
        raise ValueError("Unsupported number of atoms")

    # --- Build all jobs ---
    combinations = []
    for omega_idx in range(n_freqs):
        new_freqs = (
            freq_samples[omega_idx]
            if N_atoms == 1
            else (freq_samples[0][omega_idx], freq_samples[1][omega_idx])
        )
        for tau_coh_idx, tau_coh in enumerate(tau_coh_vals):
            for phi1_idx, phi1 in enumerate(phases):
                for phi2_idx, phi2 in enumerate(phases):
                    sys_copy = deepcopy(system)
                    if N_atoms == 1:
                        sys_copy.omega_A_cm = new_freqs
                    elif N_atoms == 2:
                        sys_copy.omega_A_cm = new_freqs[0]
                        sys_copy.omega_B_cm = new_freqs[1]
                    combinations.append(
                        (
                            omega_idx,
                            tau_coh_idx,
                            phi1_idx,
                            phi2_idx,
                            sys_copy,
                            phi1,
                            phi2,
                            tau_coh,
                        )
                    )

    # --- Run all jobs in parallel ---
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_1d_combination,
                phi1=phi1,
                phi2=phi2,
                tau_coh=tau_coh,
                T_wait=T_wait,
                t_det_max=t_det_max,
                system=sys_copy,
                **kwargs,
            ): (omega_idx, tau_coh_idx, phi1_idx, phi2_idx)
            for omega_idx, tau_coh_idx, phi1_idx, phi2_idx, sys_copy, phi1, phi2, tau_coh in combinations
        }

        for future in as_completed(futures):
            omega_idx, tau_idx, phi1_idx, phi2_idx = futures[future]
            try:
                t_det_vals, data = future.result()
                results[(omega_idx, tau_idx, phi1_idx, phi2_idx)] = data
            except Exception as exc:
                print(
                    f"Failed: ω={omega_idx}, τ={tau_idx}, φ=({phi1_idx},{phi2_idx}) → {exc}",
                    flush=True,
                )

    # Organize into 4D object array: (n_freqs, n_tau, n_phases, n_phases)
    n_tau = len(tau_coh_vals)
    signal_tensor = np.empty((n_freqs, n_tau, n_phases, n_phases), dtype=object)
    for (omega_idx, tau_idx, phi1_idx, phi2_idx), data in results.items():
        signal_tensor[omega_idx, tau_idx, phi1_idx, phi2_idx] = data * 1j  # E ~ iP

    # === Build final 2D signal: one row per τ, after averaging and IFT ===
    data_list_1d = []
    for tau_idx in range(n_tau):
        results_matrix = np.empty((n_phases, n_phases), dtype=object)
        for phi1_idx in range(n_phases):
            for phi2_idx in range(n_phases):
                # average over ω at fixed τ, φ1, φ2
                try:
                    results_matrix[phi1_idx, phi2_idx] = np.mean(
                        [
                            signal_tensor[f, tau_idx, phi1_idx, phi2_idx]
                            for f in range(n_freqs)
                        ],
                        axis=0,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Averaging failed at τ={tau_idx}, φ=({phi1_idx},{phi2_idx}): {e}"
                    )

        # Apply IFT or return raw signal based on flag
        if apply_ift:
            signal_2d = extract_ift_signal_component(
                results_matrix=results_matrix, phases=phases, component=ift_component
            )
        else:
            # Return phase-averaged raw signal E = i*P
            # Collect all valid phase combination arrays for this τ
            phase_signals = []
            for phi1_idx in range(n_phases):
                for phi2_idx in range(n_phases):
                    if results_matrix[phi1_idx, phi2_idx] is not None:
                        phase_signals.append(results_matrix[phi1_idx, phi2_idx])

            if phase_signals:
                # Stack and average along phase axis (axis=0) to preserve t_det dimensions
                signal_2d = np.mean(np.array(phase_signals), axis=0)
            else:
                # Fallback: create zero array with correct shape
                signal_2d = (
                    np.zeros_like(results_matrix[0, 0])
                    if results_matrix[0, 0] is not None
                    else np.array([])
                )
                print(
                    "Warning: No valid phase signals found, returning empty signal.",
                    flush=True,
                )

        data_list_1d.append(signal_2d)

    # Final stacking
    data_avg_2d = np.vstack(data_list_1d)
    return tau_coh_vals, t_det_vals, data_avg_2d


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
    l, m, n = component
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
        for phi2_idx, DETECTION_PHASE in enumerate(phases):
            if results_matrix[phi1_idx, phi2_idx] is not None:
                # IFT phase factor with parameterized coefficients
                phase_factor = np.exp(
                    -1j * (l * phi_1 + m * DETECTION_PHASE)
                )  # n * phi_3 is 0!
                signal += results_matrix[phi1_idx, phi2_idx] * phase_factor

    # Normalize by number of phase combinations
    signal /= n_phases * n_phases

    return signal


def _configure_solver_options(solver_options: dict) -> dict:
    """Configure solver options with defaults."""
    options = solver_options.copy() if solver_options else {}

    # Update options with defaults only if not already set
    for key, value in DEFAULT_SOLVER_OPTIONS.items():
        options.setdefault(key, value)

    return options


def _configure_paper_equations_solver(system: SystemParameters) -> None:
    """Configure system for paper equations solver."""
    if not system.RWA_laser:
        logger.info("Paper equations require RWA - enabling RWA_laser")
        system.RWA_laser = True
