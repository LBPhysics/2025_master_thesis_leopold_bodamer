# -*- coding: utf-8 -*-

from copy import deepcopy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Union, Tuple
import numpy as np
from qutip import Qobj, Result, liouvillian, mesolve, brmesolve, expect
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

    state : Qobj/array-like
        A single or a `list` of quantum states or density matrices.

    Returns
    -------
    polarization : complex/array-like
        Complex polarization value(s). A (nested) array of polarization values
        if ``state`` is an array.

    Examples
    --------
    >>> complex_polarization(system, rho) # Single density matrix
    >>> complex_polarization(system, [rho1, rho2, rho3]) # Multiple states

    """
    if isinstance(state, Qobj):
        return _single_qobj_polarization(system, state)

    elif isinstance(state, (list)):
        return np.array(
            [_single_qobj_polarization(system, s) for s in state], dtype=np.complex64
        )

    raise TypeError("State must be a quantum object or array of quantum objects")


def _single_qobj_polarization(system: SystemParameters, state: Qobj) -> complex:
    """
    Private function used by complex_polarization to calculate polarization values of Qobjs.
    """
    if not (state.isket or state.isoper):
        raise TypeError("State must be a ket or density matrix")

    Dip = system.Dip_op

    if system.N_atoms == 1:
        # For a single atom, the polarization is simply the expectation value of the dipole operator
        return Dip[1, 0] * state[0, 1]
    elif system.N_atoms == 2:
        return (
            Dip[1, 0] * state[0, 1]
            + Dip[2, 0] * state[0, 2]
            + Dip[3, 1] * state[1, 3]
            + Dip[3, 2] * state[2, 3]
        )
    else:
        # General case for N>2
        pol = 0j
        dim = state.shape[0]
        for i in range(dim):
            for j in range(dim):
                if i != j and abs(Dip[i, j]) > 0:
                    pol += Dip[i, j] * state[j, i]
        return pol


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
    # Set solver options
    # =============================
    # Initialize with provided solver_options or empty dict
    options = solver_options.copy() if solver_options else {}

    # Add default options if not already present
    default_options = {
        # Increasing max steps and atol/rtol for better stability
        "nsteps": 200000,
        "atol": 1e-6,
        "rtol": 1e-4,
    }

    # Update options with defaults only if not already set
    for key, value in default_options.items():
        if key not in options:
            options[key] = value
    # Initialize result storage for different regions
    all_states = []
    all_times = []
    current_state = psi_ini

    # Build Hamiltonian components
    H_free = system.H0_diagonalized  # already includes the RWA, if present!
    H_int_evo = QobjEvo(
        lambda t, args=None: H_free + H_int(t, pulse_seq, system)
    )  # also add H_int, with potential RWA

    # =============================
    # Choose solver and compute the evolution
    # =============================
    if system.ODE_Solver not in ["ME", "BR", "Paper_eqs", "Paper_BR"]:
        raise ValueError(f"Unknown ODE solver: {system.ODE_Solver}")

    if system.ODE_Solver == "Paper_eqs":
        c_ops_list = []
        if not system.RWA_laser:
            print(
                "The equations of the paper only make sense with RWA -> switched it on!",
                flush=True,
            )
            system.RWA_laser = True
        # For Paper_eqs, we need to define the full Liouville operator, which includes the decay channels
        EVO_obj = QobjEvo(lambda t, args=None: matrix_ODE_paper(t, pulse_seq, system))
        # no explicit c_ops, matrix_ODE represents the full Liouville operator

    # =============================
    # Split evolution by pulse regions for Paper_eqs
    # =============================
    # Find pulse regions in the time array using the dedicated function
    pulse_regions = identify_non_zero_pulse_regions(times, pulse_seq)
    # BASED ON THIS split the time range into regions where the pulse envelope is zero
    split_times = split_by_active_regions(times, pulse_regions)

    for i, times_ in enumerate(split_times):
        # Extend times_ by one point if not the last segment
        if i < len(split_times) - 1:
            # Find the first time point of the next segment
            next_times = split_times[i + 1]
            if len(next_times) > 0:
                # Extend current times_ with the first point of the next segment
                times_ = np.append(times_, next_times[0])

        # Find the indices in the original times array for this split
        start_idx = np.abs(times - times_[0]).argmin()

        # Check if this region has an active pulse by looking at the first time point
        has_pulse = pulse_regions[start_idx]

        # Set up collapse operators based on solver type; no c_ops for "BR"
        if system.ODE_Solver == "ME":
            if has_pulse:
                c_ops_list = []
                EVO_obj = H_free
            else:
                c_ops_list = system.me_decay_channels  # explicit c_ops
                EVO_obj = H_int_evo  # For ME, we use the Hamiltonian evolution operator directly
        elif system.ODE_Solver == "Paper_BR":
            # no explicit c_ops for Paper_BR, but we need to define the R operator
            c_ops_list = []
            EVO_obj = liouvillian(H_int_evo) + R_paper(
                system
            )  # TODO PROBLEM SOMEHOW INCLUDES RWA twice? -> double the oscillation

        if system.ODE_Solver == "BR":
            a_ops_list = system.br_decay_channels
            result = brmesolve(
                H_int_evo,
                current_state,
                times_,
                a_ops=a_ops_list,
                options=options,
            )
        else:
            result = mesolve(
                EVO_obj,
                current_state,
                times_,
                c_ops=c_ops_list,
                options=options,
            )

        if hasattr(result, "states") and result.states:
            # Store states - exclude last point for all but the final segment
            if i < len(split_times) - 1:
                # Not the last segment: exclude the last state/time to avoid duplication
                all_states.extend(result.states[:-1])
                all_times.extend(result.times[:-1])
            else:
                # Last segment: include all states and times
                all_states.extend(result.states)
                all_times.extend(result.times)

            # Update current state for next evolution
            current_state = result.states[-1]
        elif hasattr(result, "final_state"):
            current_state = result.final_state
        else:
            raise RuntimeError(
                "No valid state found in result object for next evolution step."
            )
    # =============================
    # Create combined result object using first result as base
    # =============================
    result_ = result

    if all_states:
        # store_states=True: assign all collected states/times
        result_.states = all_states
        result_.times = np.array(all_times)
    else:
        # store_states=False: assign only the final state/time
        if hasattr(result_, "final_state"):
            result_.states = [result_.final_state]
            # Optionally, assign the last time point if available
            if hasattr(result_, "times") and len(result_.times) > 0:
                result_.times = [result_.times[-1]]
            else:
                result_.times = []
        else:
            result_.states = []
            result_.times = []

    return result_


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
    if max_workers is None:
        max_workers = mp.cpu_count()

    phases = [k * np.pi / 2 for k in range(n_phases)]  # [0, π/2, π, 3π/2]
    if n_phases != 4:
        print(f"Warning: Phases {phases} may not be optimal for IFT", flush=True)

    print(
        f"Processing {n_freqs} frequencies with {n_phases}×{n_phases} phase combinations",
        flush=True,
    )
    print(f"Using {max_workers} parallel workers", flush=True)

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
                print(
                    f"Combination ({omega_idx},{phi1_idx},{phi2_idx}) failed: {exc}",
                    flush=True,
                )

    # Fill 3D result array
    results_cube = np.zeros((n_freqs, n_phases, n_phases), dtype=object)
    for (omega_idx, phi1_idx, phi2_idx), data in results.items():
        results_cube[omega_idx, phi1_idx, phi2_idx] = data

    # Average over frequencies before IFT
    results_matrix_avg = np.mean(results_cube, axis=0)

    # Final IFT
    photon_echo_signal = extract_ift_signal_component(
        results_matrix=results_matrix_avg, phases=phases, component=(-1, 1, 0)
    )

    return t_det_vals, photon_echo_signal * 1j  # because E ~ i P


def parallel_compute_2d_E_with_inhomogenity(
    n_freqs: int,
    n_phases: int,
    T_wait: float,
    t_det_max: np.ndarray,
    system,
    max_workers: int = None,
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
        signal_tensor[omega_idx, tau_idx, phi1_idx, phi2_idx] = data

    # === Build final 2D signal: one row per τ, after averaging and IFT ===
    data_avg_2d = []
    for tau_idx in reversed(range(n_tau)):  # latest τ on top # HERE I CHANGED: ordering
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

        # Apply IFT to this averaged phase matrix
        signal_2d = extract_ift_signal_component(
            results_matrix=results_matrix, phases=phases, component=(-1, 1, 0)
        )
        data_avg_2d.append(signal_2d * 1j)  # E ~ iP

    # Final stacking
    data_avg_2d = np.vstack(data_avg_2d)
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
