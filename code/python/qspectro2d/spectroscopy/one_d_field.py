"""Compute 1D emitted electric field E_ks(t_det) via phase-cycled third-order polarization.

This module provides a clean, focused API that mirrors the physics and flow you described:

Steps:
- E_ks(t) ∝ i P_ks(t)
- P_{l,m}(t) = Σ_{phi1} Σ_{phi2} P_{phi1,phi2}(t) * exp(-i(l phi1 + m phi2 + n PHI_DET))
- P_{phi1,phi2}(t) = P_total(t) - Σ_i P_i(t), with P_total using all pulses and P_i with only pulse i active
- P(t) is the complex/analytical polarization: P(t) = ⟨μ_+⟩(t), using the positive-frequency part of μ

Reuses existing building blocks:
- Evolution: compute_seq_evolution(sim_oqs, ...)
- Polarization extraction: complex_polarization(dipole_op, states)

Supports ME and BR solvers via the internals of SimulationModuleOQS.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Optional, Dict, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from copy import deepcopy

from qutip import Qobj, Result, QobjEvo, mesolve, brmesolve

from qspectro2d.core.simulation.simulation_class import SimulationModuleOQS
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
from qspectro2d.spectroscopy.polarization import complex_polarization
from qspectro2d.spectroscopy.inhomogenity import sample_from_gaussian
from project_config.logging_setup import get_logger
from qspectro2d.config.default_simulation_params import (
    PHASE_CYCLING_PHASES,
    COMPONENT_MAP,
    DETECTION_PHASE,
)


logger = get_logger(__name__)


__all__ = [
    "parallel_compute_1d_e_comps",
]


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------
def _result_detection_slice(res: Result, t_det: np.ndarray) -> List[Qobj]:
    """Slice the list of states in `res` to only keep the detection window portion."""
    if not hasattr(res, "states") or res.states is None:
        raise RuntimeError("Evolution result did not store states; pass store_states=True.")
    n_det = len(t_det)
    n_all = len(res.times)
    start = n_all - n_det
    if start < 0:
        raise ValueError("Detection window longer than total evolution.")
    return res.states[start:]


def _run_evolution(
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


def compute_evolution(
    sim_oqs: SimulationModuleOQS,
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

    current_state = sim_oqs.system.psi_ini
    actual_times = sim_oqs.times_local
    decay_ops_list = sim_oqs.decay_channels
    evo_obj = sim_oqs.evo_obj_int
    return _run_evolution(
        sim_oqs.simulation_config.ode_solver,
        evo_obj,
        decay_ops_list,
        current_state,
        actual_times,
        options,
    )


def _compute_polarization_over_window(
    sim: SimulationModuleOQS, *, store_states: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Evolve the system once (with current laser settings) and return (t_det, P(t_det)).

    - Uses `compute_seq_evolution` which dispatches ME/BR according to sim.simulation_config.
    - Extracts complex/analytical polarization over the detection window only.
    """
    t_det = sim.times_det_actual
    # Ensure we store states to extract polarization
    res: Result = compute_evolution(sim, store_states=store_states)
    det_states = _result_detection_slice(res, t_det)

    # Analytical polarization using positive-frequency part of dipole operator
    mu_op = sim.system.to_eigenbasis(sim.system.dipole_op)
    P_t = complex_polarization(mu_op, det_states)  # np.ndarray[complex]
    return t_det, P_t


def _with_only_pulse_i_active(sim: SimulationModuleOQS, i: int) -> SimulationModuleOQS:
    """Return a deep-copied sim where only pulse i is active (others have amplitude 0).

    Notes:
    - Deep-copy is used to avoid mutating the input and to be process/thread-safe.
    - Phases and timings remain unchanged.
    """
    sim_i = deepcopy(sim)
    # Build a one-pulse sequence matching the i-th pulse timing and phase
    pulse_i = sim_i.laser.pulses[i]
    sim_i.laser = LaserPulseSequence(pulses=[pulse_i])
    return sim_i


def _set_pulse_phases_inplace(sim: SimulationModuleOQS, phi1: float, phi2: float) -> None:
    """Set the phases of the first two pulses in-place, preserving others."""
    if len(sim.laser.pulses) < 2:
        raise ValueError("At least two pulses are required for phase cycling (phi1, phi2).")
    sim.laser.pulses[0].pulse_phase = float(phi1)
    sim.laser.pulses[1].pulse_phase = float(phi2)


def _compute_P_phi1_phi2(
    sim: SimulationModuleOQS, phi1: float, phi2: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute P_{phi1,phi2}(t_det) = P_total - Σ_i P_i for current phases.

    Returns (t_det, P_phi1_phi2(t_det)).
    """
    # Work on copies to avoid permanently mutating `sim`
    sim_work = deepcopy(sim)
    _set_pulse_phases_inplace(sim_work, phi1, phi2)

    # Total signal with all pulses
    t_det, P_total = _compute_polarization_over_window(sim_work)

    # Linear signals: only pulse i active
    P_linear_sum = np.zeros_like(P_total, dtype=np.complex128)
    for i in range(len(sim_work.laser.pulses)):
        sim_i = _with_only_pulse_i_active(sim_work, i)
        _, P_i = _compute_polarization_over_window(sim_i)
        P_linear_sum += P_i

    P_phi = P_total - P_linear_sum
    return t_det, P_phi


def _worker_P_phi_pair(
    sim_template: SimulationModuleOQS, phi1: float, phi2: float
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Thread worker: compute (phi1, phi2, t_det, P_{phi1,phi2}).

    A deep copy of the simulation is created per worker to avoid shared state.
    """
    sim_local = deepcopy(sim_template)
    t_det, P_phi = _compute_P_phi1_phi2(sim_local, phi1, phi2)
    return phi1, phi2, t_det, P_phi


def _phase_cycle_component(
    phases1: Sequence[float],
    phases2: Sequence[float],
    P_grid: np.ndarray,
    *,
    lmn: Tuple[int, int, int] = (0, 0, 0),
    phi_det: float = 0.0,  # default is overridden at call site using DETECTION_PHASE
    normalize: bool = True,
) -> np.ndarray:
    """Extract P_{l,m,n}(t) from a grid P[phi1,phi2,t].

    P_{l,m,n}(t) = Σ_{phi1} Σ_{phi2} P_{phi1,phi2}(t) exp(-i(l phi1 + m phi2 + n phi_det))
    """
    l, m, n = lmn
    P_out = np.zeros(P_grid.shape[-1], dtype=np.complex128)
    for i, phi1 in enumerate(phases1):
        for j, phi2 in enumerate(phases2):
            phase = -1j * (l * phi1 + m * phi2 + n * phi_det)
            P_out += P_grid[i, j, :] * np.exp(phase)
    if normalize:
        P_out /= len(phases1) * len(phases2)
    return P_out


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def parallel_compute_1d_e_comps(
    sim_oqs: SimulationModuleOQS,
    *,
    phases: Optional[Sequence[float]] = None,
    lmn: Optional[Tuple[int, int, int]] = None,
    phi_det: Optional[float] = None,
    time_cut: Optional[float] = None,
) -> List[np.ndarray]:
    """Compute 1D electric field components E_kS(t_det) with phase cycling only.

    This simplified function assumes the provided `sim_oqs` already encodes a single
    inhomogeneous realization (i.e., system frequencies are already set). No internal
    sampling or averaging over inhomogeneity is performed. Use external batching if needed.

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Prepared simulation (system, laser sequence, solver config).
    phases : Optional[Sequence[float]]
        Phase grid for (phi1, phi2). If None, use PHASE_CYCLING_PHASES truncated to n_phases.
    lmn : Optional[Tuple[int,int,int]]
        Component to extract; if None, derive from signal types via COMPONENT_MAP.
    phi_det : Optional[float]
        Detection phase; if None, use DETECTION_PHASE.
    time_cut : Optional[float]
        Truncate detection times after this value [fs] (soft mask applied).

    Returns
    -------
    List[np.ndarray]
        List of complex E-components, one per entry in `sim_oqs.simulation_config.signal_types`.
        Each array has length len(sim_oqs.times_det). A soft time_cut is applied by zeroing beyond cutoff.
    """
    # Determine phases from config defaults if not provided
    n_ph = sim_oqs.simulation_config.n_phases
    phases_src = phases if phases is not None else PHASE_CYCLING_PHASES
    phases_eff = tuple(float(x) for x in phases_src[:n_ph])

    # Prepare grid and helpers
    n_t = len(sim_oqs.times_det)
    sig_types = sim_oqs.simulation_config.signal_types
    phi_det_val = phi_det if phi_det is not None else float(DETECTION_PHASE)

    # Optional time mask (keep length constant)
    t_mask = None
    if time_cut is not None and np.isfinite(time_cut):
        t_mask = (sim_oqs.times_det <= float(time_cut)).astype(np.float64)

    # Compute P_{phi1,phi2} grid once for this realization
    P_grid = np.zeros((len(phases_eff), len(phases_eff), n_t), dtype=np.complex128)
    futures = []
    with ProcessPoolExecutor() as ex:
        for phi1 in phases_eff:
            for phi2 in phases_eff:
                futures.append(ex.submit(_worker_P_phi_pair, deepcopy(sim_oqs), phi1, phi2))
        temp_results: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
        for fut in as_completed(futures):
            phi1_v, phi2_v, t_det, P_phi = fut.result()
            temp_results[(phi1_v, phi2_v)] = (t_det, P_phi)

    for i, phi1 in enumerate(phases_eff):
        for j, phi2 in enumerate(phases_eff):
            _, P_phi = temp_results[(phi1, phi2)]
            P_grid[i, j, :] = P_phi

    # Extract components for this realization
    E_list: List[np.ndarray] = []
    for sig in sig_types:
        lmn_tuple = COMPONENT_MAP[sig] if lmn is None else lmn
        P_comp = _phase_cycle_component(
            phases_eff,
            phases_eff,
            P_grid,
            lmn=lmn_tuple,
            phi_det=phi_det_val,
            normalize=True,
        )
        E_comp = 1j * P_comp
        if t_mask is not None:
            E_comp = E_comp * t_mask
        E_list.append(E_comp)

    return E_list
