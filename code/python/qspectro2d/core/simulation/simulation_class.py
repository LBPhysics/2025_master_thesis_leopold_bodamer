"""Core simulation builders (model assembly & interaction Hamiltonians)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import numpy as np
from qutip import (
    Qobj,
    QobjEvo,
    ket2dm,
)
from typing import List, Tuple
from qutip import BosonicEnvironment

from .sim_config import SimulationConfig
from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
from qspectro2d.core.laser_system.laser_fcts import e_pulses, epsilon_pulses
from qspectro2d.core.system_bath_class import SystemBathCoupling
from qspectro2d.core.system_laser_class import SystemLaserCoupling
from qspectro2d.constants import HBAR


def H_int_(t: float, lowering_op: Qobj, rwa_sl: bool, laser: LaserPulseSequence) -> Qobj:
    """Interaction Hamiltonian (-μ·E) with optional RWA.

    Parameters
    ----------
    t : float
        Time.
    lowering_op : Qobj
        System lowering operator in eigenbasis!.
    rwa_sl : bool
        Apply rotating wave approximation.
    laser : LaserPulseSequence
        Pulse sequence.
    """
    if rwa_sl:
        E_field_RWA = e_pulses(t, laser)
        return -(lowering_op.dag() * E_field_RWA + lowering_op * np.conj(E_field_RWA))
    dipole_op = lowering_op + lowering_op.dag()
    E_field = epsilon_pulses(t, laser)
    return -dipole_op * (E_field + np.conj(E_field))


def paper_eqs_evo(sim: "SimulationModuleOQS", t: float) -> Qobj:  # pragma: no cover simple wrapper
    """Global helper for 'Paper_eqs' solver evolution.

    Kept at module scope so partial(paper_eqs_evo, sim) remains pickleable.
    Lazy import inside to avoid circular import at module load.
    """
    from qspectro2d.core.simulation.liouvillian_paper import (
        matrix_ODE_paper as _matrix_ODE_paper,
    )

    return _matrix_ODE_paper(t, sim)


@dataclass
class SimulationModuleOQS:
    simulation_config: SimulationConfig

    system: AtomicSystem
    laser: LaserPulseSequence
    bath: BosonicEnvironment

    sb_coupling: SystemBathCoupling = field(init=False)

    def __post_init__(self) -> None:
        # TODO THEY COULD POTENTIALLY CHANGE AFTER CONSTRUCTION
        self.sb_coupling = SystemBathCoupling(self.system, self.bath)
        # Defer solver-dependent initialization (evo object & decay channels)
        # until first access so that changing simulation_config.ode_solver
        # after construction (for experimentation) is possible via reset.
        self._evo_obj = None  # type: ignore[attr-defined]
        self._decay_channels = None  # type: ignore[attr-defined]

    # --- Deferred solver-dependent initialization ---------------------------------
    @property
    def evo_obj(self):  # type: ignore[override]
        solver = self.simulation_config.ode_solver
        if solver == "Paper_eqs":
            # Keep pickleable: use module-level function partially bound to self.
            self._evo_obj = partial(paper_eqs_evo, self)
        elif solver == "ME" or solver == "BR":
            self._evo_obj = QobjEvo(self.H_total_t)
        else:  # Fallback: create generic evolution
            self._evo_obj = QobjEvo(self.H0_diagonalized)
        return self._evo_obj

    @property
    def decay_channels(self):  # type: ignore[override]
        solver = self.simulation_config.ode_solver
        if solver == "Paper_eqs":
            self._decay_channels = []
        elif solver == "ME":
            self._decay_channels = self.sb_coupling.me_decay_channels
        elif solver == "BR":
            self._decay_channels = self.sb_coupling.br_decay_channels
        else:  # Fallback: create generic evolution with no decay channels.
            self._decay_channels = []
        return self._decay_channels

    def reset_solver_objects(self) -> None:
        """Reset cached solver objects so they are rebuilt on next access."""
        self._evo_obj = None
        self._decay_channels = None

    # --- Hamiltonians & Evolutions -------------------------------------------------
    @property
    def H0_diagonalized(self) -> Qobj:
        """Return diagonal Hamiltonian (optionally shifted by laser frequency under RWA)."""
        Es, _ = self.system.eigenstates
        H_diag = Qobj(np.diag(Es), dims=self.system.hamiltonian.dims)
        if self.simulation_config.rwa_sl:
            omega_L = self.laser._carrier_freq_fs
            # Determine excitation number for each eigenstate
            # Based on index: 0 -> 0 excitations, 1..N -> 1, N+1..end -> 2
            H_diag -= HBAR * omega_L * self.system.number_op  # is the same in both bases
        return H_diag

    def H_int_sl(self, t: float) -> Qobj:
        lowering_op = self.system.lowering_op
        H_int = H_int_(
            t, self.system.to_eigenbasis(lowering_op), self.simulation_config.rwa_sl, self.laser
        )
        return H_int

    def H_total_t(self, t: float) -> Qobj:
        """Return total Hamiltonian H0 + H_int(t) at time t."""
        H_total = self.H0_diagonalized + self.H_int_sl(t)
        print(f"H at t={t}: {H_total.norm()}")
        return H_total

    # TODO also add time dependent eigenenergies / states? and also all the other operators?
    def time_dep_eigenstates(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues & eigenstates (cached).

        Invalidated when `update_frequencies_cm` is called.
        """
        return self.H_total_t(t).eigenstates()

    def time_dep_omega_ij(self, i: int, j: int, t: float) -> float:
        """Return energy difference (frequency) between instantaneous eigenstates i and j in fs^-1."""
        return self.time_dep_eigenstates(t)[0][i] - self.time_dep_eigenstates(t)[0][j]

    # --- Observables ---------------------------------------------------------------
    @property
    def observable_ops(self) -> List[Qobj]:
        sys = self.system
        n = sys.n_atoms

        eigenstates = sys.eigenstates[1]
        ops = [ket2dm(state) for state in eigenstates]  # populations

        # Add coherences: |g><e|, |g><f|, |e><f|
        dim = sys.dimension
        if dim > 1:
            # |g><e| for all  for e (1, ..., n_atoms)
            ops.append(sum(eigenstates[0] * eigenstates[e].dag() for e in range(1, dim)))
        if dim > n + 1:
            # |g><f| for f (n_atoms+1, ..., dim)
            ops.append(sum(eigenstates[0] * eigenstates[f].dag() for f in range(n + 1, dim)))
            # |e><f| for e (1, ..., n_atoms) and f (n_atoms+1, ..., dim)
            ops.append(
                sum(
                    eigenstates[e] * eigenstates[f].dag()
                    for e in range(1, n + 1)
                    for f in range(n + 1, dim)
                )
            )

        return ops

    @property
    def observable_strs(self) -> List[str]:
        sys = self.system
        n = sys.n_atoms
        dim = sys.dimension
        strs = []
        # Populations
        strs.extend([f"pop_{i}" for i in range(dim)])
        # Coherences
        if dim > 1:
            strs.append(r"\text{coh}_{\text{ge}}")
        if dim > n + 1:
            strs.append(r"\text{coh}_{\text{gf}}")
            strs.append(r"\text{coh}_{\text{ef}}")
        return strs

    # --- Time grids ----------------------------------------------------------------
    @property
    def times_local(self):
        if hasattr(self, "_times_local_manual"):
            return self._times_local_manual

        t0 = -2 * self.laser.pulse_fwhms[0]
        cfg = self.simulation_config
        t_max_curr = cfg.t_coh + cfg.t_wait + cfg.t_det_max
        dt = cfg.dt
        # Compute number of steps to cover from t0 to t_max_curr with step dt
        n_steps = int(np.floor((t_max_curr - t0) / dt)) + 10  # small buffer
        # Generate time grid: [t0, t0 + dt, ..., t_max_curr]
        times = t0 + dt * np.arange(n_steps, dtype=float)
        return times

    @times_local.setter
    def times_local(self, times: np.ndarray):
        self._times_local_manual = np.asarray(times, dtype=float).reshape(-1)

    def reset_times_local(self):
        if hasattr(self, "_times_local_manual"):
            delattr(self, "_times_local_manual")

    @property
    def t_det(self):
        # Detection time grid with exact spacing dt starting at the first time >0 in times_local.

        dt = self.simulation_config.dt
        t_det_max = self.simulation_config.t_det_max
        # Compute the first time > 0
        times_local = self.times_local
        t_start = times_local[times_local >= 0][0]
        n_steps = int(np.floor(t_det_max / dt)) + 1
        if t_start + dt * (n_steps - 1) > t_det_max:
            # Cap it to avoid overshooting t_det_max and times_local
            n_steps = int(np.floor((t_det_max - t_start) / dt)) + 1
            n_steps = min(n_steps, times_local[times_local >= 0].size)
        return t_start + dt * np.arange(n_steps, dtype=float)

    @property
    def t_det_actual(self):
        cfg = self.simulation_config
        t_det0 = cfg.t_coh + cfg.t_wait
        return self.t_det + t_det0

    # Evolution objects (Paper specific ones live in liouvillian_paper / redfield)
    def summary(self) -> str:
        return (
            "SimulationModuleOQS Summary\n"
            f"Solver: {self.simulation_config.ode_solver}\n"
            f"Decay channels: {len(self.decay_channels)}\n"
        )

    def __str__(self) -> str:  # pragma: no cover simple repr
        return self.summary()
