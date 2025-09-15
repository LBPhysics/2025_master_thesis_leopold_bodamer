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
from typing import List
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

    sl_coupling: SystemLaserCoupling = field(init=False)
    sb_coupling: SystemBathCoupling = field(init=False)

    def __post_init__(self) -> None:
        self.sb_coupling = SystemBathCoupling(self.system, self.bath)
        self.sl_coupling = SystemLaserCoupling(self.system, self.laser)

        solver = self.simulation_config.ode_solver
        H0_diagonalized = self.H0_diagonalized

        # if solver != "Paper_eqs" or "BR" or "ME": -> case already covered with warning
        if solver == "Paper_eqs":
            # Use a module-level wrapper + functools.partial to keep object pickleable under Windows spawn.
            self.evo_obj_free = partial(paper_eqs_evo, self)
            self.evo_obj_int = self.evo_obj_free
            self.decay_channels = []

        elif solver == "ME":
            self.decay_channels = self.sb_coupling.me_decay_channels
            self.evo_obj_free = H0_diagonalized
            self.evo_obj_int = QobjEvo(self.H_total)

        elif solver == "BR":
            self.decay_channels = self.sb_coupling.br_decay_channels
            self.evo_obj_free = H0_diagonalized
            # BR needs the full system Hamiltonian at all times; include H0 + time-dependent interaction
            self.evo_obj_int = QobjEvo(self.H_total)

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

    @property
    def H_total(self, t: float) -> Qobj:
        """Return total (time-independent) Hamiltonian H0 + H_int at t=0."""
        return self.H0_diagonalized + self.H_int_sl(t)

    # TODO also add time dependent eigenenergies / states? and also all the other operators?

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
    def times_global(self):  # lazy compute
        if not hasattr(self, "_times_global"):
            t0 = -self.laser.pulse_fwhms[0]
            t_max = self.simulation_config.t_max
            dt = self.simulation_config.dt
            n_steps = int(np.round((t_max - t0) / dt)) + 1
            self._times_global = np.linspace(t0, t_max, n_steps)
        return self._times_global

    @times_global.setter
    def times_global(self, value):
        self._times_global = value

    @property
    def times_local(self):
        if hasattr(self, "_times_local_manual"):
            return self._times_local_manual
        tg = self.times_global
        cfg = self.simulation_config
        t_max_curr = cfg.t_coh + cfg.t_wait + cfg.t_det_max
        idx = np.abs(tg - t_max_curr).argmin()
        return tg[: idx + 1]

    @times_local.setter
    def times_local(self, value):
        self._times_local_manual = value

    def reset_times_local(self):
        if hasattr(self, "_times_local_manual"):
            delattr(self, "_times_local_manual")

    @property
    def times_det(self):
        dt = self.simulation_config.dt
        t_det_max = self.simulation_config.t_det_max
        n_steps = int(np.round(t_det_max / dt)) + 1
        return np.linspace(0, t_det_max, n_steps)

    @property
    def times_det_actual(self):
        self.reset_times_local()
        td = self.times_det
        return self.times_local[-len(td) :]

    # Evolution objects (Paper specific ones live in liouvillian_paper / redfield)
    def summary(self) -> str:
        return (
            "SimulationModuleOQS Summary\n"
            f"Solver: {self.simulation_config.ode_solver}\n"
            f"Decay channels: {len(self.decay_channels)}\n"
        )

    def __str__(self) -> str:  # pragma: no cover simple repr
        return self.summary()
