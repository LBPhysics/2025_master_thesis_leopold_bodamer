"""Core simulation builders (model assembly & interaction Hamiltonians)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import numpy as np
from qutip import (
    Qobj,
    QobjEvo,
    ket2dm,
    liouvillian,
)
from typing import List
from qutip import BosonicEnvironment

from .config import SimulationConfig
from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
from qspectro2d.core.laser_system.laser_fcts import E_pulse, Epsilon_pulse
from qspectro2d.core.system_bath_class import SystemBathCoupling
from qspectro2d.core.system_laser_class import SystemLaserCoupling
from qspectro2d.constants import HBAR


def H_int_(t: float, sm_op: Qobj, rwa_sl: bool, laser: LaserPulseSequence) -> Qobj:
    """Interaction Hamiltonian (-μ·E) with optional RWA.

    Parameters
    ----------
    t : float
        Time.
    sm_op : Qobj
        System lowering operator.
    rwa_sl : bool
        Apply rotating wave approximation.
    laser : LaserPulseSequence
        Pulse sequence.
    """
    if rwa_sl:
        E_field_RWA = E_pulse(t, laser)
        return -(sm_op.dag() * E_field_RWA + sm_op * np.conj(E_field_RWA))
    dip_op = sm_op + sm_op.dag()
    E_field = Epsilon_pulse(t, laser)
    return -dip_op * (E_field + np.conj(E_field))


def paper_eqs_evo(
    sim: "SimulationModuleOQS", t: float
) -> Qobj:  # pragma: no cover simple wrapper
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

        if solver == "Paper_eqs":
            # Use a module-level wrapper + functools.partial to keep object pickleable under Windows spawn.
            self.evo_obj_free = partial(paper_eqs_evo, self)
            self.evo_obj_int = self.evo_obj_free
            self.decay_channels = []

        elif solver == "Paper_BR":
            # TODO somehow contains 2 RWAs for n_atoms == 2.
            from qspectro2d.core.simulation.redfield import R_paper as _R_paper

            # This version is computable with mesolve H -> evo_obj; no need for a_ops
            custom_free = liouvillian(self.H0_diagonalized) + _R_paper(self)
            self.evo_obj_free = custom_free
            self.evo_obj_int = custom_free + liouvillian(QobjEvo(self.H_int_sl))

            """
            # TODO This version can be passed to brmesolve?
            R_super = _R_paper(self)  # time-independent Redfield tensor (Qobj)
            self.evo_obj_free = H0_diagonalized
            self.evo_obj_int = H0_diagonalized + QobjEvo(self.H_int_sl)
            self.decay_channels = R_super  # Redfield handled separately
            """

        elif solver == "ME":
            self.decay_channels = self.sb_coupling.me_decay_channels
            self.evo_obj_free = H0_diagonalized
            self.evo_obj_int = H0_diagonalized + QobjEvo(self.H_int_sl)

        elif solver == "BR":
            self.decay_channels = self.sb_coupling.br_decay_channels
            self.evo_obj_free = H0_diagonalized
            self.evo_obj_int = QobjEvo(self.H_int_sl)
        else:
            # Fallback: treat as ME-style with no decay channels
            # TODO return warning
            self.decay_channels = []
            self.evo_obj_free = H0_diagonalized
            self.evo_obj_int = QobjEvo(self.H_int_sl)

    # --- Hamiltonians & Evolutions -------------------------------------------------
    @property
    def H0_diagonalized(self) -> Qobj:
        """Return diagonal Hamiltonian (optionally shifted by laser frequency under RWA).

        IMPORTANT: We copy eigenvalues to avoid mutating cached arrays.
        """
        Es, _ = self.system.eigenstates
        Es = Es.copy()  # avoid in-place modification of upstream arrays
        N = self.system.n_atoms

        if self.simulation_config.rwa_sl:
            omega_L = self.laser.omega_laser

            # Determine excitation number for each eigenstate
            # Based on index: 0 -> 0 excitations, 1..N -> 1, N+1..end -> 2
            for i in range(len(Es)):
                n_exc = self.system.excitation_number_from_index(i)
                Es[i] -= n_exc * HBAR * omega_L

        return Qobj(np.diag(Es), dims=self.system.H0_N_canonical.dims)

    def H_int_sl(self, t: float) -> Qobj:
        return H_int_(t, self.system.sm_op, self.simulation_config.rwa_sl, self.laser)

    # --- Observables ---------------------------------------------------------------
    @property
    def observable_ops(self) -> List[Qobj]:
        if self.system.n_atoms == 1:
            return [
                ket2dm(self.system._atom_g),
                self.system._atom_g * self.system._atom_e.dag(),
                self.system._atom_e * self.system._atom_g.dag(),
                ket2dm(self.system._atom_e),
            ]
        if self.simulation_config.keep_track == "basis":
            return [ket2dm(b) for b in self.system.basis]
        return [ket2dm(state) for state in self.system.eigenstates[1]]

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
