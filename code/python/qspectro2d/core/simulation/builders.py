"""Core simulation builders (model assembly & interaction Hamiltonians)."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from qutip import (
    Qobj,
    QobjEvo,
    ket2dm,
    liouvillian,
)
from typing import List

from .config import SimulationConfig
from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
from qspectro2d.core.laser_system.laser_fcts import E_pulse, Epsilon_pulse
from qspectro2d.core.system_bath_class import SystemBathCoupling
from qspectro2d.core.system_laser_class import SystemLaserCoupling
from qutip import BosonicEnvironment
from qspectro2d.config import HBAR


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
        if solver == "ME":
            self.decay_channels = self.sb_coupling.me_decay_channels
        elif solver == "BR":
            self.decay_channels = self.sb_coupling.br_decay_channels
        else:  # Paper variants manage decay separately
            self.decay_channels = []

    # --- Hamiltonians & Evolutions -------------------------------------------------
    @property
    def H0_diagonalized(self) -> Qobj:
        """Return diagonal Hamiltonian (optionally shifted by laser frequency under RWA).

        IMPORTANT: We copy eigenvalues to avoid mutating cached arrays.
        """
        Es, _ = self.system.eigenstates
        Es = Es.copy()  # avoid in-place modification of upstream arrays
        n_atoms = self.system.n_atoms
        if self.simulation_config.rwa_sl:
            if n_atoms == 1:
                Es[1] -= HBAR * self.laser.omega_laser
            elif n_atoms == 2:
                Es[1] -= HBAR * self.laser.omega_laser
                Es[2] -= HBAR * self.laser.omega_laser
                Es[3] -= 2 * HBAR * self.laser.omega_laser
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
