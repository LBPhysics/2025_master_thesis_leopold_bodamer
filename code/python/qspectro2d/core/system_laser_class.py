from dataclasses import dataclass  # for the class definiton
from qspectro2d.core.laser_system.laser_class import LaserPulseSystem
from core.atomic_system.system_class import AtomicSystem
import numpy as np

from qspectro2d.core.utils_and_config import HBAR


@dataclass
class SystemLaserCoupling:
    system: AtomicSystem
    laser: LaserPulseSystem

    # DERIVED QUANTITIES FROM SYSTEM / LASER PARAMETERS
    @property
    def rabi_0(self):
        return self.system.mu_A * self.laser.E0 / HBAR

    @property
    def delta_rabi(self):
        return self.laser.omega - self.system.omega_A

    @property
    def rabi_gen(self):
        return np.sqrt(self.rabi_0**2 + self.delta_rabi**2)

    @property
    def t_prd(self):
        """Calculate the period of the Rabi oscillation. (for TLS: one full cycle between |g> and |e>)"""
        return 2 * np.pi / self.rabi_gen if self.rabi_gen != 0 else 0.0

    def summary(self):
        print("=== SystemLaserCoupling Summary ===")
        print(f"Rabi Frequency (0th order): {self.rabi_0:.4f} fs⁻¹")
        print(f"Detuning (Delta Rabi): {self.delta_rabi:.4f} fs⁻¹")
        print(f"Rabi Frequency (Generalized): {self.rabi_gen:.4f} fs⁻¹")
        print(f"Period (T_prd): {self.t_prd:.4f} fs")
        # what about the rest of the stuff
