from dataclasses import dataclass  # for the class definiton
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
from qspectro2d.core.atomic_system.system_class import AtomicSystem
import numpy as np
import json

from qspectro2d.constants import HBAR


@dataclass
class SystemLaserCoupling:
    system: AtomicSystem
    laser: LaserPulseSequence

    # DERIVED QUANTITIES FROM SYSTEM / LASER PARAMETERS
    def rabi_0(self, i: int = 0):
        """Calculate the Rabi frequency for the i-th transition. [fs⁻¹]"""
        return self.system.dip_moments[i] * self.laser.E0 / HBAR

    def delta_rabi(self, i: int = 0):
        """Calculate the detuning for the i-th transition. [fs⁻¹]"""
        return self.laser.omega_laser - self.system._frequencies_fs[i]

    def rabi_gen(self, i: int = 0):
        """Calculate the generalized Rabi frequency for the i-th transition. [fs⁻¹]"""
        return np.sqrt(self.rabi_0(i) ** 2 + self.delta_rabi(i) ** 2)

    def t_prd(self, i: int = 0):
        """Calculate the period of the Rabi oscillation. (for TLS: one full cycle between |g> and |e>)"""
        return 2 * np.pi / self.rabi_gen(i) if self.rabi_gen(i) != 0 else 0.0

    def summary(self) -> str:
        rabi_0_values = [self.rabi_0(i) for i in range(self.system.n_atoms)]
        delta_rabi_values = [self.delta_rabi(i) for i in range(self.system.n_atoms)]
        rabi_gen_values = [self.rabi_gen(i) for i in range(self.system.n_atoms)]
        t_prd_values = [self.t_prd(i) for i in range(self.system.n_atoms)]
        lines = [
            "=== SystemLaserCoupling Summary ===",
            f"Rabi Frequencies (0th order): {rabi_0_values}",
            f"Detunings (Delta Rabi): {delta_rabi_values}",
            f"Rabi Frequencies (Generalized): {rabi_gen_values}",
            f"Periods (T_prd): {t_prd_values}",
        ]
        return "\n".join(lines)

    # SERIALIZATION METHODS
    def to_dict(self):
        return {"system": self.system.to_dict(), "laser": self.laser.to_dict()}

    @classmethod
    def from_dict(cls, d):
        return cls(
            system=AtomicSystem.from_dict(d["system"]),
            laser=LaserPulseSequence.from_dict(d["laser"]),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize from JSON string."""
        d = json.loads(json_str)
        return cls.from_dict(d)

    def __str__(self) -> str:
        return self.summary()


if __name__ == "__main__":
    print("Testing SystemLaserCoupling class...")

    # Create mock AtomicSystem and LaserPulseSequence objects
    mock_atomic_system = AtomicSystem(
        n_atoms=2, at_frequencies_cm_cm=[16000.0, 16100.0], dip_moments=[1.0, 2.0]
    )
    seq = LaserPulseSequence.from_delays(
        delays=[100.0, 300.0],
        base_amplitude=0.05,
        pulse_fwhm=10.0,
        carrier_freq_cm=15800.0,
        relative_E0s=[1.0, 1.0, 0.1],
        phases=[0.0, 0.5, 1.0],
    )

    # Instantiate SystemLaserCoupling
    coupling = SystemLaserCoupling(system=mock_atomic_system, laser=seq)

    # Print summary
    coupling.summary()

    # Test serialization
    json_str = coupling.to_json()
    print("Serialized JSON string:")
    print(json_str)

    # Test deserialization
    coupling_from_json = SystemLaserCoupling.from_json(json_str)
    print("Deserialized object:")
    coupling_from_json.summary()
