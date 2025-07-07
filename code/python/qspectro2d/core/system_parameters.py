# =============================
# DEFINE THE SYSTEM PARAMETERS CLASS
# =============================
from dataclasses import dataclass, field  # for the class definiton
from typing import Optional, List
import numpy as np
from qutip import basis, ket2dm, tensor, Qobj
from qspectro2d.core.utils_and_config import convert_cm_to_fs, HBAR

@dataclass
class SystemParameters:
    atom_g: Qobj = field(default_factory=lambda: basis(2, 0))
    atom_e: Qobj = field(default_factory=lambda: basis(2, 1))
    psi_ini: Optional[Qobj] = None

    N_atoms: int = 1
    freqs_cm: List[float] = field(default_factory=lambda: [16000.0])  # in cm^-1
    dip_moments: List[float] = field(default_factory=lambda: [1.0])
    Delta_cm: Optional[float] = 0.0
    J_cm: Optional[float] = None  # For N_atoms = 2

    def __post_init__(self):
        if len(self.freqs_cm) != self.N_atoms:
            raise ValueError("Length of freqs_cm must match N_atoms")
        if len(self.dip_moments) != self.N_atoms:
            raise ValueError("Length of dip_moments must match N_atoms")

        if self.psi_ini is None:
            self.psi_ini = ket2dm(self.atom_g if self.N_atoms == 1 else tensor(*[self.atom_g] * self.N_atoms))

        if self.N_atoms == 2 and self.J_cm is None:
            self.J_cm = 0.0

        if self.Delta_cm is None:
            self.Delta_cm = 200.0

        self.H0_undiagonalized = self.Hamilton_N_atoms()

    def freqs_fs(self, i):
        return convert_cm_to_fs(self.freqs_cm[i])

    @property
    def J(self):
        return convert_cm_to_fs(self.J_cm)

    @property
    def theta(self):
        return np.arctan(2 * self.J / (self.freqs_fs(0) * self.freqs_fs(1))) / 2

    @property
    def Delta(self):
        return convert_cm_to_fs(self.Delta_cm)

    def Hamilton_tls(self):
        return HBAR * self.freqs_fs(0) * ket2dm(self.atom_e)

    def Hamilton_dimer_sys(self):
        H = HBAR * (
            self.freqs_fs(0) * ket2dm(tensor(self.atom_e, self.atom_g))
            + self.freqs_fs(1) * ket2dm(tensor(self.atom_g, self.atom_e))
            + self.J * (
                tensor(self.atom_e, self.atom_g) * tensor(self.atom_g, self.atom_e).dag()
                + tensor(self.atom_g, self.atom_e) * tensor(self.atom_e, self.atom_g).dag()
            )
            + (self.freqs_fs(0) + self.freqs_fs(1)) * ket2dm(tensor(self.atom_e, self.atom_e))
        )
        return H

    def Hamilton_N_atoms(self):
        if self.N_atoms == 1:
            return self.Hamilton_tls()
        elif self.N_atoms == 2:
            return self.Hamilton_dimer_sys()
        else: # TODO IMPLEMENT THE N_atoms > 2 CASE IN SINGLE EXCITATION SUBSPACE
            raise ValueError("Only N_atoms=1 or 2 are supported.")

    @property
    def eigenstates(self):
        return self.H0_undiagonalized.eigenstates()

    @property
    def SM_op(self):
        if self.N_atoms == 1:
            return self.dip_moments[0] * (self.atom_g * self.atom_e.dag())
        elif self.N_atoms == 2:
            C_A_1 = -np.sin(self.theta)
            C_A_2 = np.cos(self.theta)
            C_B_1 = C_A_2
            C_B_2 = -C_A_1
            mu_A = self.dip_moments[0]
            mu_B = self.dip_moments[1]
            mu_10 = mu_A * C_A_1 + mu_B * C_A_2
            mu_20 = mu_A * C_B_1 + mu_B * C_B_2
            mu_31 = mu_B * C_A_1 + mu_A * C_B_1
            mu_32 = mu_B * C_B_1 + mu_A * C_B_2
            _, eigenvecs = self.eigenstates
            return sum([
                mu_10 * (eigenvecs[0] * eigenvecs[1].dag()),
                mu_20 * (eigenvecs[0] * eigenvecs[2].dag()),
                mu_31 * (eigenvecs[1] * eigenvecs[3].dag()),
                mu_32 * (eigenvecs[2] * eigenvecs[3].dag()),
            ])
        else: # TODO IMPLEMENT THE N_atoms > 2 CASE IN SINGLE EXCITATION SUBSPACE
            raise ValueError("Only N_atoms=1 or 2 are supported.")

    @property
    def Dip_op(self):
        return self.SM_op + self.SM_op.dag()

    @property
    def Deph_op(self): # TODO IMPLEMENT THE N_atoms > 2 CASE IN SINGLE EXCITATION SUBSPACE
        if self.N_atoms == 1:
            return ket2dm(self.atom_e)
        elif self.N_atoms == 2:
            return sum([
                ket2dm(tensor(self.atom_e, self.atom_g)),
                ket2dm(tensor(self.atom_g, self.atom_e)),
                ket2dm(tensor(self.atom_e, self.atom_e))
            ])
        else:
            raise ValueError("Only N_atoms=1 or 2 are supported.")

    def omega_ij(self, i: int, j: int):
        """ transition frequency between eigenstates i and j """
        return self.eigenstates[0][i] - self.eigenstates[0][j]

    def summary(self):
        print("=== SystemParameters Summary ===")
        print(f"\n# The system with:")
        print(f"    {'N_atoms':<20}: {self.N_atoms}")

        print(f"\n# Frequencies and Dipole Moments:")
        for i in range(self.N_atoms):
            print(f"    Atom {i}: ω = {self.freqs_cm[i]} cm^-1, μ = {self.dip_moments[i]}")

        print(f"\n# Coupling / Inhomogeneity:")
        if self.N_atoms == 2:
            print(f"    {'J':<20}: {self.J_cm} cm^-1")
        print(f"    {'Delta':<20}: {self.Delta_cm} cm^-1")

        print(f"\n    {'psi_ini':<20}:")
        print(self.psi_ini)
        print(f"\n    {'System Hamiltonian (undiagonalized)':<20}:")
        print(self.H0_undiagonalized)

        print("\n# Dipole operator (Dip_op):")
        print(self.Dip_op)
        print("\n=== End of Summary ===")

    def to_dict(self):
        return {
            "N_atoms": self.N_atoms,
            "freqs_cm": self.freqs_cm,
            "dip_moments": self.dip_moments,
            "Delta_cm": self.Delta_cm,
            "J_cm": self.J_cm
        }

if __name__ == "__main__":
    print("Testing SystemParameters class...")
    print("\n=== Testing N_atoms=1 ===")
    system1 = SystemParameters(N_atoms=1)
    system1.summary()
    print("\n=== Testing N_atoms=2 ===")
    system2 = SystemParameters(N_atoms=2, freqs_cm=[16000.0, 15640.0], dip_moments=[1.0, 1.0])
    system2.summary()
    print("\n✅ SystemParameters tests completed successfully!")
