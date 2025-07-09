# =============================
# DEFINE THE SYSTEM PARAMETERS CLASS
# =============================
import numpy as np
import json
from dataclasses import dataclass, field  # for the class definiton
from typing import Optional, List
from qutip import basis, ket2dm, tensor, Qobj
from qspectro2d.core.utils_and_config import convert_cm_to_fs, HBAR
from functools import cached_property  # for caching properties


@dataclass
class AtomicSystem:
    # VITAL ATTRIBUTES
    N_atoms: int = 1
    freqs_cm: List[float] = field(default_factory=lambda: [16000.0])  # in cm^-1
    dip_moments: List[float] = field(default_factory=lambda: [1.0])
    J_cm: Optional[float] = None  # only For N_atoms >= 2
    psi_ini: Optional[Qobj] = None  # initial state, default is ground state
    Delta_cm: Optional[float] = None  # inhomogeneous broadening, default is None

    @property
    def atom_g(self):
        return basis(2, 0)

    @property
    def atom_e(self):
        return basis(2, 1)

    @property
    def basis(self):
        return self._basis

    def __post_init__(self):
        if len(self.freqs_cm) != self.N_atoms:
            raise ValueError(
                f"freqs_cm has {len(self.freqs_cm)} elements but N_atoms={self.N_atoms}. "
                f"Expected {self.N_atoms} frequencies."
            )
        # store the initial frequencies in history
        self._freqs_cm_history = [self.freqs_cm.copy()]

        if len(self.dip_moments) != self.N_atoms:
            raise ValueError("Length of dip_moments must match N_atoms")

        if self.N_atoms == 2 and self.J_cm is None:
            self.J_cm = 0.0

        # If basis is not provided, set a default basis (optional)
        if self.N_atoms == 1:
            self._basis = [self.atom_g, self.atom_e]  # GROUND, EXCITED
        elif self.N_atoms == 2:
            self._basis = [
                tensor(self.atom_g, self.atom_g),  # GROUND
                tensor(self.atom_e, self.atom_g),  # A
                tensor(self.atom_g, self.atom_e),  # B
                tensor(self.atom_e, self.atom_e),  # AB
            ]
        else:
            N_atoms = self.N_atoms
            self._basis = [
                basis(N_atoms, i) for i in range(N_atoms)
            ]  # GROUND, atom 1, atom 2, ...

        self.H0_undiagonalized = self.Hamilton_N_atoms()

        self.psi_ini = ket2dm(self.basis[0])

    def update_freqs_cm(self, new_freqs: List[float]):
        if len(new_freqs) != self.N_atoms:
            raise ValueError(
                f"Expected {self.N_atoms} frequencies, got {len(new_freqs)}"
            )

        # Save current freqs before updating
        self._freqs_cm_history.append(new_freqs.copy())
        self.freqs_cm = new_freqs.copy()

        # Recompute Hamiltonian and eigenstates
        self.H0_undiagonalized = self.Hamilton_N_atoms()

        if "eigenstates" in self.__dict__:
            del self.__dict__["eigenstates"]  # reset cached property

    @property
    def freqs_cm_history(self):
        """Access history of all frequency lists (including current)."""
        return self._freqs_cm_history

    def freqs_fs(self, i):
        """Return frequency in fs^-1 for the i-th atom."""
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
        return HBAR * self.freqs_fs(0) * ket2dm(self.basis[1])

    def Hamilton_dimer_sys(self):
        H = HBAR * (
            self.freqs_fs(0) * ket2dm(self.basis[1])
            + self.freqs_fs(1) * ket2dm(self.basis[2])
            + self.J
            * (
                self.basis[1] * self.basis[2].dag()
                + self.basis[2] * self.basis[1].dag()
            )
            + (self.freqs_fs(0) + self.freqs_fs(1)) * ket2dm(self.basis[3])
        )
        return H

    def Hamilton_N_atoms(self):
        N_atoms = self.N_atoms
        if N_atoms == 1:
            return self.Hamilton_tls()
        elif N_atoms == 2:
            return self.Hamilton_dimer_sys()
        else:  # TODO IMPLEMENT THE N_atoms > 2 CASE IN SINGLE EXCITATION SUBSPACE
            H = 0
            '''

            # =============================
            # GEOMETRY DEFINITIONS
            # =============================
            def chain_positions(distance, N_atoms):
                """ Generate atomic positions in a linear chain. """
                return np.array([[0, 0, i * distance] for i in range(N_atoms)])

            def z_rotation(angle):
                """ Generate a 3D rotation matrix around the z-axis. """
                return np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])

            def ring_positions(distance, n_chains):
                """ Generate atomic positions in a ring. """
                dphi = 2 * np.pi / n_chains
                radius = 0 if n_chains == 1 else distance / (2 * np.sin(np.pi / n_chains))
                return np.array([z_rotation(dphi * i) @ [radius, 0, 0] for i in range(n_chains)])

            def cyl_positions(distance, N_atoms, n_chains):
                """ Generate atomic positions in a cylindrical structure. """
                Pos_chain = chain_positions(distance, N_atoms // n_chains)
                Pos_ring = ring_positions(distance, n_chains)
                return np.vstack([Pos_chain + Pos_ring[i] for i in range(n_chains)])

                
            ALPHA = 1e-3 # Coupling strength of dipoles (Fine structure constant?)
            n_chains = 1                    # Number of chains
            n_rings = 1                     # Number of rings
            N_atoms = n_chains * n_rings
            Pos = cyl_positions(distance, N_atoms, n_chains)
            atom_frequencies = [omega_a]*N_atoms # sample_frequencies(omega_a, 0.0125 * omega_a, N_atoms)
            for a in range(N_atoms):
                for b in range(N_atoms):
                    sm_a = self.basis[0]*self.basis[a].dag()
                    sm_b = self.basis[0]*self.basis[b].dag()
                    factor = self.dip_moments[a] * self.dip_moments[b]
                    op = factor * sm_a.dag() * sm_b
                    if a != b:
                        H += ALPHA / (np.linalg.norm(Pos[a] - Pos[b]))**3 * op
                    else:
                        H += atom_frequencies[a] * op
            '''
            return H

    @cached_property  # from functools
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
            return sum(
                [
                    mu_10 * (eigenvecs[0] * eigenvecs[1].dag()),
                    mu_20 * (eigenvecs[0] * eigenvecs[2].dag()),
                    mu_31 * (eigenvecs[1] * eigenvecs[3].dag()),
                    mu_32 * (eigenvecs[2] * eigenvecs[3].dag()),
                ]
            )
        else:  # TODO IMPLEMENT THE N_atoms > 2 CASE IN SINGLE EXCITATION SUBSPACE
            raise NotImplementedError("N_atoms > 2 not yet implemented")

    @property
    def Dip_op(self):
        return self.SM_op + self.SM_op.dag()

    @property
    def Deph_op(
        self,
    ):
        if self.N_atoms == 1:
            return ket2dm(self.atom_e)
        elif self.N_atoms == 2:
            return sum(
                [
                    ket2dm(tensor(self.atom_e, self.atom_g)),
                    ket2dm(tensor(self.atom_g, self.atom_e)),
                    ket2dm(tensor(self.atom_e, self.atom_e)),
                ]
            )
        else:  # TODO OVERTHINK / IMPLEMENT THE N_atoms > 2 CASE IN SINGLE EXCITATION SUBSPACE
            return sum([ket2dm(self.basis[i]) for i in range(1, self.N_atoms)])

    def omega_ij(self, i: int, j: int):
        """Return energy difference (frequency) between eigenstates i and j in fs^-1."""
        return self.eigenstates[0][i] - self.eigenstates[0][j]

    def summary(self):
        print("=== AtomicSystem Summary ===")
        print(f"\n# The system with:")
        print(f"    {'N_atoms':<20}: {self.N_atoms}")

        print(f"\n# Frequencies and Dipole Moments:")
        for i in range(self.N_atoms):
            print(
                f"    Atom {i}: ω = {self.freqs_cm[i]} cm^-1, μ = {self.dip_moments[i]}"
            )

        print(f"\n# Coupling / Inhomogeneity:")
        if self.N_atoms == 2:
            if self.J_cm is not None:
                print(f"    {'J':<20}: {self.J_cm} cm^-1")
            if self.Delta_cm is not None:
                print(f"    {'Delta':<20}: {self.Delta_cm} cm^-1")

        if self.psi_ini is not None:
            print(f"\n    {'psi_ini':<20}:")
            print(self.psi_ini)
        if self.H0_undiagonalized is not None:
            print(f"\n    {'System Hamiltonian (undiagonalized)':<20}:")
            print(self.H0_undiagonalized)

        if self.Dip_op is not None:
            print("\n# Dipole operator (Dip_op):")
            print(self.Dip_op)
        print("\n=== End of Summary ===")

    def __str__(self) -> str:
        from io import StringIO
        import sys

        # Capture the output of the summary method
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        self.summary()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        return output.strip()

    def to_dict(self):
        d = {
            "N_atoms": self.N_atoms,
            "freqs_cm": self.freqs_cm,
            "dip_moments": self.dip_moments,
        }
        if self.Delta_cm is not None:
            d["Delta_cm"] = self.Delta_cm
        if self.J_cm is not None:
            d["J_cm"] = self.J_cm
        return d

    def to_json(self):
        """
        Serialize the system parameters to a JSON string.

        Only basic attributes are included: N_atoms, freqs_cm, dip_moments, Delta_cm, J_cm.
        Quantum objects (Qobj) and computed properties (like Hamiltonians or eigenstates)
        are not serialized and will be recomputed on deserialization.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d):
        """
        Create a AtomicSystem object from a dictionary of parameters.

        Expects keys to match those produced by to_dict().
        """
        return cls(**d)

    @classmethod
    def from_json(cls, json_str):
        """
        Deserialize a JSON string into a AtomicSystem object.

        Only reconstructs basic attributes. Complex internal states are recomputed on init.
        """
        d = json.loads(json_str)
        return cls.from_dict(d)


# =============================
# TESTING THE SYSTEM PARAMETERS CLASS
# =============================
""" HOW TO REPLICATE ONE:
# Create an object
sp = AtomicSystem(N_atoms=2, freqs_cm=[16000, 16100], dip_moments=[1.0, 1.2])

# Serialize to JSON
json_str = sp.to_json()

# Deserialize it
sp2 = AtomicSystem.from_json(json_str)
"""
if __name__ == "__main__":
    print("Testing AtomicSystem class...")
    print("\n=== Testing N_atoms=1 ===")
    system1 = AtomicSystem(N_atoms=1)
    system1.summary()
    print("\n=== Testing N_atoms=2 ===")
    system2 = AtomicSystem(
        N_atoms=2, freqs_cm=[16000.0, 15640.0], dip_moments=[1.0, 1.0]
    )
    system2.summary()
    print("\n✅ AtomicSystem tests completed successfully!")

    print("\n=== Testing JSON serialization ===")
    json_str = system1.to_json()
    print("Serialized JSON string:")
    print(json_str)
