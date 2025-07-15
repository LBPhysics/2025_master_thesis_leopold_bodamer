# =============================
# DEFINE THE SYSTEM PARAMETERS CLASS
# =============================
import numpy as np
import json
from dataclasses import dataclass, field  # for the class definiton
from typing import Optional, List
from qutip import basis, ket2dm, tensor, Qobj
from qspectro2d.core.utils_and_config import convert_cm_to_fs, HBAR

# from functools import cached_property  # for caching properties


@dataclass
class AtomicSystem:
    # VITAL ATTRIBUTES
    n_atoms: int = 1
    freqs_cm: List[float] = field(default_factory=lambda: [16000.0])  # in cm^-1
    dip_moments: List[float] = field(default_factory=lambda: [1.0])
    J_cm: Optional[float] = None  # only For n_atoms >= 2

    psi_ini: Optional[Qobj] = None  # initial state, default is ground state
    delta_cm: Optional[float] = None  # inhomogeneous broadening, default is None

    @property
    def basis(self):
        return self._basis

    def __post_init__(self):
        # mostly for validation and initialization
        # Cache ground and excited states for single atom
        self._atom_g = basis(2, 0)  # ground state |g>
        self._atom_e = basis(2, 1)  # excited state |e>

        # Handle case where single frequency is provided for multiple atoms
        if len(self.freqs_cm) == 1 and self.n_atoms > 1:
            self.freqs_cm = self.freqs_cm * self.n_atoms  # repeat for all atoms
        elif len(self.freqs_cm) != self.n_atoms:
            raise ValueError(
                f"freqs_cm has {len(self.freqs_cm)} elements but n_atoms={self.n_atoms}. "
                f"Expected either 1 frequency (applied to all atoms) or {self.n_atoms} frequencies."
            )

        # Handle case where single dipole moment is provided for multiple atoms
        if len(self.dip_moments) == 1 and self.n_atoms > 1:
            self.dip_moments = self.dip_moments * self.n_atoms  # repeat for all atoms
        elif len(self.dip_moments) != self.n_atoms:
            raise ValueError(
                f"dip_moments has {len(self.dip_moments)} elements but n_atoms={self.n_atoms}. "
                f"Expected either 1 dipole moment (applied to all atoms) or {self.n_atoms} dipole moments."
            )

        # store the initial frequencies in history
        self._freqs_cm_history = [self.freqs_cm.copy()]

        if self.n_atoms >= 2 and self.J_cm is None:
            self.J_cm = 0.0

        # set a default basis
        if self.n_atoms == 1:
            self._basis = [self._atom_g, self._atom_e]  # GROUND, EXCITED
        elif self.n_atoms == 2:
            self._basis = [
                tensor(self._atom_g, self._atom_g),  # GROUND
                tensor(self._atom_e, self._atom_g),  # A
                tensor(self._atom_g, self._atom_e),  # B
                tensor(self._atom_e, self._atom_e),  # AB
            ]
        else:  # SINGLE EXCITATION SUBSPACE FOR n_atoms > 2
            n_atoms = self.n_atoms
            self._basis = [
                basis(n_atoms, i) for i in range(n_atoms)
            ]  # GROUND, atom 1, atom 2, ...

        if self.psi_ini is None:
            self.psi_ini = ket2dm(self.basis[0])

    def update_freqs_cm(self, new_freqs: List[float]):
        if len(new_freqs) != self.n_atoms:
            raise ValueError(
                f"Expected {self.n_atoms} frequencies, got {len(new_freqs)}"
            )

        # Save current freqs before updating
        self._freqs_cm_history.append(new_freqs.copy())
        self.freqs_cm = new_freqs.copy()

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
        return convert_cm_to_fs(self.delta_cm)

    def _Hamilton_tls(self):
        return HBAR * self.freqs_fs(0) * ket2dm(self.basis[1])

    def _Hamilton_dimer_sys(self):
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

    @property
    def H0_N_canonical(self):
        n_atoms = self.n_atoms
        if n_atoms == 1:
            return self._Hamilton_tls()
        elif n_atoms == 2:
            return self._Hamilton_dimer_sys()
        else:  # TODO IMPLEMENT THE n_atoms > 2 CASE IN SINGLE EXCITATION SUBSPACE
            H = 0
            '''

            # =============================
            # GEOMETRY DEFINITIONS
            # =============================
            def chain_positions(distance, n_atoms):
                """ Generate atomic positions in a linear chain. """
                return np.array([[0, 0, i * distance] for i in range(n_atoms)])

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

            def cyl_positions(distance, n_atoms, n_chains):
                """ Generate atomic positions in a cylindrical structure. """
                Pos_chain = chain_positions(distance, n_atoms // n_chains)
                Pos_ring = ring_positions(distance, n_chains)
                return np.vstack([Pos_chain + Pos_ring[i] for i in range(n_chains)])

                
            ALPHA = 1e-3 # Coupling strength of dipoles (Fine structure constant?)
            n_chains = 1                    # Number of chains
            n_rings = 1                     # Number of rings
            n_atoms = n_chains * n_rings
            Pos = cyl_positions(distance, n_atoms, n_chains)
            atom_frequencies = [omega_a]*n_atoms # sample_frequencies(omega_a, 0.0125 * omega_a, n_atoms)
            for a in range(n_atoms):
                for b in range(n_atoms):
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

    @property  # cached -> to safe some space? TODO
    def eigenstates(self):
        return self.H0_N_canonical.eigenstates()

    @property
    def sm_op(self):
        if self.n_atoms == 1:
            return self.dip_moments[0] * (self._atom_g * self._atom_e.dag())
        elif self.n_atoms == 2:
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
        else:  # TODO IMPLEMENT THE n_atoms > 2 CASE IN SINGLE EXCITATION SUBSPACE
            raise NotImplementedError("n_atoms > 2 not yet implemented")

    @property
    def dip_op(self):
        return self.sm_op + self.sm_op.dag()

    def deph_op_i(self, i: int):
        """Return dephasing operator for the i-th eigenstate. i elem (1, ..., n_atoms)."""
        return ket2dm(self.basis[i])

    def omega_ij(self, i: int, j: int):
        """Return energy difference (frequency) between eigenstates i and j in fs^-1."""
        return self.eigenstates[0][i] - self.eigenstates[0][j]

    def summary(self):
        print("=== AtomicSystem Summary ===")
        print(f"\n# The system with:")
        print(f"    {'n_atoms':<20}: {self.n_atoms}")

        print(f"\n# Frequencies and Dipole Moments:")
        for i in range(self.n_atoms):
            print(
                f"    Atom {i}: ω = {self.freqs_cm[i]} cm^-1, μ = {self.dip_moments[i]}"
            )

        print(f"\n# Coupling / Inhomogeneity:")
        if self.n_atoms == 2:
            if self.J_cm is not None:
                print(f"    {'J':<20}: {self.J_cm} cm^-1")
            if self.delta_cm is not None:
                print(f"    {'Delta':<20}: {self.delta_cm} cm^-1")

        print(f"\n    {'psi_ini':<20}:")
        print(self.psi_ini)

        print(f"\n    {'System Hamiltonian (undiagonalized)':<20}:")
        print(self.H0_N_canonical)

        print("\n# Dipole operator (dip_op):")
        print(self.dip_op)
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
            "n_atoms": self.n_atoms,
            "freqs_cm": self.freqs_cm,
            "dip_moments": self.dip_moments,
        }
        if self.delta_cm is not None:
            d["delta_cm"] = self.delta_cm
        if self.J_cm is not None:
            d["J_cm"] = self.J_cm
        return d

    def to_json(self):
        """
        Serialize the system parameters to a JSON string.

        Only basic attributes are included: n_atoms, freqs_cm, dip_moments, delta_cm, J_cm.
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
# USAGE EXAMPLES
# =============================
""" HOW TO USE AtomicSystem:

# Create a single atom system
system1 = AtomicSystem(n_atoms=1, freqs_cm=[16000.0], dip_moments=[1.0])

# Create a two-atom system with coupling
system2 = AtomicSystem(
    n_atoms=2, 
    freqs_cm=[16000.0, 15640.0], 
    dip_moments=[1.0, 1.2],
    J_cm=50.0
)

# Serialize to JSON for saving/loading
json_str = system2.to_json()
system2_loaded = AtomicSystem.from_json(json_str)

# View system properties
system2.summary()

# Access computed properties
eigenvals, eigenvecs = system2.eigenstates
hamiltonian = system2.H0_N_canonical
dipole_op = system2.dip_op
"""

if __name__ == "__main__":
    print("AtomicSystem class loaded successfully!")
    print("Run 'python test_system_class.py' to execute tests.")
    print("\nFor usage examples, see the docstring above.")
