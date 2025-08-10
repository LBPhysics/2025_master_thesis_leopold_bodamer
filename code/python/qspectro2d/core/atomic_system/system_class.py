# =============================
# DEFINE THE SYSTEM PARAMETERS CLASS
# =============================
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from functools import cached_property
from qutip import basis, ket2dm, tensor, Qobj
from qspectro2d.constants import HBAR, convert_cm_to_fs


@dataclass
class AtomicSystem:
    # VITAL ATTRIBUTES
    n_atoms: int = 1
    at_freqs_cm: List[float] = field(default_factory=lambda: [16000.0])  # in cm^-1
    dip_moments: List[float] = field(default_factory=lambda: [1.0])
    at_coupling_cm: Optional[float] = None  # only For n_atoms >= 2

    psi_ini: Optional[Qobj] = None  # initial state, default is ground state
    delta_cm: Optional[float] = None  # inhomogeneous broadening, default is None

    def __post_init__(self):
        # mostly for validation and initialization
        # Cache ground and excited states for single atom
        self._atom_g = basis(2, 0)  # ground state |g>
        self._atom_e = basis(2, 1)  # excited state |e>

        # Handle case where single frequency is provided for multiple atoms
        if len(self.at_freqs_cm) == 1 and self.n_atoms > 1:
            self.at_freqs_cm = self.at_freqs_cm * self.n_atoms  # repeat for all atoms
        elif len(self.at_freqs_cm) != self.n_atoms:
            raise ValueError(
                f"at_freqs_cm has {len(self.at_freqs_cm)} elements but n_atoms={self.n_atoms}. "
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
        self._freqs_cm_history = [self.at_freqs_cm.copy()]

        if self.n_atoms >= 2 and self.at_coupling_cm is None:
            self.at_coupling_cm = 0.0

        # set a default basis
        if self.n_atoms == 1:
            self._basis = [self._atom_g, self._atom_e]  # ground, excited
        elif self.n_atoms == 2:
            self._basis = [
                tensor(self._atom_g, self._atom_g),  # ground
                tensor(self._atom_e, self._atom_g),  # A
                tensor(self._atom_g, self._atom_e),  # B
                tensor(self._atom_e, self._atom_e),  # AB
            ]
        else:  # single excitation subspace n_atoms > 2
            n_atoms = self.n_atoms
            self._basis = [
                basis(n_atoms, i) for i in range(n_atoms)
            ]  # fock basis: ground, atom 1, atom 2, ...

        if self.psi_ini is None:
            self.psi_ini = ket2dm(self.basis[0])

    def update_freqs_cm(self, new_freqs: List[float]):
        if len(new_freqs) != self.n_atoms:
            raise ValueError(
                f"Expected {self.n_atoms} frequencies, got {len(new_freqs)}"
            )

        # Save current freqs before updating
        self._freqs_cm_history.append(new_freqs.copy())
        self.at_freqs_cm = new_freqs.copy()

        # Invalidate cached properties depending on frequencies
        self.reset_cache()

    def reset_cache(self):
        """Reset cached spectral/operator properties.

        Removes cached properties whose values depend on the intrinsic
        system parameters (frequencies, coupling, dipole moments). Use this
        after any *in-place* modification of such parameters. Note that
        :meth:`update_freqs_cm` already calls this automatically.

        Cached attributes cleared:
        - eigenstates
        - sm_op
        - dip_op
        """
        for attr in ("eigenstates", "sm_op", "dip_op"):
            if attr in self.__dict__:
                del self.__dict__[attr]

    @property
    def basis(self):
        return self._basis

    @property
    def freqs_cm_history(self):
        """Access history of all frequency lists (including current)."""
        return self._freqs_cm_history

    def at_freqs_fs(self, i: int) -> float:
        """Return frequency in fs^-1 for the i-th atom."""
        return convert_cm_to_fs(self.at_freqs_cm[i])

    @property
    def at_coupling(self) -> float:
        return convert_cm_to_fs(self.at_coupling_cm)

    @property
    def theta(self) -> float:
        """Return dimer mixing angle θ (radians) for n_atoms == 2.

        Definition (standard exciton / coupled two-level system):

            tan(2θ) = 2J / Δ

        where
            J  = at_coupling (fs^-1)
            Δ  = ω_1 - ω_2 (fs^-1)  (bare transition frequency detuning)

        We compute:
            θ = 0.5 * arctan2(2J, Δ)

        Range: θ ∈ (-π/4, π/4]; magnitude governs the degree of state mixing.
        The previous implementation used arctan(J/Δ), which differs by a
        factor-of-two in the argument and is non-standard. Coefficients used
        in `sm_op` retain their algebraic definitions (sinθ, cosθ), so this
        correction yields physically consistent mixing when J ~ Δ.

        Raises
        ------
        ValueError: if called for systems with n_atoms != 2.
        """
        if self.n_atoms != 2:
            raise ValueError("theta is only defined for n_atoms == 2")
        detuning = self.at_freqs_fs(0) - self.at_freqs_fs(1)  # Δ
        return 0.5 * np.arctan2(2 * self.at_coupling, detuning)

    @property
    def Delta(self) -> float:
        return convert_cm_to_fs(self.delta_cm)

    def _Hamilton_tls(self) -> Qobj:
        return HBAR * self.at_freqs_fs(0) * ket2dm(self.basis[1])

    def _Hamilton_dimer_sys(self) -> Qobj:
        H = HBAR * (
            self.at_freqs_fs(0) * ket2dm(self.basis[1])
            + self.at_freqs_fs(1) * ket2dm(self.basis[2])
            + self.at_coupling
            * (
                self.basis[1] * self.basis[2].dag()
                + self.basis[2] * self.basis[1].dag()
            )
            + (self.at_freqs_fs(0) + self.at_freqs_fs(1)) * ket2dm(self.basis[3])
        )
        return H

    @property
    def H0_N_canonical(self) -> Qobj:
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

                
            BATH_COUPLING = 1e-3 # Coupling strength of dipoles (Fine structure constant?)
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
                        H += BATH_COUPLING / (np.linalg.norm(Pos[a] - Pos[b]))**3 * op
                    else:
                        H += atom_frequencies[a] * op
            '''
            return H

    @cached_property
    def eigenstates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues & eigenstates (cached).

        Invalidated when `update_freqs_cm` is called.
        """
        return self.H0_N_canonical.eigenstates()

    @cached_property
    def sm_op(self) -> Qobj:
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

    @cached_property
    def dip_op(self) -> Qobj:
        return self.sm_op + self.sm_op.dag()

    def deph_op_i(self, i: int) -> Qobj:
        """Return dephasing operator for the i-th eigenstate. i elem (1, ..., n_atoms)."""
        return ket2dm(self.basis[i])

    def omega_ij(self, i: int, j: int) -> float:
        """Return energy difference (frequency) between eigenstates i and j in fs^-1."""
        return self.eigenstates[0][i] - self.eigenstates[0][j]

    def summary(self):
        lines = [
            "=== AtomicSystem Summary ===",
            "",
            "# The system with:",
            f"    {'n_atoms':<20}: {self.n_atoms}",
        ]
        lines.append("\n# Frequencies and Dipole Moments:")
        for i in range(self.n_atoms):
            lines.append(
                f"    Atom {i}: ω = {self.at_freqs_cm[i]} cm^-1, μ = {self.dip_moments[i]}"
            )
        lines.append("\n# Coupling / Inhomogeneity:")
        if self.n_atoms == 2:
            if self.at_coupling_cm is not None:
                lines.append(f"    {'at_coupling':<20}: {self.at_coupling_cm} cm^-1")
            if self.delta_cm is not None:
                lines.append(f"    {'Delta':<20}: {self.delta_cm} cm^-1")
        lines.append(f"\n    {'psi_ini':<20}:")
        lines.append(str(self.psi_ini))
        lines.append(f"\n    {'System Hamiltonian (undiagonalized)':<20}:")
        lines.append(str(self.H0_N_canonical))
        lines.append("\n# Dipole operator (dip_op):")
        lines.append(str(self.dip_op))
        lines.append("\n=== End of Summary ===")
        return "\n".join(lines)

    def __str__(self) -> str:
        # Return string representation without side effects (used by print())
        return self.summary()

    def to_dict(self):
        d = {
            "n_atoms": self.n_atoms,
            "at_freqs_cm": self.at_freqs_cm,
            "dip_moments": self.dip_moments,
        }
        if self.delta_cm is not None:
            d["delta_cm"] = self.delta_cm
        if self.at_coupling_cm is not None:
            d["at_coupling_cm"] = self.at_coupling_cm
        return d

    def to_json(self):
        """
        Serialize the system parameters to a JSON string.

        Only basic attributes are included: n_atoms, at_freqs_cm, dip_moments, delta_cm, at_coupling_cm.
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
system1 = AtomicSystem(n_atoms=1, at_freqs_cm=[16000.0], dip_moments=[1.0])

# Create a two-atom system with coupling
system2 = AtomicSystem(
    n_atoms=2, 
    at_freqs_cm=[16000.0, 15640.0], 
    dip_moments=[1.0, 1.2],
    at_coupling_cm=50.0
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
