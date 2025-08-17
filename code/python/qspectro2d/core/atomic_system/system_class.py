# =============================
# DEFINE THE SYSTEM PARAMETERS CLASS
# =============================
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from functools import cached_property
from qutip import basis, ket2dm, Qobj
from qspectro2d.constants import HBAR, convert_cm_to_fs


@dataclass
class AtomicSystem:
    # VITAL ATTRIBUTES
    n_atoms: int = 1
    n_chains: int = 1
    frequencies_cm: List[float] = field(default_factory=lambda: [16000.0])  # in cm^-1
    dip_moments: List[float] = field(default_factory=lambda: [1.0])
    coupling_cm: float = 0.0
    # 1 = existing behaviour (ground + single exc manifold); 2 adds double manifold
    max_excitation: int = 1

    psi_ini: Optional[Qobj] = None  # initial state, default is ground state
    delta_cm: Optional[float] = None  # inhomogeneous broadening, default is None

    def __post_init__(self):
        # Only basic validation - no complex setup
        self._validate_parameters()
        self._normalize_arrays()
        # Build basis (and mappings if needed) according to selected excitation truncation
        self._build_basis()

        # we always start in the ground state
        self.psi_ini = ket2dm(self.basis[0])

        # store the initial frequencies in history
        self._frequencies_cm_history = [self.frequencies_cm.copy()]

        # Always set cylindrical positions and compute isotropic couplings
        self.n_rings = self.n_atoms // self.n_chains
        self._setup_geometry_and_couplings()

    def _validate_parameters(self):
        """Simple parameter validation only."""
        if self.max_excitation not in (1, 2):
            raise ValueError("max_excitation must be 1 or 2")

    def _normalize_arrays(self):
        """Ensure arrays have correct length."""
        if len(self.frequencies_cm) == 1 and self.n_atoms > 1:
            self.frequencies_cm = (
                self.frequencies_cm * self.n_atoms
            )  # repeat for all atoms
        elif len(self.frequencies_cm) != self.n_atoms:
            raise ValueError(
                f"frequencies_cm has {len(self.frequencies_cm)} elements but n_atoms={self.n_atoms}. "
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

    def update_frequencies_cm(self, new_freqs: List[float]):
        if len(new_freqs) != self.n_atoms:
            raise ValueError(
                f"Expected {self.n_atoms} frequencies, got {len(new_freqs)}"
            )

        # Save current freqs before updating
        self._frequencies_cm_history.append(new_freqs.copy())
        self.frequencies_cm = new_freqs.copy()

        # Invalidate cached properties depending on frequencies
        self.reset_cache()

    def reset_cache(self):
        """Reset cached spectral/operator properties.

        Removes cached properties whose values depend on the intrinsic
        system parameters (frequencies, coupling, dipole moments). Use this
        after any *in-place* modification of such parameters. Note that
        :meth:`update_frequencies_cm` already calls this automatically.

        Cached attributes cleared:
        - eigenstates
        - lowering_op
        - dipole_op
        """
        for attr in ("eigenstates", "lowering_op", "dipole_op", "coupling_op"):
            if attr in self.__dict__:
                del self.__dict__[attr]

        # Recompute geometry and couplings if parameters changed
        if hasattr(self, "_positions"):
            self._compute_and_cache_couplings()

    # === CORE PARAMETERS ===
    @cached_property
    def dimension(self):
        N = self.n_atoms
        exc = self.max_excitation
        if exc == 1:
            dim = 1 + N
        elif exc == 2:
            n_pairs = N * (N - 1) // 2
            dim = 1 + N + n_pairs
        return dim

    @property
    def basis(self):
        return self._basis

    # Store everything in fs^-1 internally
    @property
    def frequencies(self) -> np.ndarray:
        """Frequencies in fs^-1 (internal)."""
        return np.array([convert_cm_to_fs(f) for f in self.frequencies_cm])

    @property
    def coupling(self) -> float:
        """Coupling in fs^-1 (internal)."""
        return convert_cm_to_fs(self.coupling_cm) if self.coupling_cm else 0.0

    @property
    def delta(self) -> float:
        """Return the inhomogeneous broadening, IF the simulation is run with a lot of atoms (fs^-1)."""
        return convert_cm_to_fs(self.delta_cm)

    @property
    def frequencies_cm_history(self):
        """Access history of all frequency lists (including current)."""
        return self._frequencies_cm_history

    # === QUANTUM OPERATORS ===
    @property
    def hamiltonian(self) -> Qobj:
        return self._build_hamiltonian()

    @property
    def lowering_op(self) -> Qobj:
        lowering_op = 0
        if self.max_excitation == 1:
            # Single-excitation lowering operator: sum_i μ_i |0><i|
            for i, mu in enumerate(self.dip_moments, start=1):
                lowering_op += mu * (self.basis[0] * self.basis[i].dag())
            return lowering_op
        # max_excitation == 2: add terms connecting double -> single manifolds
        # |0><i|
        for i, mu in enumerate(self.dip_moments, start=1):
            lowering_op += mu * (self.basis[0] * self.basis[i].dag())
        # |j><i,j| and |j><j,i| but only one ordering stored (i<j)
        for (i, j), idx in self._pair_to_index.items():  # i<j
            mu_i = self.dip_moments[i - 1]
            mu_j = self.dip_moments[j - 1]
            # Annihilating excitation on site i from |i,j> leaves |j>, and vice versa
            lowering_op += mu_i * (self.basis[j] * self.basis[idx].dag())
            lowering_op += mu_j * (self.basis[i] * self.basis[idx].dag())
        return lowering_op

    @property
    def dipole_op(self) -> Qobj:
        return self.lowering_op + self.lowering_op.dag()

    @cached_property
    def eigenstates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues & eigenstates (cached).

        Invalidated when `update_frequencies_cm` is called.
        """
        return self.hamiltonian.eigenstates()

    def coupling_op(self):
        """Build the inter-site coupling operator using isotropic couplings.

        Uses the coupling matrix computed from cylindrical geometry.
        Single-manifold couplings: |i><j| + |j><i| with strength ħ J_ij.
        If max_excitation == 2, also couples double states that share one site.
        """
        N = self.n_atoms
        if N <= 1:
            return 0

        # Use the isotropic coupling matrix
        J_matrix = self._coupling_matrix  # Already in fs^-1
        HJ = 0

        # Singles excitation couplings
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                Jij = float(J_matrix[i - 1, j - 1])
                if Jij == 0.0:
                    continue
                ket_i = self.basis[i]
                ket_j = self.basis[j]
                HJ += HBAR * Jij * (ket_i * ket_j.dag() + ket_j * ket_i.dag())
        return HJ

    def _setup_geometry_and_couplings(self):
        """Set up cylindrical geometry and compute isotropic couplings."""
        self._set_cylindrical_positions(
            distance=1.0
        )  # TODO this could be parameterized
        self._compute_and_cache_couplings()

    def _set_cylindrical_positions(self, distance: float):
        """Set cylindrical atom positions."""
        n_rings = self.n_rings
        n_chains = self.n_chains

        # Ring centers in xy-plane
        if n_chains == 1:
            # Linear chain
            positions = np.array([[0.0, 0.0, z * distance] for z in range(n_rings)])
        else:
            # Multiple chains in cylinder
            dphi = 2.0 * np.pi / n_chains
            radius = distance / (2.0 * np.sin(np.pi / n_chains))
            ring_centers = np.array(
                [
                    [radius * np.cos(k * dphi), radius * np.sin(k * dphi), 0.0]
                    for k in range(n_chains)
                ]
            )

            positions = np.array(
                [
                    ring_centers[c] + np.array([0.0, 0.0, z * distance])
                    for c in range(n_chains)
                    for z in range(n_rings)
                ]
            )

        self._positions = positions

    def _compute_and_cache_couplings(self):
        """Compute isotropic coupling matrix and cache it."""
        self._coupling_matrix = self._compute_isotropic_couplings()

    def _compute_isotropic_couplings(self, power: float = 3.0) -> np.ndarray:
        """Compute isotropic J_ij = coupling * μ_i * μ_j / r^power."""
        N = self.n_atoms
        pos = self._positions
        J = np.zeros((N, N), dtype=float)

        base_coupling_fs = self.coupling  # Convert cm^-1 to fs^-1

        for i in range(N):
            for j in range(i + 1, N):
                r_vec = pos[j] - pos[i]
                r = float(np.linalg.norm(r_vec))
                if r == 0:
                    raise ValueError("Duplicate positions encountered (zero distance).")

                # Isotropic coupling with dipole product
                coupling_ij = (
                    base_coupling_fs
                    * self.dip_moments[i]
                    * self.dip_moments[j]
                    / (r**power)
                )
                J[i, j] = coupling_ij
                J[j, i] = coupling_ij

        return J

    def _build_basis(self):
        """Construct basis depending on n_atoms and max_excitation.

        Conventions (1-indexed atom labels i,j in [1..N]):
          - Ground state index: 0
          - Single excitations: |i> mapped to index i (1..N) when max_excitation>=1
          - Double excitations: |i,j> (i<j) mapped to indices N + p where
                p runs from 1..N_pairs in lexicographic order over (i,j).

        For n_atoms == 2, the existing explicit 4-level basis is retained but unified
        with the generic mapping when max_excitation == 2.
        """
        # Generic cases
        N = self.n_atoms
        dim = self.dimension
        self._basis = [basis(dim, i) for i in range(dim)]
        # Build mapping (i<j) -> index
        pair_index_start = 1 + N  # first double-excitation basis index
        pair_to_index = {}
        index_to_pair = {}
        idx = pair_index_start
        for i in range(1, N):
            for j in range(i + 1, N + 1):
                pair_to_index[(i, j)] = idx
                index_to_pair[idx] = (i, j)
                idx += 1
        self._pair_to_index = pair_to_index
        self._index_to_pair = index_to_pair

    def _build_hamiltonian(self) -> Qobj:
        """Return Hamiltonian including up to two excitations (hard-core boson model / fock-basis).

        H = Σ_i ħ ω_i a_i^† a_i + Σ_{i≠j} ħ J_ij a_i^† a_j, truncated to 0,1,2 excitations.

        Basis ordering (see _build_basis):
          0            : |0>
          1..N         : |i>
          N+1 .. end   : |i,j> with i<j in lexicographic order

        Matrix elements:
          - Diagonal singles: <i|H|i> = ħ ω_i
          - Diagonal doubles: <ij|H|ij> = ħ (ω_i + ω_j)
          - Single-single off-diagonals: ħ J_ij (i≠j)
          - Double-double couplings: if states share exactly one site, e.g. |i,j> ↔ |i,k>, element = ħ J_jk
            (other elements zero in this simple Frenkel model)
        """
        N = self.n_atoms
        omegas = self.frequencies

        # Helper lambdas for kets
        def ket(idx: int):
            return self._basis[idx]

        H = 0
        # Singles diagonal
        for i in range(1, N + 1):
            H += HBAR * omegas[i - 1] * ket(i) * ket(i).dag()
        J_fs = self.coupling_op()  # Call the method
        H += J_fs

        # If only max_excitation==1 we stop here
        if self.max_excitation == 1:
            return H

        # Double-excitation diagonal terms
        for idx, (i, j) in self._index_to_pair.items():  # idx >= 1+N
            H += HBAR * (omegas[i - 1] + omegas[j - 1]) * ket2dm(ket(idx))
        return H

    # === GEOMETRY AND POSITIONS ===
    @property
    def positions(self) -> np.ndarray:
        """Return atom positions (always set during initialization)."""
        return self._positions.copy()

    @property
    def coupling_matrix_cm(self) -> np.ndarray:
        """Return coupling matrix in cm^-1 for display."""
        return self._coupling_matrix / convert_cm_to_fs(1.0)

    def excitation_number_from_index(self, idx: int) -> int:
        if idx == 0:
            return 0
        elif 1 <= idx <= self.n_atoms:
            return 1
        else:
            return 2

    def deph_op_i(self, i: int) -> Qobj:
        """Return site i population operator in the site basis (|i><i|)."""
        if i == 0:
            raise ValueError("indexing ground state -> use i elem 1,...,N+1")
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
                f"    Atom {i}: ω = {self.frequencies_cm[i]} cm^-1, μ = {self.dip_moments[i]}"
            )
        lines.append("\n# Coupling / Inhomogeneity:")
        if self.n_atoms == 2:
            if self.coupling_cm is not None:
                lines.append(f"    {'coupling':<20}: {self.coupling_cm} cm^-1")
            if self.delta_cm is not None:
                lines.append(f"    {'delta':<20}: {self.delta_cm} cm^-1")
        elif self.n_atoms > 2 and self.n_rings is not None:
            lines.append(
                f"    {'n_rings':<20}: {self.n_rings} (n_chains = {self.n_chains})"
            )
            lines.append(f"    {'positions shape':<20}: {self.positions.shape}")
            lines.append(f"    {'coupling matrix (cm^-1)':<20}:")
            lines.append(str(self.coupling_matrix_cm))
        lines.append(f"\n    {'psi_ini':<20}:")
        lines.append(str(self.psi_ini))
        lines.append(f"\n    {'System Hamiltonian (undiagonalized)':<20}:")
        lines.append(str(self.hamiltonian))
        lines.append("\n# Dipole operator (dipole_op):")
        lines.append(str(self.dipole_op))
        lines.append("\n=== End of Summary ===")
        return "\n".join(lines)

    def __str__(self) -> str:
        # Return string representation without side effects (used by print())
        return self.summary()

    def to_dict(self):
        d = {
            "n_atoms": self.n_atoms,
            "frequencies_cm": self.frequencies_cm,
            "dip_moments": self.dip_moments,
        }
        if self.delta_cm is not None:
            d["delta_cm"] = self.delta_cm
        if self.coupling_cm is not None:
            d["coupling_cm"] = self.coupling_cm
        return d

    def to_json(self):
        """
        Serialize the system parameters to a JSON string.

        Only basic attributes are included: n_atoms, frequencies_cm, dip_moments, delta_cm, coupling_cm.
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
