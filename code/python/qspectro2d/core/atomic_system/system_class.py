# =============================
# DEFINE THE SYSTEM PARAMETERS CLASS
# =============================
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
from functools import cached_property
from qutip import basis, ket2dm, tensor, Qobj
from qspectro2d.constants import HBAR, convert_cm_to_fs


@dataclass
class AtomicSystem:
    # VITAL ATTRIBUTES
    n_atoms: int = 1
    n_rings: Optional[int] = (
        None  # If n_atoms>2 and None -> defaults to linear chain: n_chains=1, n_rings=n_atoms
    )
    at_freqs_cm: List[float] = field(default_factory=lambda: [16000.0])  # in cm^-1
    dip_moments: List[float] = field(default_factory=lambda: [1.0])
    at_coupling_cm: Optional[float] = (
        None  # legacy uniform coupling (used for n_atoms == 2 or fallback)
    )
    max_excitation: int = (
        1  # 1 = existing behaviour (ground + single manifold); 2 adds double manifold
    )

    psi_ini: Optional[Qobj] = None  # initial state, default is ground state
    delta_cm: Optional[float] = None  # inhomogeneous broadening, default is None

    def __post_init__(self):
        # mostly for validation and initialization
        # Cache ground and excited states for single atom
        self._atom_g = basis(2, 0)  # ground state |g>
        self._atom_e = basis(2, 1)  # excited state |e>
        # Internal storage for couplings & geometry (not part of public dataclass fields)
        self._positions: Optional[np.ndarray] = None

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

        # Validate max_excitation
        if self.max_excitation not in (1, 2):
            raise ValueError("max_excitation must be 1 or 2")

        # Build basis (and mappings if needed) according to selected excitation truncation
        self._build_basis()

        if self.psi_ini is None:
            self.psi_ini = ket2dm(self.basis[0])

        # Geometry spec for multi-atom (N>2) systems
        # If user omitted n_rings -> default to a single linear chain: n_chains=1, n_rings=n_atoms
        self._n_chains: Optional[int] = None
        if self.n_atoms > 2:
            if self.n_rings is None:
                self.n_rings = self.n_atoms  # all sites along z
                self._n_chains = 1
            else:
                if self.n_rings < 1:
                    raise ValueError("n_rings must be >= 1")
                if self.n_atoms % self.n_rings != 0:
                    raise ValueError(
                        f"n_rings={self.n_rings} does not divide n_atoms={self.n_atoms}."
                    )
                self._n_chains = self.n_atoms // self.n_rings
            self.set_cylindrical_geometry(distance=1.0)

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
        for attr in ("eigenstates", "sm_op", "dip_op", "coupling_operator"):
            if attr in self.__dict__:
                del self.__dict__[attr]

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
    def theta(
        self,
    ) -> (
        float
    ):  # NOTE could also move this to system_bath_class as it is just needed for the paper
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

    # Removed duplicate per-case Hamiltonian helpers in favor of unified logic

    # =============================
    # SINGLE-EXCITATION HAMILTONIAN FOR n_atoms > 2
    # =============================
    @cached_property
    def coupling_operator(self):
        """Build the system-bath-free inter-site coupling operator from J_ij.

        Priority:
        1) If a coupling matrix has been set (e.g., via geometry), use it.
        2) Else, if uniform coupling is provided (at_coupling_cm), use J_ij = const for i!=j.
        3) Else, no coupling.

        Includes:
        - Single-manifold couplings: |i><j| + |j><i| with strength ħ J_ij.
        - If max_excitation == 2, also couples double states that share one site:
          |i,k> <-> |j,k| with strength ħ J_ij.
        """
        N = self.n_atoms
        if N <= 1:
            return 0

        J = self.coupling_matrix  # shape (N,N)
        # Start in the truncated Hilbert space (0,1,2)-excitation basis
        HJ = 0

        # Singles couplings
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                Jij = float(J[i - 1, j - 1])
                if Jij == 0.0:
                    continue
                ket_i = self.basis[i]
                ket_j = self.basis[j]
                HJ += HBAR * Jij * (ket_i * ket_j.dag() + ket_j * ket_i.dag())
        """
        # Double-manifold couplings if present
        if self.max_excitation == 2 and hasattr(self, "_index_to_pair"):
            # For pairs that differ by one index, couple them with the J of the differing sites
            for idx_a, (i, k) in self._index_to_pair.items():
                for j in range(1, N + 1):
                    if j in (i, k):
                        continue
                    # Candidate paired state shares k: (j,k) if ordering j<k else (k,j)
                    p = (j, k) if j < k else (k, j)
                    idx_b = self._pair_to_index.get(p)
                    if idx_b is None:
                        continue
                    # Add each unordered pair (idx_a, idx_b) only once
                    if idx_a >= idx_b:
                        continue
                    Jij = float(J[i - 1, j - 1])
                    if Jij == 0.0:
                        continue
                    HJ += (
                        HBAR
                        * Jij
                        * (
                            self.basis[idx_a] * self.basis[idx_b].dag()
                            + self.basis[idx_b] * self.basis[idx_a].dag()
                        )
                    )
        """
        return HJ

    # -----------------------------
    # Coupling matrix helpers
    # -----------------------------
    @property
    def coupling_matrix(self) -> np.ndarray:
        """Return the NxN coupling matrix J (fs^-1) with cylindrical-geometry priority.

        Priority order:
        1) If an explicit/cached matrix exists, return a copy of it.
        2) If positions are defined (e.g. via set_cylindrical_geometry or set_positions) and
           no explicit J is cached, compute isotropic couplings J_ij ∝ μ_i μ_j / r^3 and cache.
        3) Else, if at_coupling_cm is provided, return uniform off-diagonal J.
        4) Else, return zeros.
        """
        # 1) Return cached/explicit J if present
        if getattr(self, "_J", None) is not None:
            return self._J.copy()

        # 2) Geometry-first: build J from positions if available and not already cached
        if (
            getattr(self, "_positions", None) is not None
            and getattr(self, "_J", None) is None
        ):
            J = self._compute_isotropic_J(power=3.0, use_dipole_product=True)
            # Cache and return
            self._J = J
            return J.copy()

        # 3) Uniform fallback
        N = self.n_atoms
        if self.at_coupling_cm is not None:
            J = np.full((N, N), float(self.at_coupling), dtype=float)
            np.fill_diagonal(J, 0.0)
            return J

        # 4) No coupling
        return np.zeros((N, N), dtype=float)

    def set_coupling_matrix(self, J: Union[np.ndarray, List[List[float]]]):
        """Set the NxN coupling matrix (in fs^-1) and reset caches."""
        arr = np.array(J, dtype=float)
        if arr.shape != (self.n_atoms, self.n_atoms):
            raise ValueError(
                f"Coupling matrix must be shape ({self.n_atoms}, {self.n_atoms}), got {arr.shape}"
            )
        # Enforce zero diagonal (no on-site hopping)
        np.fill_diagonal(arr, 0.0)
        self._J = arr
        self.reset_cache()

    def clear_coupling_matrix(self):
        """Clear any custom coupling matrix (reverting to uniform or zero)."""
        if hasattr(self, "_J"):
            delattr(self, "_J")
        self.reset_cache()

    # =============================
    # MULTI-ATOM HAMILTONIAN WITH DOUBLE-EXCITATION MANIFOLD (OPTIONAL)
    # =============================
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
        N = self.n_atoms
        if N == 1:
            self._basis = [self._atom_g, self._atom_e]
            self._pair_to_index = {}
            self._index_to_pair = {}
            return

        if N == 2 and self.max_excitation == 2:
            self._basis = [
                tensor(self._atom_g, self._atom_g),
                tensor(self._atom_e, self._atom_g),
                tensor(self._atom_g, self._atom_e),
                tensor(self._atom_e, self._atom_e),
            ]
            self._pair_to_index = {(1, 2): 3}
            self._index_to_pair = {3: (1, 2)}
            return

        # Generic N >= 2 cases
        dim = self.dimension
        if self.max_excitation == 1:
            self._basis = [basis(dim, i) for i in range(dim)]
            self._pair_to_index = {}
            self._index_to_pair = {}
            return

        # max_excitation == 2: full truncated (0,1,2) excitation space (hard-core bosons)
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

    def build_Hamilton(self) -> Qobj:
        """Return Hamiltonian including up to two excitations (hard-core boson model).

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
        omegas = np.array([self.at_freqs_fs(i) for i in range(N)], dtype=float)

        # Start with zero operator in full truncated space
        H = 0

        # Helper lambdas for kets
        def ket(idx: int):
            return self._basis[idx]

        # Singles diagonal
        for i in range(1, N + 1):
            H += HBAR * omegas[i - 1] * ket(i) * ket(i).dag()
        J_fs = self.coupling_operator
        H += J_fs

        # If only max_excitation==1 we stop here
        if self.max_excitation == 1:
            return H

        # Double-excitation diagonal terms
        for idx, (i, j) in self._index_to_pair.items():  # idx >= 1+N
            H += HBAR * (omegas[i - 1] + omegas[j - 1]) * ket2dm(ket(idx))
        return H

    # =============================
    # GEOMETRY / COUPLING HELPERS (MINIMAL INITIAL SET)
    # =============================
    def set_positions(self, positions: Union[np.ndarray, List[List[float]]]):
        """Set Cartesian positions (shape (N,3)); does NOT build couplings automatically."""
        arr = np.array(positions, dtype=float)
        if arr.shape != (self.n_atoms, 3):
            raise ValueError(f"positions must have shape ({self.n_atoms}, 3)")
        self._positions = arr
        # Invalidate only if later couplings depend on geometry (user may rebuild)
        # (No cached Hamiltonian property currently, so just clear eigen-related caches)
        self.reset_cache()

    def build_isotropic_couplings(
        self, use_dipole_product: bool = True, power: float = 3.0
    ):
        """Construct isotropic J_ij ∝ (μ_i μ_j)/r^power (default 1/r^3).

        Parameters
        ----------
        use_dipole_product : bool
            If True multiply by μ_i μ_j; else omit dipole magnitudes.
        power : float
            Exponent of distance in denominator (default 3).
        """
        if self._positions is None:
            raise RuntimeError(
                "Positions not set. Call set_positions or set_cylindrical_geometry first."
            )
        J = self._compute_isotropic_J(
            power=power, use_dipole_product=use_dipole_product
        )
        self.set_coupling_matrix(J)

    def _compute_isotropic_J(
        self, power: float = 3.0, use_dipole_product: bool = True
    ) -> np.ndarray:
        """Compute isotropic geometry-based J using current positions and dipoles.

        J_ij = at_coupling / r_ij^power * (μ_i μ_j if use_dipole_product else 1)

        Returns a new NxN numpy array (fs^-1). Raises if positions are unset or invalid.
        """
        if self._positions is None:
            raise RuntimeError("Positions not set; cannot compute isotropic couplings.")
        N = self.n_atoms
        pos = self._positions
        J = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                r_vec = pos[j] - pos[i]
                r = float(np.linalg.norm(r_vec))
                if r == 0:
                    raise ValueError("Duplicate positions encountered (zero distance).")
                base = float(self.at_coupling) / (r**power)
                if use_dipole_product:
                    base *= float(self.dip_moments[i]) * float(self.dip_moments[j])
                J[i, j] = base
                J[j, i] = base
        np.fill_diagonal(J, 0.0)
        return J

    def set_cylindrical_geometry(
        self,
        distance: float,
        build_couplings: bool = True,
        use_dipole_product: bool = True,
    ):
        """Construct cylindrical geometry using mandatory n_rings (for n_atoms>2).

        Assumes: n_atoms = n_chains * n_rings with n_rings provided at init.
        distance: axial spacing and chord length between adjacent chains.
        """
        if self.n_atoms <= 2:
            raise ValueError("Cylindrical geometry only meaningful for n_atoms > 2")
        if self.n_rings is None or self._n_chains is None:
            # Should not happen: __post_init__ sets defaults; safeguard fallback to linear chain
            self.n_rings = self.n_atoms
            self._n_chains = 1
        n_rings = self.n_rings
        n_chains = self._n_chains

        # Ring centers in xy-plane
        if n_chains == 1:
            ring_centers = np.array([[0.0, 0.0, 0.0]])
        else:
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
        self.set_positions(positions)

        if not build_couplings:
            return
        self.build_isotropic_couplings(
            use_dipole_product=use_dipole_product,
            power=3.0,
        )

    @property
    def n_chains(self) -> Optional[int]:
        """Derived number of chains if `n_rings` specified (only meaningful for n_atoms>2)."""
        return self._n_chains

    @property
    def positions(self) -> Optional[np.ndarray]:  # type: ignore
        return getattr(self, "_positions", None)

    @property
    def H0_N_canonical(self) -> Qobj:
        return self.build_Hamilton()

    def excitation_number_from_index(self, idx: int) -> int:
        if idx == 0:
            return 0
        elif 1 <= idx <= self.n_atoms:
            return 1
        else:
            return 2

    @cached_property
    def eigenstates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues & eigenstates (cached).

        Invalidated when `update_freqs_cm` is called.
        """
        return self.H0_N_canonical.eigenstates()

    @cached_property
    def sm_op(self) -> Qobj:
        sm = 0
        if self.max_excitation == 1:
            # Single-excitation lowering operator: sum_i μ_i |0><i|
            for i, mu in enumerate(self.dip_moments, start=1):
                sm += mu * (self.basis[0] * self.basis[i].dag())
            return sm
        # max_excitation == 2: add terms connecting double -> single manifolds
        # |0><i|
        for i, mu in enumerate(self.dip_moments, start=1):
            sm += mu * (self.basis[0] * self.basis[i].dag())
        # |j><i,j| and |j><j,i| but only one ordering stored (i<j)
        for (i, j), idx in self._pair_to_index.items():  # i<j
            mu_i = self.dip_moments[i - 1]
            mu_j = self.dip_moments[j - 1]
            # Annihilating excitation on site i from |i,j> leaves |j>, and vice versa
            sm += mu_i * (self.basis[j] * self.basis[idx].dag())
            sm += mu_j * (self.basis[i] * self.basis[idx].dag())
        return sm

    @cached_property
    def dip_op(self) -> Qobj:
        return self.sm_op + self.sm_op.dag()

    def deph_op_i(self, i: int) -> Qobj:
        """Return site i population operator in the site basis (|i><i|)."""
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
        elif self.n_atoms > 2 and self.n_rings is not None:
            lines.append(
                f"    {'n_rings':<20}: {self.n_rings} (derived n_chains = {self.n_chains})"
            )
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
