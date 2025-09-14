# DEFINE THE SYSTEM PARAMETERS CLASS

import numpy as np
import json
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from functools import cached_property
from qutip import basis, ket2dm, Qobj
from qspectro2d.constants import HBAR, convert_cm_to_fs, convert_fs_to_cm


def pair_to_index(i: int, j: int, n: int) -> int:
    """0-based canonical index for |i,j> with i<j, given n atoms.
    ground=0, singles=1..n, doubles=(n+1)..
    Returns the index+1 of the basis corresponding to the double excitation |i,j>.
    """
    assert 1 <= i < j <= n
    return 1 + n + math.comb(j - 1, 2) + i


def index_to_pair(k: int, n: int) -> Tuple[int, int]:
    """Inverse map: from index k (in double block == corresponds to state basis[k-1]) to (atoms i,j excited)."""
    pair_rank = k - (1 + n)  # rank inside double block (0-based)

    # find j with C(j-1,2) < r <= C(j,2)
    j = 2
    while math.comb(j, 2) < pair_rank:
        j += 1
    i = pair_rank - math.comb(j - 1, 2)
    return i, j


@dataclass
class AtomicSystem:
    # VITAL ATTRIBUTES
    n_atoms: int = 1
    n_chains: int = 1
    frequencies_cm: List[float] = field(default_factory=lambda: [16000.0])  # in cm^-1
    dip_moments: List[float] = field(default_factory=lambda: [1.0])
    coupling_cm: float = 0.0
    delta_cm: float = 0.0  # inhomogeneous broadening

    # 1 = existing behaviour (ground + single exc manifold); 2 adds double manifold
    max_excitation: int = 1

    psi_ini: Optional[Qobj] = None  # initial state, default is ground state

    def __post_init__(self):
        # Build basis
        self._build_basis()

        # spectroscopy always starts in the ground state
        self.psi_ini = ket2dm(self._basis[0])

        # store the initial frequencies in history
        self._frequencies_cm_history = [self.frequencies_cm.copy()]

        # Internal fs^-1 storage (single source of truth for dynamics)
        self._frequencies_fs = np.asarray(convert_cm_to_fs(self.frequencies_cm), dtype=float)
        self._coupling_fs = convert_cm_to_fs(self.coupling_cm)
        self._delta_fs = convert_cm_to_fs(self.delta_cm)

        # Always set cylindrical positions and compute isotropic couplings
        self.n_rings = self.n_atoms // self.n_chains
        self._setup_geometry_and_couplings()

    def update_frequencies_cm(self, new_freqs: List[float]):
        if len(new_freqs) != self.n_atoms:
            raise ValueError(f"Expected {self.n_atoms} frequencies, got {len(new_freqs)}")

        # Save current freqs before updating
        self._frequencies_cm_history.append(new_freqs.copy())
        self.frequencies_cm = new_freqs.copy()
        # keep fs cache in sync
        self._frequencies_fs = np.asarray(convert_cm_to_fs(self.frequencies_cm), dtype=float)
        # cached spectrum/operators depend on frequencies
        self.reset_cache()

    def update_coupling_cm(self, new_coupling_cm: float) -> None:
        """Update base coupling (cm^-1) and refresh internal fs cache/coupling matrix."""
        self.coupling_cm = new_coupling_cm
        self._coupling_fs = float(convert_cm_to_fs(self.coupling_cm))
        # Coupling affects J matrix and H; recompute couplings and reset caches
        self._compute_isotropic_couplings()
        self.reset_cache()

    def update_delta_cm(self, new_delta_cm: float) -> None:
        """Update inhomogeneous broadening (cm^-1)."""
        self.delta_cm = new_delta_cm
        self._delta_fs = float(convert_cm_to_fs(self.delta_cm))
        # Not strictly needed for operators, but keep consistency
        self.reset_cache()

    def reset_cache(self) -> None:
        """Invalidate cached spectral quantities affected by parameter changes."""
        # delete cached_properties to force recompute on parameter changes
        for key in ("eigenstates", "eigenbasis_transform"):
            if key in self.__dict__:
                del self.__dict__[key]

    # === CORE PARAMETERS ===
    @cached_property
    def dimension(self):
        """Dimension of the Hilbert space (ground + single + double excitations)."""
        N = self.n_atoms
        exc = self.max_excitation
        if exc == 1:
            dim = 1 + N
        elif exc == 2:
            n_pairs = math.comb(N, 2)
            dim = 1 + N + n_pairs
        return dim

    # === QUANTUM OPERATORS ===
    def _build_basis(self):
        """Construct basis depending on n_atoms and max_excitation.
        - Ground state index: 0
        - Single excitations: |i> mapped to index i (1..N) when max_excitation>=1
        - Double excitations: |i,j> (i<j) mapped to indices N + p where
              p runs from 1..N_pairs in lexicographic order over (i,j).
        """
        dim = self.dimension
        self._basis = [basis(dim, i) for i in range(dim)]

    @property
    def hamiltonian(self) -> Qobj:
        """Return Hamiltonian including up to two excitations (hard-core boson model / fock-basis).

        H = Σ_i ħ ω_i a_i^† a_i +
            Σ_{i<j} ħ (J_ij a_i^† a_j + h.c.)
         , truncated to 1,2 excitations.

            Matrix elements:
          - Diagonal singles: <i|H|i> = ħ ω_i
          - Single-single off-diagonals: ħ J_ij (i≠j)
          - Diagonal doubles: <ij|H|ij> = ħ (ω_i + ω_j)
          - double-off-diagonal: <ik|H|ij> = J_jk
            if states share exactly one site, e.g. |i,j> ↔ |i,k>, element = ħ J_jk
        """
        N = self.n_atoms
        omegas = self._frequencies_fs

        # Helper lambdas for kets
        def ket(idx: int):
            return self._basis[idx]

        H = 0
        # Singles diagonal
        for i in range(1, N + 1):
            H += HBAR * omegas[i - 1] * ket(i) * ket(i).dag()
        # Inter-site couplings (single manifold)
        H += self.coupling_op

        if self.max_excitation == 1:
            return H

        # Double-excitation diagonal terms
        for i in range(1, N):
            for j in range(i + 1, N + 1):
                idx = pair_to_index(i, j, N)
                H += HBAR * (omegas[i - 1] + omegas[j - 1]) * ket2dm(ket(idx - 1))
        return H

    @cached_property
    def eigenstates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues & eigenstates (cached).

        Invalidated when `update_frequencies_cm` is called.
        """
        return self.hamiltonian.eigenstates()

    @cached_property
    def eigenbasis_transform(self) -> Qobj:
        """Unitary matrix U whose columns are eigenstates (site → eigen basis)."""
        _, basis_states = self.eigenstates
        u = Qobj(
            np.column_stack([e.full() for e in basis_states]),
            dims=self.hamiltonian.dims,
        )
        return u

    def to_eigenbasis(self, operator: Qobj) -> Qobj:
        """Transform an operator into the eigenbasis of the Hamiltonian using cached U."""
        return self.eigenbasis_transform.dag() * operator * self.eigenbasis_transform

    @property
    def lowering_op(self) -> Qobj:
        """return the lowering operator in the canonical basis"""
        lowering_op = 0
        # Single-excitation lowering operator: sum_i μ_i |0><i|
        for i in range(1, self.n_atoms + 1):
            mu_i = self.dip_moments[i - 1]
            lowering_op += mu_i * (self._basis[0] * self._basis[i].dag())

        # max_excitation == 2: add terms connecting double -> single manifolds
        # |j><i,j| and |j><j,i| but only one ordering stored (i<j)
        if self.max_excitation == 2:
            N = self.n_atoms
            for i in range(1, N):
                for j in range(i + 1, N + 1):
                    idx = pair_to_index(i, j, N)
                    mu_i = self.dip_moments[i - 1]
                    mu_j = self.dip_moments[j - 1]
                    # Annihilating excitation on site i from |i,j> leaves |j>, and vice versa
                    lowering_op += mu_i * (self._basis[j] * self._basis[idx - 1].dag())
                    lowering_op += mu_j * (self._basis[i] * self._basis[idx - 1].dag())

        return lowering_op

    @property
    def dipole_op(self) -> Qobj:
        """return the dipole operator in the canonical basis"""
        dip_op = self.lowering_op + self.lowering_op.dag()
        return dip_op

    @property
    def number_op(self) -> Qobj:
        """
        Total excitation number operator (in the canonical basis).

        Definition in the canonical/site basis:
            N = sum_i |i><i| + 2 * sum_{i<j} |i,j><i,j|

        - For max_excitation == 1, only the single-excitation projectors are included.
        """
        N_op = 0
        dim = self.dimension

        for i in range(dim):
            N_op += self.excitation_number_from_index(i) * ket2dm(self._basis[i])

        return N_op

    @cached_property
    def coupling_op(self) -> Qobj:
        """Inter-site coupling operator (cached) using isotropic dipole couplings ~1/r^3.

        Uses the coupling matrix computed from cylindrical geometry.
        Single-manifold couplings: |i><j| + |j><i| with strength ħ J_ij.
        NOTE: If max_excitation == 2, also couples double states that share one site.
        """
        N = self.n_atoms
        if N <= 1:
            return 0

        # Use the isotropic coupling matrix (fs^-1)
        J_matrix = self._coupling_matrix_fs
        HJ = 0

        # Singles excitation couplings
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                Jij = J_matrix[i - 1, j - 1]
                if np.isclose(Jij, 0.0):
                    continue
                ket_i = self._basis[i]
                ket_j = self._basis[j]
                HJ += HBAR * Jij * (ket_i * ket_j.dag() + ket_j * ket_i.dag())

        # double-excitation couplings
        if self.max_excitation == 2 and N > 2:
            visited_pairs = set()  # deduplicate symmetric connections

            # Iterate over all double states |i,j>, i<j
            for i in range(1, N):
                for j in range(i + 1, N + 1):
                    idx_ij = pair_to_index(i, j, N) - 1  # basis index of |i,j>

                    # Move one excitation along coupling graph (hard-core constraint)
                    for k in range(1, N + 1):
                        if k == i or k == j:
                            continue

                        # |i,j> <-> |i,k> with amplitude J_{j,k}
                        Jjk = J_matrix[j - 1, k - 1]
                        if Jjk != 0.0:
                            a, b = (min(i, k), max(i, k))
                            idx_ik = pair_to_index(a, b, N) - 1
                            key = (min(idx_ij, idx_ik), max(idx_ij, idx_ik))
                            if key not in visited_pairs:
                                ket_a = self._basis[idx_ij]
                                ket_b = self._basis[idx_ik]
                                HJ += HBAR * Jjk * (ket_a * ket_b.dag() + ket_b * ket_a.dag())
                                visited_pairs.add(key)

                        # |i,j> <-> |k,j> with amplitude J_{i,k}
                        Jik = J_matrix[i - 1, k - 1]
                        if Jik != 0.0:
                            a2, b2 = (min(k, j), max(k, j))
                            idx_kj = pair_to_index(a2, b2, N) - 1
                            key2 = (min(idx_ij, idx_kj), max(idx_ij, idx_kj))
                            if key2 not in visited_pairs:
                                ket_a = self._basis[idx_ij]
                                ket_b = self._basis[idx_kj]
                                HJ += HBAR * Jik * (ket_a * ket_b.dag() + ket_b * ket_a.dag())
                                visited_pairs.add(key2)
        return HJ

    # === GEOMETRY AND POSITIONS ===
    @property
    def coupling_matrix_cm(self) -> np.ndarray:
        """Return coupling matrix in cm^-1 for display."""
        return convert_fs_to_cm(self._coupling_matrix_fs)

    def _setup_geometry_and_couplings(self):
        """Set up cylindrical geometry and compute isotropic couplings."""
        self._set_cylindrical_positions()
        self._compute_isotropic_couplings()

    def _set_cylindrical_positions(self, distance: float = 1.0):  # TODO this could be parameterized
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

    def _compute_isotropic_couplings(
        self, power: float = 3.0
    ) -> np.ndarray:  # TODO extend to vectorized dipoles
        """Compute isotropic J_ij = coupling * μ_i * μ_j / r^power."""
        N = self.n_atoms
        pos = self._positions
        J = np.zeros((N, N), dtype=float)

        base_coupling_fs = self._coupling_fs

        for i in range(N):
            for j in range(i + 1, N):
                r_vec = pos[j] - pos[i]
                r = float(np.linalg.norm(r_vec))
                if r == 0:
                    raise ValueError("Duplicate positions encountered (zero distance).")

                # Isotropic coupling with dipole product
                coupling_ij = (
                    base_coupling_fs * self.dip_moments[i] * self.dip_moments[j] / (r**power)
                )
                J[i, j] = coupling_ij
                J[j, i] = coupling_ij

        self._coupling_matrix_fs = J

    def excitation_number_from_index(self, idx: int) -> int:
        if idx == 0:
            return 0
        elif 1 <= idx <= self.n_atoms:
            return 1
        else:
            return 2

    def deph_op_i(self, i: int) -> Qobj:
        """Return site i population operator in the eigenbasis (|i><i|).
        Also works for double states |i,j> -> returns |i,j><i,j|."""
        if i == 0:
            raise ValueError("indexing ground state -> use i elem 1,...,N+1")
        op = ket2dm(self._basis[i])
        return op

    def omega_ij(self, i: int, j: int) -> float:
        """Return energy difference (frequency) between eigenstates i and j in fs^-1."""
        return self.eigenstates[0][i] - self.eigenstates[0][j]

    def omega_ij_cm(self, i: int, j: int) -> float:
        """Return energy difference (frequency) between eigenstates i and j in cm^-1."""
        return float(convert_fs_to_cm(self.omega_ij(i, j)))

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
            lines.append(f"    {'coupling':<20}: {self.coupling_cm} cm^-1")
            lines.append(f"    {'delta':<20}: {self.delta_cm} cm^-1")
        elif self.n_atoms > 2 and self.n_rings is not None:
            lines.append(f"    {'n_rings':<20}: {self.n_rings} (n_chains = {self.n_chains})")
            lines.append(f"    {'positions shape':<20}: {self._positions.shape}")
            lines.append(f"    {'coupling matrix (cm^-1)':<20}:")
            lines.append(str(self.coupling_matrix_cm))
        lines.append(f"\n    {'psi_ini':<20}:<")
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
            "delta_cm": self.delta_cm,
            "coupling_cm": self.coupling_cm,
        }
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
