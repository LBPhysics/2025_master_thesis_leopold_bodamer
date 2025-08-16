# =============================
# REFACTORED ATOMIC SYSTEM - SIMPLIFIED AND MODULAR
# =============================
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
from abc import ABC, abstractmethod
from qutip import basis, ket2dm, tensor, Qobj
from qspectro2d.constants import HBAR, convert_cm_to_fs


# =============================
# CORE SYSTEM DEFINITION (SINGLE RESPONSIBILITY)
# =============================
@dataclass
class AtomicSystem:
    """Core atomic system definition - only frequencies and dipoles.

    This class handles ONLY the basic system parameters. All complex
    functionality (Hamiltonians, geometry, operators) is delegated
    to separate classes.
    """

    n_atoms: int
    frequencies: np.ndarray  # Always in fs^-1 internally
    dipole_moments: np.ndarray
    max_excitation: int = 1

    def __post_init__(self):
        # Convert to numpy arrays and validate
        self.frequencies = np.atleast_1d(self.frequencies).astype(float)
        self.dipole_moments = np.atleast_1d(self.dipole_moments).astype(float)

        # Simple validation only
        if len(self.frequencies) != self.n_atoms:
            raise ValueError(
                f"Expected {self.n_atoms} frequencies, got {len(self.frequencies)}"
            )
        if len(self.dipole_moments) != self.n_atoms:
            raise ValueError(
                f"Expected {self.n_atoms} dipole moments, got {len(self.dipole_moments)}"
            )
        if self.max_excitation not in (1, 2):
            raise ValueError("max_excitation must be 1 or 2")

    @classmethod
    def from_cm(
        cls, n_atoms: int, freqs_cm: List[float], dipole_moments: List[float], **kwargs
    ):
        """Create system with frequencies in cm^-1 (converted internally to fs^-1)."""
        freqs_fs = np.array([convert_cm_to_fs(f) for f in freqs_cm])
        return cls(
            n_atoms=n_atoms,
            frequencies=freqs_fs,
            dipole_moments=dipole_moments,
            **kwargs,
        )

    @property
    def frequencies_cm(self) -> np.ndarray:
        """Return frequencies in cm^-1 for display/output."""
        return np.array([freq / convert_cm_to_fs(1.0) for freq in self.frequencies])

    def to_dict(self) -> dict:
        """Serialize basic parameters only."""
        return {
            "n_atoms": self.n_atoms,
            "frequencies_cm": self.frequencies_cm.tolist(),
            "dipole_moments": self.dipole_moments.tolist(),
            "max_excitation": self.max_excitation,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Deserialize from dictionary."""
        return cls.from_cm(
            n_atoms=data["n_atoms"],
            freqs_cm=data["frequencies_cm"],
            dipole_moments=data["dipole_moments"],
            max_excitation=data.get("max_excitation", 1),
        )


# =============================
# COUPLING MATRIX MANAGEMENT
# =============================
class CouplingMatrix:
    """Manages inter-site coupling strengths J_ij."""

    def __init__(self, n_atoms: int, coupling_fs: Optional[np.ndarray] = None):
        self.n_atoms = n_atoms
        if coupling_fs is None:
            self._matrix = np.zeros((n_atoms, n_atoms))
        else:
            self.set_matrix(coupling_fs)

    def set_matrix(self, matrix: np.ndarray):
        """Set coupling matrix (fs^-1) with validation."""
        matrix = np.array(matrix, dtype=float)
        if matrix.shape != (self.n_atoms, self.n_atoms):
            raise ValueError(
                f"Expected shape ({self.n_atoms}, {self.n_atoms}), got {matrix.shape}"
            )
        # Enforce zero diagonal
        np.fill_diagonal(matrix, 0.0)
        self._matrix = matrix.copy()

    def set_uniform(self, coupling_strength: float):
        """Set uniform off-diagonal coupling."""
        self._matrix = np.full((self.n_atoms, self.n_atoms), coupling_strength)
        np.fill_diagonal(self._matrix, 0.0)

    @classmethod
    def from_cm(cls, n_atoms: int, coupling_cm: float):
        """Create with uniform coupling in cm^-1."""
        instance = cls(n_atoms)
        instance.set_uniform(convert_cm_to_fs(coupling_cm))
        return instance

    @property
    def matrix(self) -> np.ndarray:
        """Get coupling matrix (fs^-1)."""
        return self._matrix.copy()

    @property
    def matrix_cm(self) -> np.ndarray:
        """Get coupling matrix in cm^-1."""
        return self._matrix / convert_cm_to_fs(1.0)


# =============================
# GEOMETRY MANAGEMENT
# =============================
class SystemGeometry(ABC):
    """Abstract base for system geometries."""

    @abstractmethod
    def get_positions(self) -> np.ndarray:
        """Return positions as (n_atoms, 3) array."""
        pass

    @abstractmethod
    def compute_coupling_matrix(
        self, base_strength: float, dipole_moments: np.ndarray
    ) -> np.ndarray:
        """Compute distance-based coupling matrix."""
        pass


class LinearGeometry(SystemGeometry):
    """Linear chain geometry."""

    def __init__(self, n_atoms: int, spacing: float = 1.0):
        self.n_atoms = n_atoms
        self.spacing = spacing

    def get_positions(self) -> np.ndarray:
        """Linear chain along z-axis."""
        return np.array([[0.0, 0.0, i * self.spacing] for i in range(self.n_atoms)])

    def compute_coupling_matrix(
        self, base_strength: float, dipole_moments: np.ndarray
    ) -> np.ndarray:
        """1/r^3 coupling for linear chain."""
        positions = self.get_positions()
        matrix = np.zeros((self.n_atoms, self.n_atoms))

        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                r = np.linalg.norm(positions[j] - positions[i])
                coupling = (
                    base_strength * dipole_moments[i] * dipole_moments[j] / (r**3)
                )
                matrix[i, j] = matrix[j, i] = coupling

        return matrix


class CylindricalGeometry(SystemGeometry):
    """Cylindrical geometry (simplified from original)."""

    def __init__(self, n_atoms: int, n_rings: int, axial_spacing: float = 1.0):
        if n_atoms % n_rings != 0:
            raise ValueError(f"n_rings={n_rings} must divide n_atoms={n_atoms}")

        self.n_atoms = n_atoms
        self.n_rings = n_rings
        self.n_chains = n_atoms // n_rings
        self.axial_spacing = axial_spacing

    def get_positions(self) -> np.ndarray:
        """Generate cylindrical positions."""
        if self.n_chains == 1:
            # Linear chain
            return np.array(
                [[0.0, 0.0, z * self.axial_spacing] for z in range(self.n_rings)]
            )

        # Multiple chains in cylinder
        radius = self.axial_spacing / (2.0 * np.sin(np.pi / self.n_chains))
        positions = []

        for chain in range(self.n_chains):
            phi = 2.0 * np.pi * chain / self.n_chains
            x_center = radius * np.cos(phi)
            y_center = radius * np.sin(phi)

            for ring in range(self.n_rings):
                z = ring * self.axial_spacing
                positions.append([x_center, y_center, z])

        return np.array(positions)

    def compute_coupling_matrix(
        self, base_strength: float, dipole_moments: np.ndarray
    ) -> np.ndarray:
        """1/r^3 coupling for cylindrical geometry."""
        positions = self.get_positions()
        matrix = np.zeros((self.n_atoms, self.n_atoms))

        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                r = np.linalg.norm(positions[j] - positions[i])
                if r > 0:  # Avoid division by zero
                    coupling = (
                        base_strength * dipole_moments[i] * dipole_moments[j] / (r**3)
                    )
                    matrix[i, j] = matrix[j, i] = coupling

        return matrix


# =============================
# BASIS CONSTRUCTION (UNIFIED, NO SPECIAL CASES)
# =============================
class TruncatedBasis:
    """Constructs truncated Hilbert space basis for hard-core bosons."""

    def __init__(self, n_atoms: int, max_excitation: int):
        self.n_atoms = n_atoms
        self.max_excitation = max_excitation
        self._build_basis()

    def _build_basis(self):
        """Build basis states and index mappings.

        Unified indexing (no special cases):
        - Index 0: ground state |0⟩
        - Indices 1..N: single excitations |i⟩
        - Indices N+1...: double excitations |i,j⟩ with i<j
        """
        if self.n_atoms == 1:
            # Single atom: just |g⟩, |e⟩
            self.basis_states = [basis(2, 0), basis(2, 1)]
            self.pair_to_index = {}
            self.index_to_pair = {}
            return

        # Multi-atom case: use computational basis
        dim = self._compute_dimension()
        self.basis_states = [basis(dim, i) for i in range(dim)]

        # Build mappings for double excitations
        self.pair_to_index = {}
        self.index_to_pair = {}

        if self.max_excitation == 2:
            idx = 1 + self.n_atoms  # Start after single excitations
            for i in range(1, self.n_atoms):
                for j in range(i + 1, self.n_atoms + 1):
                    self.pair_to_index[(i, j)] = idx
                    self.index_to_pair[idx] = (i, j)
                    idx += 1

    def _compute_dimension(self) -> int:
        """Compute Hilbert space dimension."""
        if self.max_excitation == 1:
            return 1 + self.n_atoms
        elif self.max_excitation == 2:
            n_pairs = self.n_atoms * (self.n_atoms - 1) // 2
            return 1 + self.n_atoms + n_pairs
        else:
            raise ValueError("max_excitation must be 1 or 2")

    @property
    def dimension(self) -> int:
        return len(self.basis_states)


# =============================
# HAMILTONIAN AND OPERATOR CONSTRUCTION
# =============================
class SystemBuilder:
    """Builds quantum operators from system definition and coupling."""

    def __init__(self, system: AtomicSystem, coupling: Optional[CouplingMatrix] = None):
        self.system = system
        self.coupling = coupling or CouplingMatrix(system.n_atoms)
        self.basis = TruncatedBasis(system.n_atoms, system.max_excitation)

        # Cache for expensive computations
        self._hamiltonian = None
        self._dipole_operator = None
        self._eigenstates = None

    def build_hamiltonian(self) -> Qobj:
        """Construct system Hamiltonian."""
        if self._hamiltonian is not None:
            return self._hamiltonian

        N = self.system.n_atoms
        omegas = self.system.frequencies
        J_matrix = self.coupling.matrix

        H = 0  # Start with zero operator

        # Single excitation diagonal terms: ħωᵢ|i⟩⟨i|
        for i in range(1, N + 1):
            H += HBAR * omegas[i - 1] * ket2dm(self.basis.basis_states[i])

        # Single excitation off-diagonal couplings: ħJᵢⱼ(|i⟩⟨j| + |j⟩⟨i|)
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                J_ij = J_matrix[i - 1, j - 1]
                if J_ij != 0:
                    ket_i = self.basis.basis_states[i]
                    ket_j = self.basis.basis_states[j]
                    H += HBAR * J_ij * (ket_i * ket_j.dag() + ket_j * ket_i.dag())

        # Double excitation terms (if max_excitation == 2)
        if self.system.max_excitation == 2:
            # Diagonal: ħ(ωᵢ + ωⱼ)|i,j⟩⟨i,j|
            for idx, (i, j) in self.basis.index_to_pair.items():
                energy = omegas[i - 1] + omegas[j - 1]
                H += HBAR * energy * ket2dm(self.basis.basis_states[idx])

            # Double-double couplings: |i,j⟩ ↔ |i,k⟩ with strength ħJⱼₖ
            for (i1, j1), idx1 in self.basis.pair_to_index.items():
                for (i2, j2), idx2 in self.basis.pair_to_index.items():
                    if idx1 >= idx2:  # Avoid double counting
                        continue

                    # Check if states share exactly one site
                    shared_sites = set([i1, j1]) & set([i2, j2])
                    if len(shared_sites) == 1:
                        # Find the coupling between non-shared sites
                        sites1 = set([i1, j1]) - shared_sites
                        sites2 = set([i2, j2]) - shared_sites
                        site1 = sites1.pop()
                        site2 = sites2.pop()

                        J_coupling = J_matrix[site1 - 1, site2 - 1]
                        if J_coupling != 0:
                            ket1 = self.basis.basis_states[idx1]
                            ket2 = self.basis.basis_states[idx2]
                            H += (
                                HBAR
                                * J_coupling
                                * (ket1 * ket2.dag() + ket2 * ket1.dag())
                            )

        self._hamiltonian = H
        return H

    def build_dipole_operator(self) -> Qobj:
        """Construct dipole operator μ = σ + σ†."""
        if self._dipole_operator is not None:
            return self._dipole_operator

        lowering_op = self.build_lowering_operator()
        self._dipole_operator = lowering_op + lowering_op.dag()
        return self._dipole_operator

    def build_lowering_operator(self) -> Qobj:
        """Construct lowering operator σ = Σᵢ μᵢ σᵢ⁻."""
        N = self.system.n_atoms
        dipoles = self.system.dipole_moments

        sigma = 0

        # Single excitations: μᵢ|0⟩⟨i|
        for i in range(1, N + 1):
            mu_i = dipoles[i - 1]
            ket_0 = self.basis.basis_states[0]
            ket_i = self.basis.basis_states[i]
            sigma += mu_i * (ket_0 * ket_i.dag())

        # Double excitation terms (if max_excitation == 2)
        if self.system.max_excitation == 2:
            for (i, j), idx in self.basis.pair_to_index.items():
                ket_ij = self.basis.basis_states[idx]
                ket_i = self.basis.basis_states[i]
                ket_j = self.basis.basis_states[j]

                # μᵢ|j⟩⟨i,j| + μⱼ|i⟩⟨i,j|
                sigma += dipoles[i - 1] * (ket_j * ket_ij.dag())
                sigma += dipoles[j - 1] * (ket_i * ket_ij.dag())

        return sigma

    def get_eigenstates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get eigenvalues and eigenvectors of Hamiltonian."""
        if self._eigenstates is None:
            H = self.build_hamiltonian()
            self._eigenstates = H.eigenstates()
        return self._eigenstates

    def invalidate_cache(self):
        """Clear cached computations."""
        self._hamiltonian = None
        self._dipole_operator = None
        self._eigenstates = None


# =============================
# CONVENIENCE FACTORY FUNCTIONS
# =============================
def create_single_atom(freq_cm: float, dipole: float = 1.0) -> SystemBuilder:
    """Create single atom system."""
    system = AtomicSystem.from_cm(1, [freq_cm], [dipole])
    return SystemBuilder(system)


def create_dimer(
    freq1_cm: float,
    freq2_cm: float,
    coupling_cm: float,
    dipole1: float = 1.0,
    dipole2: float = 1.0,
    max_excitation: int = 1,
) -> SystemBuilder:
    """Create two-atom system with uniform coupling."""
    system = AtomicSystem.from_cm(
        2, [freq1_cm, freq2_cm], [dipole1, dipole2], max_excitation=max_excitation
    )
    coupling = CouplingMatrix.from_cm(2, coupling_cm)
    return SystemBuilder(system, coupling)


def create_linear_chain(
    n_atoms: int,
    freq_cm: float,
    coupling_cm: float,
    spacing: float = 1.0,
    dipole: float = 1.0,
    max_excitation: int = 1,
) -> SystemBuilder:
    """Create linear chain with geometric coupling."""
    system = AtomicSystem.from_cm(
        n_atoms, [freq_cm] * n_atoms, [dipole] * n_atoms, max_excitation=max_excitation
    )

    # Use geometry to compute couplings
    geometry = LinearGeometry(n_atoms, spacing)
    coupling_matrix = geometry.compute_coupling_matrix(
        convert_cm_to_fs(coupling_cm), system.dipole_moments
    )
    coupling = CouplingMatrix(n_atoms, coupling_matrix)

    return SystemBuilder(system, coupling)


# =============================
# EXAMPLE USAGE
# =============================
if __name__ == "__main__":
    print("=== Refactored AtomicSystem Demo ===\n")

    # Single atom
    print("1. Single atom:")
    single = create_single_atom(freq_cm=16000.0, dipole=1.0)
    print(f"   Hamiltonian:\n{single.build_hamiltonian()}\n")

    # Dimer
    print("2. Dimer:")
    dimer = create_dimer(freq1_cm=16000, freq2_cm=15640, coupling_cm=50.0)
    H_dimer = dimer.build_hamiltonian()
    evals, evecs = dimer.get_eigenstates()
    print(f"   Eigenvalues: {evals}\n")

    # Linear chain
    print("3. Linear chain (3 atoms):")
    chain = create_linear_chain(
        n_atoms=3, freq_cm=16000, coupling_cm=100.0, spacing=1.0
    )
    print(f"   Coupling matrix:\n{chain.coupling.matrix_cm}\n")

    # Serialization
    print("4. Serialization:")
    system_dict = dimer.system.to_dict()
    print(f"   Serialized: {system_dict}")

    restored_system = AtomicSystem.from_dict(system_dict)
    print(
        f"   Frequencies match: {np.allclose(dimer.system.frequencies, restored_system.frequencies)}"
    )
