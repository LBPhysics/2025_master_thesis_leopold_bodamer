from dataclasses import dataclass  # for the class definition
import numpy as np

# Use TYPE_CHECKING to avoid circular imports during runtime
from qspectro2d.core.atomic_system.system_class import AtomicSystem

from qutip import ket2dm, BosonicEnvironment

from qspectro2d.core.bath_system.bath_fcts import bath_to_rates


@dataclass
class SystemBathCoupling:
    system: AtomicSystem
    bath: BosonicEnvironment

    # TODO now the decay channels DO NOT match the paper cases
    @property
    def br_decay_channels(self):
        """Generate the a_ops list for Bloch-Redfield solver.
        Includes:
          - Pure dephasing projectors for all single-excitation states
          - (If max_excitation == 2) dephasing for each double-excitation state
          - Radiative decay channels (lowering operators) from singles to ground
          - (If max_excitation == 2) lowering from double states to singles

        NOTE: BR formalism usually builds rates from bath correlation functions; here we
        provide operator structures only. Coupling strengths are encoded in the bath object.
        """
        sys = self.system
        br_ops: list[list] = []
        n_atoms = sys.n_atoms
        max_exc = sys.max_excitation

        # Helper: add dephasing projector for basis index k
        def add_projector(k: int):
            """Return site k population operator in the site basis (|k><k|)."""
            if k == 0:
                raise ValueError("indexing ground state -> use k elem 1,...,N+1")

            br_ops.append([sys.deph_op_i(k), self.bath])

        # Multi-atom:
        for i_atom in range(1, n_atoms + 1):
            # singles manifold projectors
            add_projector(i_atom)

            # Radiative-like decay (lowering + raising) operators from singles to ground: |0><i| + |i><0|
            Li = sys.basis[0] * sys.basis[i_atom].dag()
            br_ops.append([Li + Li.dag(), self.bath])

        # Double manifold projectors only if available
        index_to_pair = sys._index_to_pair
        for idx, (i, j) in index_to_pair.items():  # i<j atom labels
            add_projector(idx)

            """
            # Double -> single lowering (if double manifold present)
            # |i> corresponds to basis index i, etc.
            Li = sys.basis[i] * sys.basis[idx].dag()
            Lj = sys.basis[j] * sys.basis[idx].dag()
            br_ops.append([Li + Li.dag(), self.bath])
            br_ops.append([Lj + Lj.dag(), self.bath])
            """
        return br_ops

    @property
    def me_decay_channels(self):
        """Generate c_ops for Lindblad solver respecting max_excitation.

        Strategy:
          - Pure dephasing: projectors on each populated basis state (singles; doubles if present)
          - Population relaxation/excitation: per-site lowering/raising (|0><i|, |i><0|) with rates from bath
            (double-manifold relaxation: |i,j><i| and |i,j><j| with rates from bath)
        """
        sys = self.system
        n_atoms = sys.n_atoms
        c_ops = []

        # Dephasing rate (assumed identical structure for all single excitations)
        deph_rate = bath_to_rates(self.bath, mode="deph")

        for i_atom in range(1, n_atoms + 1):
            # singles dephasing
            c_ops.append(sys.deph_op_i(i_atom) * np.sqrt(deph_rate))

            # Radiative-like single-site relaxation (singles -> ground) and thermal excitation
            freq = sys.frequencies[i_atom - 1]
            down_rate, up_rate = bath_to_rates(self.bath, freq, mode="decay")
            L_down = sys.basis[0] * sys.basis[i_atom].dag()  # |0><i|
            if down_rate > 0:
                c_ops.append(L_down * np.sqrt(down_rate))
            if up_rate > 0:
                c_ops.append(L_down.dag() * np.sqrt(up_rate))  # |i><0|

        # Double-state dephasing if manifold present
        index_to_pair = sys._index_to_pair
        for idx, (i, j) in index_to_pair.items():  # i<j atom labels
            c_ops.append(ket2dm(sys.basis[idx]) * np.sqrt(deph_rate))

            """
            # Double -> single lowering (if double manifold present)
            freq_idx = sys.frequencies[idx - 1]
            freq_i = sys.frequencies[i - 1]
            freq_j = sys.frequencies[j - 1]

            down_rate_i, up_rate_i = bath_to_rates(self.bath, (freq_idx - freq_i), mode="decay")
            down_rate_j, up_rate_j = bath_to_rates(self.bath, (freq_idx - freq_j), mode="decay")
            # |i> corresponds to basis index i, etc.
            Li = sys.basis[i] * sys.basis[idx].dag()
            c_ops.append([Li * np.sqrt(down_rate_i)])
            c_ops.append([Li.dag() * np.sqrt(up_rate_i)])

            Lj = sys.basis[j] * sys.basis[idx].dag()
            c_ops.append([Lj * np.sqrt(down_rate_j)])
            c_ops.append([Lj.dag() * np.sqrt(up_rate_j)])
            """

        return c_ops

    @property
    def theta(self) -> float:
        """Return dimer mixing angle θ (radians) for n_atoms == 2.

        Definition (standard exciton / coupled two-level system):

            tan(2θ) = 2J / Δ

        where
            J  = coupling (fs^-1)
            Δ  = ω_1 - ω_2 (fs^-1)  (bare transition frequency detuning)

        We compute:
            θ = 0.5 * arctan2(2J, Δ)

        Range: θ ∈ (-π/4, π/4]; magnitude governs the degree of state mixing.
        The previous implementation used arctan(J/Δ), which differs by a
        factor-of-two in the argument and is non-standard. Coefficients used
        in `lowering_op` retain their algebraic definitions (sinθ, cosθ), so this
        correction yields physically consistent mixing when J ~ Δ.

        Raises
        ------
        ValueError: if called for systems with n_atoms != 2.
        """
        if self.system.n_atoms != 2:
            raise ValueError("theta is only defined for n_atoms == 2")
        detuning = self.system.frequencies[0] - self.system.frequencies[1]  # Δ
        return 0.5 * np.arctan2(2 * self.system.coupling, detuning)

    # only for the paper solver
    def gamma_small_ij(self, i: int, j: int) -> float:
        """
        Calculate the population relaxation rates. for the dimer system, analogous to the gamma_ij in the paper.

        Parameters:
            i (int): Index of the first state.
            j (int): Index of the second state.

        Returns:
            float: Population relaxation rate.
        """

        w_ij = self.system.omega_ij(i, j)
        return np.sin(2 * self.theta) ** 2 * self.bath.power_spectrum(w_ij)

    def Gamma_big_ij(self, i: int, j: int) -> float:
        """
        Calculate the pure dephasing rates. for the dimer system, analogous to the gamma_ij in the paper.

        Parameters:
            i (int): Index of the first state.
            j (int): Index of the second state.

        Returns:
            float: Pure dephasing rate.
        """
        # Pure dephasing rates helper
        P_0 = self.bath.power_spectrum(0)
        Gamma_t_ab = 2 * np.cos(2 * self.theta) ** 2 * P_0  # tilde
        Gamma_t_a0 = (1 - 0.5 * np.sin(2 * self.theta) ** 2) * P_0
        Gamma_11 = self.gamma_small_ij(2, 1)
        Gamma_22 = self.gamma_small_ij(1, 2)
        Gamma_abar_0 = 2 * P_0
        Gamma_abar_a = Gamma_abar_0  # holds for dimer
        if i == 1:
            if j == 0:
                return Gamma_t_a0 + 0.5 * self.gamma_small_ij(2, i)
            elif j == 1:
                return Gamma_11
            elif j == 2:
                return Gamma_t_ab + 0.5 * (
                    self.gamma_small_ij(i, j) + self.gamma_small_ij(j, i)
                )
        if i == 2:
            if j == 0:
                return Gamma_t_a0 + 0.5 * self.gamma_small_ij(1, i)
            elif j == 1:
                return Gamma_t_ab + 0.5 * (
                    self.gamma_small_ij(i, j) + self.gamma_small_ij(j, i)
                )
            elif j == 2:
                return Gamma_22
        elif i == 3:
            if j == 0:
                return Gamma_abar_0
            elif j == 1:
                return Gamma_abar_a + 0.5 * (self.gamma_small_ij(2, j))
            elif j == 2:
                return Gamma_abar_a + 0.5 * (self.gamma_small_ij(1, j))
        else:
            raise ValueError("Invalid indices for i and j.")

    def summary(self) -> str:
        lines = [
            "=== SystemBathCoupling Summary ===",
            "System Parameters:",
            str(self.system),
            "Bath Parameters:",
            str(self.bath),
            "Decay Channels:",
            f"  Bloch-Redfield decay channels: {len(self.br_decay_channels)}",
            f"  Lindblad decay channels: {len(self.me_decay_channels)}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


if __name__ == "__main__":
    from qutip import OhmicEnvironment

    # Create mock instances of AtomicSystem and BathSystem
    atomic_system = AtomicSystem(n_atoms=1, frequencies_cm=[16000.0], dip_moments=[1.0])
    bath_class = OhmicEnvironment(alpha=0.1, T=0.1, wc=10, s=1.0)

    # Instantiate SystemBathCoupling
    system_bath_coupling = SystemBathCoupling(system=atomic_system, bath=bath_class)

    # Call the summary method
    system_bath_coupling.summary()
