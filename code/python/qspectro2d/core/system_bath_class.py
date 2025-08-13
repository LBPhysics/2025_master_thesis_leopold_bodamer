from dataclasses import dataclass  # for the class definition
import numpy as np

# Use TYPE_CHECKING to avoid circular imports during runtime
from qspectro2d.core.atomic_system.system_class import AtomicSystem

from qutip import ket2dm, BosonicEnvironment

from qspectro2d.core.bath_system.bath_fcts import bath_to_rates, rates_to_alpha


@dataclass
class SystemBathCoupling:
    system: AtomicSystem
    bath: BosonicEnvironment

    @property
    def br_decay_channels(self):
        """Generate the a_ops list for Bloch-Redfield solver.
        Index  State (site1 site2 site3 site4)
        -----  -------------------------------
        0      0 0 0 0    (ground)
        1      1 0 0 0    (excite site 1)
        2      0 1 0 0    (excite site 2)
        3      0 0 1 0    (excite site 3)
        4      0 0 0 1    (excite site 4)
        5      1 1 0 0    (sites 1 & 2 excited)
        6      1 0 1 0    (sites 1 & 3 excited)
        7      1 0 0 1    (sites 1 & 4 excited)
        8      0 1 1 0    (sites 2 & 3 excited)
        9      0 1 0 1    (sites 2 & 4 excited)
        10     0 0 1 1    (sites 3 & 4 excited)

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
        max_exc = getattr(sys, "max_excitation", 1)

        # Helper: add dephasing projector for basis index k
        def add_projector(k: int):
            br_ops.append([ket2dm(sys.basis[k]), self.bath])

        if n_atoms == 1:
            # Single atom: dephasing + total dipole (emission/absorption channel)
            add_projector(1)
            br_ops.append([sys.dip_op, self.bath])
            return br_ops

        # Multi-atom: singles manifold projectors
        # For dimer truncated (max_exc=1) or generic N, singles indices are 1..n_atoms
        for single_idx in range(1, n_atoms + 1):
            add_projector(single_idx)

        # Double manifold projectors only if available
        if max_exc == 2 and n_atoms >= 2:
            # Dimer special-case: full basis has index 3 as double excitation
            if n_atoms == 2:
                add_projector(3)
            else:
                # Generic mapping stored internally
                index_to_pair = getattr(sys, "_index_to_pair", {})
                for idx in index_to_pair.keys():
                    add_projector(idx)

        # Radiative-like decay (lowering) operators from singles to ground: |0><i|
        for single_idx in range(1, n_atoms + 1):
            Li = sys.basis[0] * sys.basis[single_idx].dag()
            br_ops.append([Li, self.bath])

        # Double -> single lowering (if double manifold present)
        if max_exc == 2 and n_atoms >= 2:
            if n_atoms == 2:
                # |eg><ee| and |ge><ee|
                L1 = sys.basis[1] * sys.basis[3].dag()
                L2 = sys.basis[2] * sys.basis[3].dag()
                br_ops.append([L1, self.bath])
                br_ops.append([L2, self.bath])
            else:
                index_to_pair = getattr(sys, "_index_to_pair", {})
                for idx, (i, j) in index_to_pair.items():  # i<j atom labels
                    # |i> corresponds to basis index i, etc.
                    Li = sys.basis[i] * sys.basis[idx].dag()
                    Lj = sys.basis[j] * sys.basis[idx].dag()
                    br_ops.append([Li, self.bath])
                    br_ops.append([Lj, self.bath])

        return br_ops

    @property
    def me_decay_channels(self):
        """Generate c_ops for Lindblad solver respecting max_excitation.

        Strategy:
          - Pure dephasing: projectors on each populated basis state (singles; doubles if present)
          - Population relaxation/excitation: per-site lowering/raising (|0><i|, |i><0|) with rates from bath
            (double-manifold relaxation omitted for simplicity; can be added later)
        """
        sys = self.system
        n_atoms = sys.n_atoms
        max_exc = getattr(sys, "max_excitation", 1)
        c_ops = []

        # Dephasing rate (assumed identical structure for all single excitations)
        deph_rate = bath_to_rates(self.bath, mode="deph")

        if n_atoms == 1:
            # Single atom: projector on excited state
            c_ops.append(sys.deph_op_i(0) * np.sqrt(deph_rate))
            # Relaxation / excitation between |g> and |e>
            down_rate, up_rate = bath_to_rates(
                self.bath, sys.at_freqs_fs(0), mode="decay"
            )
            sm = sys.sm_op  # |g><e|
            c_ops.append(sm * np.sqrt(down_rate))
            if up_rate > 0:
                c_ops.append(sm.dag() * np.sqrt(up_rate))
            return c_ops

        # Multi-atom: singles dephasing
        for single_idx in range(1, n_atoms + 1):
            c_ops.append(ket2dm(sys.basis[single_idx]) * np.sqrt(deph_rate))

        # Double-state dephasing if manifold present
        if max_exc == 2:
            if n_atoms == 2:
                c_ops.append(ket2dm(sys.basis[3]) * np.sqrt(deph_rate))
            else:
                for idx in getattr(sys, "_index_to_pair", {}).keys():
                    c_ops.append(ket2dm(sys.basis[idx]) * np.sqrt(deph_rate))

        # Radiative-like single-site relaxation (singles -> ground) and thermal excitation
        for i_atom in range(1, n_atoms + 1):
            freq = sys.at_freqs_fs(i_atom - 1)
            down_rate, up_rate = bath_to_rates(self.bath, freq, mode="decay")
            L_down = sys.basis[0] * sys.basis[i_atom].dag()  # |0><i|
            if down_rate > 0:
                c_ops.append(L_down * np.sqrt(down_rate))
            if up_rate > 0:
                c_ops.append(L_down.dag() * np.sqrt(up_rate))  # |i><0|

        return c_ops

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
        return np.sin(2 * self.system.theta) ** 2 * self.bath.power_spectrum(w_ij)

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
        Gamma_t_ab = 2 * np.cos(2 * self.system.theta) ** 2 * P_0  # tilde
        Gamma_t_a0 = (1 - 0.5 * np.sin(2 * self.system.theta) ** 2) * P_0
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
    atomic_system = AtomicSystem(n_atoms=1, at_freqs_cm=[16000.0], dip_moments=[1.0])
    bath_class = OhmicEnvironment(alpha=0.1, T=0.1, wc=10, s=1.0)

    # Instantiate SystemBathCoupling
    system_bath_coupling = SystemBathCoupling(system=atomic_system, bath=bath_class)

    # Call the summary method
    system_bath_coupling.summary()
