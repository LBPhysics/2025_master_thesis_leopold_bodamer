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
        """Generate the a_ops for the Bloch Redfield Master Equation solver."""
        sys = self.system
        br_decay_channels_ = []
        n_atoms = sys.n_atoms

        if n_atoms == 1:
            # deph_rate = 4.425221e-06
            # down_rate = 3.333437e-03
            ## alpha = rates_to_alpha(deph_rate, self.bath, sys.at_freqs_fs(0), mode="deph")
            ## self.bath.alpha = alpha  # set the alpha in the bath
            deph_i = sys.deph_op_i(0)  # dephasing operator for single atom
            dip_op = sys.dip_op
            br_decay_channels_ += [
                [
                    deph_i,
                    self.bath,
                ],
                [
                    dip_op,
                    self.bath,
                ],
            ]

        elif n_atoms == 2:  # TODO REDO this to implement the appendix C
            for i in range(n_atoms):
                deph_i = sys.deph_op_i(i)  # i = A, B
                br_decay_channels_.append(
                    [
                        deph_i,  # atom i dephasing
                        self.bath,
                    ]
                )

            # ALSO ADD THE DOUBLE EXCITED STATE
            deph_AB = ket2dm(sys.basis[3])
            br_decay_channels_ += [
                [
                    deph_AB,  # part from A on double excited state
                    self.bath,
                ],
                [
                    deph_AB,  # part from B on double excited state
                    self.bath,
                ],
            ]
        else:
            for i in range(n_atoms):  # from 0 to n_atoms - 1
                deph_i = sys.deph_op_i(i)  # i = A, B, C, ...
                br_decay_channels_.append(
                    [
                        deph_i,  # atom i dephasing
                        self.bath,
                    ]
                )
                """ TODO IF I ALSO WANT TO INCLUDE THE DECAY CHANNELS for each atom
                br_decay_channels_.append(
                    [
                        (
                            sys.basis[0]
                            * sys.basis[i].dag(),  # this is sm_m[i]
                            self.bath,
                        )
                        for i in range(1, sys.n_atoms)
                    ],
                )
                """

        return br_decay_channels_

    @property
    def me_decay_channels(self):
        """Generate the c_ops for the Linblad Master Equation solver."""
        sys = self.system
        n_atoms = sys.n_atoms

        me_decay_channels_ = []

        if n_atoms == 1:
            # deph_rate = 1 / 100
            # alpha = rates_to_alpha(deph_rate, self.bath, sys.at_freqs_fs(0), mode="deph")
            # self.bath.alpha = alpha  # set the alpha in the bath

            deph_rate = bath_to_rates(self.bath, mode="deph")
            deph_op_i = sys.deph_op_i(0)  # dephasing operator for single atom
            me_decay_channels_.append(
                [
                    deph_op_i
                    * np.sqrt(deph_rate),  # Collapse operator for pure dephasing
                ]
            )
            sm_op = sys.sm_op
            down_rate, up_rate = bath_to_rates(self.bath, sys.at_freqs_fs(0), mode="decay")
            me_decay_channels_.append(
                [
                    sm_op
                    * np.sqrt(down_rate),  # Collapse operator for population relaxation
                    sm_op.dag()
                    * np.sqrt(up_rate),  # Collapse operator for population excitation
                ],
            )

        elif n_atoms == 2:
            dephasing_rate = bath_to_rates(self.bath, mode="deph")
            for i in range(n_atoms):
                deph_op_i = sys.deph_op_i(i)
                me_decay_channels_.append(deph_op_i * np.sqrt(dephasing_rate))
            deph_AB = ket2dm(sys.basis[3])
            me_decay_channels_.append(
                deph_AB * np.sqrt(dephasing_rate)
            )  # double excited state dephasing
        else:
            dephasing_rate = bath_to_rates(self.bath, mode="deph")
            for i in range(n_atoms):
                deph_op_i = sys.deph_op_i(i)
                me_decay_channels_.append(deph_op_i * np.sqrt(dephasing_rate))
        return me_decay_channels_

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
