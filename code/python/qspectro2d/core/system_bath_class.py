from dataclasses import dataclass  # for the class definiton
from typing import Optional  # for type hinting
import numpy as np
from qutip import tensor, ket2dm

# util function from qutip
from qutip.utilities import n_thermal

from core.atomic_system.system_class import AtomicSystem
from qspectro2d.core.bath_system.bath_class import BathClass
from qspectro2d.core.bath_system.bath_fcts import power_spectrum_func_paper

from qspectro2d.core.utils_and_config import BOLTZMANN, HBAR


@dataclass
class SystemBathCoupling:
    system: AtomicSystem
    bath: BathClass

    # DERIVED QUANTITIES FROM SYSTEM / BATH PARAMETERS
    @property
    def cutoff(self):
        return self.bath.cutoff_ * self.system.omega_A

    def args_bath(self, alpha: Optional[float] = None) -> dict:
        """
        Generate arguments for the bath functions.

        Parameters:
            alpha (Optional[float]): Coupling constant. Defaults to self.gamma_0.

        Returns:
            dict: Arguments for the bath functions.
        """
        if alpha is None:
            alpha = self.bath.gamma_0
        return {
            "alpha": alpha,
            "cutoff": self.cutoff,
            "Boltzmann": BOLTZMANN,
            "hbar": HBAR,
            "Temp": self.bath.Temp,
            "s": 1.0,  # ohmic spectrum
        }

    def _coupling_paper(self, gamma):
        w_th = self.bath.BOLTZMANN * self.bath.Temp / self.bath.HBAR
        n_th_at = n_thermal(self.system.omega_A, w_th)
        alpha = (
            gamma
            / (1 + n_th_at)
            * self.cutoff
            / self.system.omega_A
            * np.exp(self.system.omega_A / self.cutoff)
        )  # for paper P(w) :> TODO PROBLEM?!?, also make omega_A dependent?
        # This is the coupling constant for the spectral density function in the paper

        return alpha

    def _coupling_ohmic(self, gamma):
        coth_term = 1 / np.tanh(
            self.system.omega_A
            / (2 * self.bath.BOLTZMANN * self.bath.Temp / self.bath.HBAR)
        )
        alpha = (
            gamma
            / self.system.omega_A
            * np.exp(self.system.omega_A / self.cutoff)
            / (1 + coth_term)
        )  # for ohmic P(w) :> TODO also make omega_A dependent?
        return alpha

    def _coupling_dl(self, gamma):
        # TODO IMPLEMENT THIS
        alpha = gamma
        return alpha

    def coupling(self, gamma):
        """
        Determine the coupling constant based on the bath type.

        Parameters:
            gamma (float): The decay rate.

        Returns:
            float: The coupling constant.
        """
        bath = self.bath.bath
        if bath == "paper":
            return self._coupling_paper(gamma)
        elif bath == "ohmic":
            return self._coupling_ohmic(gamma)
        elif bath == "dl":
            return self._coupling_dl(gamma)
        else:
            raise ValueError(f"Unknown bath type: {self.bath}")

    @property
    def br_decay_channels(self):
        """Generate the a_ops for the Bloch Redfield Master Equation solver."""
        total_dephasing = self.bath.Gamma
        relaxation_rate = self.bath.gamma_0

        alpha_deph = self.coupling(total_dephasing)
        alpha_decay = self.coupling(relaxation_rate)

        args_deph = self.args_bath(alpha_deph)
        args_decay = self.args_bath(alpha_decay)

        N_atoms = self.system.N_atoms
        if N_atoms == 1:
            Deph_op = self.system.Deph_op
            Dip_op = self.system.Dip_op
            br_decay_channels_ = [
                [
                    Deph_op,
                    lambda w: total_dephasing,
                ],
                [
                    Dip_op,
                    lambda w: self.bath.power_spectrum_func(w, args_decay),
                ],
            ]

        elif N_atoms == 2:  # TODO REDO this to implement the appendix C
            deph_A = ket2dm(self.system.basis[1])
            deph_B = ket2dm(self.system.basis[2])
            deph_AB = ket2dm(self.system.basis[3])

            br_decay_channels_ = [
                [
                    deph_A,  # atom A dephasing
                    lambda w: self.bath.power_spectrum_func(w, args_deph),
                ],
                [
                    deph_B,  # atom B dephasing
                    lambda w: self.bath.power_spectrum_func(w, args_deph),
                ],
                [
                    deph_AB,  # part from A on double excited state
                    lambda w: self.bath.power_spectrum_func(w, args_deph),
                ],
                [
                    deph_AB,  # part from B on double excited state
                    lambda w: self.bath.power_spectrum_func(w, args_deph),
                ],
            ]
        else:  # TODO IMPLEMENT THE GENERAL CASE WITHIN SINGLE EXCITATION SUBSPACE
            br_decay_channels_ = [
                [
                    (
                        ket2dm(self.system.basis[i]),
                        lambda w: self.bath.power_spectrum_func(w, args_deph),
                    )
                    for i in range(1, self.system.N_atoms)
                ],
                [
                    (
                        self.system.basis[0]
                        * self.system.basis[i].dag(),  # this is sm_m[i]
                        lambda w: self.bath.power_spectrum_func(w, args_decay),
                    )
                    for i in range(1, self.system.N_atoms)
                ],
            ]

        return br_decay_channels_

    @property
    def me_decay_channels(self):
        """Generate the c_ops for the Linblad Master Equation solver."""
        total_dephasing = self.bath.Gamma
        relaxation_rate = self.bath.gamma_0

        w_th = self.bath.BOLTZMANN * self.bath.Temp / self.bath.HBAR

        n_th_at = n_thermal(self.system.omega_A, w_th)
        N_atoms = self.system.N_atoms
        Deph_op = self.system.Deph_op
        SM_op = self.system.SM_op
        if N_atoms == 1:
            me_decay_channels_ = [
                # SM_op.dag() * np.sqrt(relaxation_rate * n_th_at),  # Collapse operator for thermal excitation
                SM_op
                * np.sqrt(
                    relaxation_rate
                ),  #  * (n_th_at + 1)) # Collapse operator for thermal relaxation
                Deph_op
                * np.sqrt(
                    2
                    * total_dephasing  # * (2 * n_th_at + 1)  # factor 2 because of |exe|
                ),  # Collapse operator for dephasing
            ]

        elif self.N_atoms == 2:
            me_decay_channels_ = [
                ket2dm(tensor(self.atom_e, self.atom_g))
                * np.sqrt(total_dephasing),  # * (n_th_at + 1)
                ket2dm(tensor(self.atom_g, self.atom_e))
                * np.sqrt(total_dephasing),  # * (n_th_at + 1)
                ket2dm(tensor(self.atom_e, self.atom_e))
                * np.sqrt(total_dephasing),  # * (n_th_at + 1)
            ]
        else:
            raise ValueError(
                "Only N_atoms=1 or 2 are supported."
            )  # TODO implement the general case

        return me_decay_channels_

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
        return np.sin(2 * self.system.theta) ** 2 * power_spectrum_func_paper(
            w_ij, self.bath.args_bath()
        )

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
        P_0 = power_spectrum_func_paper(0, self.SB_coupling.args_bath())
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

    def summary(self):
        print("=== SystemBathCoupling Summary ===")
        for i in range(self.system.N_atoms):
            for j in range(self.system.N_atoms):
                if i != j:
                    print(f"γ({i},{j}) = {self.gamma_ij(i,j):.4f} fs⁻¹")
