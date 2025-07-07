from dataclasses import dataclass, field  # for the class definiton
import numpy as np
from qutip import tensor, ket2dm, Qobj

# util function from qutip
from qutip.utilities import n_thermal

from qspectro2d.core.system_parameters import SystemParameters
from qspectro2d.baths.bath_parameters import BathParameters

from qspectro2d.core.utils_and_config import BOLTZMANN, HBAR

@dataclass
class SystemBathCoupling:
    bath: BathParameters
    system: SystemParameters

    # DERIVED QUANTITIES FROM SYSTEM / BATH PARAMETERS
    @property
    def cutoff(self):
        return self.bath.cutoff_ * self.system.omega_A


    def args_bath(self, alpha=None):
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
            self.system.omega_A / (2 * self.bath.BOLTZMANN * self.bath.Temp / self.bath.HBAR)
        )
        alpha = (
            gamma / self.system.omega_A * np.exp(self.system.omega_A / self.cutoff) / (1 + coth_term)
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
            deph_A = ket2dm(tensor(self.system.atom_e, self.system.atom_g))
            deph_B = ket2dm(tensor(self.system.atom_g, self.system.atom_e))
            deph_AB = ket2dm(tensor(self.system.atom_e, self.system.atom_e))  # optional

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
        else: # TODO IMPLEMENT THE GENERAL CASE WITHIN SINGLE EXCITATION SUBSPACE
            raise ValueError("Only N_atoms=1 or 2 are supported.")

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
                SM_op * np.sqrt(relaxation_rate), #  * (n_th_at + 1)) # Collapse operator for thermal relaxation
                Deph_op * np.sqrt(
                    2 * total_dephasing  # * (2 * n_th_at + 1)  # factor 2 because of |exe|
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
            raise ValueError("Only N_atoms=1 or 2 are supported.")

        return me_decay_channels_

    def summary(self):
        print("=== SystemBathCoupling Summary ===")
        for i in range(self.system.N_atoms):
            for j in range(self.system.N_atoms):
                if i != j:
                    print(f"γ({i},{j}) = {self.gamma_ij(i,j):.4f} fs⁻¹")


from qspectro2d.core.pulse_sequences import PulseSequence
@dataclass
class SystemLaserCoupling:
    laser: PulseSequence
    system: SystemParameters

    # DERIVED QUANTITIES FROM SYSTEM / LASER PARAMETERS
    @property
    def rabi_frequency(self) -> float:
        μ = self.system.transition_dipole_moment  # ggf. aus omega und J berechnet
        return μ * self.laser.E0 / HBAR  # oder wie auch immer du es definierst


    # The H0_diagonalized property will use H0_undiagonalized
    @property
    def H0_diagonalized(self):
        """
        Diagonalize the Hamiltonian and return the eigenvalues and eigenstates.
        WITH RWA

        Returns:
            tuple: Eigenvalues and eigenstates of the Hamiltonian.
        """
        Es, _ = self.system.eigenstates

        if self.simulation.RWA_laser: # TODO
            if self.system.N_atoms == 1:
                Es[1] -= self.laser.omega

            elif self.system.N_atoms == 2:
                Es[1] -= self.laser.omega
                Es[2] -= self.laser.omega
                Es[3] -= 2 * self.laser.omega

        H_diag = Qobj(np.diag(Es), dims=self.H0_undiagonalized.dims)
        return H_diag
    



    @property
    def cutoff(self):
        return self.bath.cutoff_ * self.system.omega_A


    def args_bath(self, alpha=None):
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
            self.system.omega_A / (2 * self.bath.BOLTZMANN * self.bath.Temp / self.bath.HBAR)
        )
        alpha = (
            gamma / self.system.omega_A * np.exp(self.system.omega_A / self.cutoff) / (1 + coth_term)
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
            deph_A = ket2dm(tensor(self.system.atom_e, self.system.atom_g))
            deph_B = ket2dm(tensor(self.system.atom_g, self.system.atom_e))
            deph_AB = ket2dm(tensor(self.system.atom_e, self.system.atom_e))  # optional

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
        else: # TODO IMPLEMENT THE GENERAL CASE WITHIN SINGLE EXCITATION SUBSPACE
            raise ValueError("Only N_atoms=1 or 2 are supported.")

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
                SM_op * np.sqrt(relaxation_rate), #  * (n_th_at + 1)) # Collapse operator for thermal relaxation
                Deph_op * np.sqrt(
                    2 * total_dephasing  # * (2 * n_th_at + 1)  # factor 2 because of |exe|
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
            raise ValueError("Only N_atoms=1 or 2 are supported.")

        return me_decay_channels_

    def summary(self):
        print("=== SystemLaserCoupling Summary ===")
        for i in range(self.system.N_atoms):
            for j in range(self.system.N_atoms):
                if i != j:
                    print(f"γ({i},{j}) = {self.gamma_ij(i,j):.4f} fs⁻¹")
