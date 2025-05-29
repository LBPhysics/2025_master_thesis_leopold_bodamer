# =============================
# DEFINE THE SYSTEM PARAMETERS CLASS
# =============================
from dataclasses import dataclass, field  # for the class definiton
from typing import Optional  # for the class definiton
import numpy as np
from qutip import basis, ket2dm, tensor, Qobj, BosonicEnvironment

# bath functions
from src.baths.bath_fcts import (
    spectral_density_func_paper,
    Power_spectrum_func_paper,
)


@dataclass
class SystemParameters:
    # =============================
    # Fundamental constants / Quantum states(unchanged)
    # =============================
    hbar: float = 1.0
    Boltzmann: float = 1.0

    atom_g: Optional[Qobj] = field(default_factory=lambda: basis(2, 0))
    atom_e: Optional[Qobj] = field(default_factory=lambda: basis(2, 1))

    # Temperature / cutoff of the bath
    Temp: float = 1.0
    cutoff_: float = 1.0  # later * omega_A
    # =============================
    # system size
    # =============================
    N_atoms: int = 1
    # Note: N_atoms validation is done in __post_init__

    # =============================
    # Solver and model control
    # =============================
    ODE_Solver: str = (
        "Paper_BR"  # "Paper_eqs" (solve the EOMs from the paper) or "Paper_BR" do d/dt rho = -i/hbar * [H0 - Dip * E, rho] + R(rho)
    )
    RWA_laser: bool = (
        True  #  CAN ONLY HANDLE TRUE For Paper_eqs   #   only valid for omega_laser ~ omega_A
    )

    # =============================
    # Laser field parameters
    # =============================
    E0: float = 0.05

    # =============================
    # Pulse and time grid parameters
    # =============================
    pulse_duration: float = 15.0  # in fs
    t_max: float = 10.0  # in fs
    dt: float = 1.0  # in fs
    omega_laser_cm: float = 16000.0
    # in cm-1

    # =============================
    # Energy and transition parameters in cm-1
    # Declare these as Optional fields. Their defaults will be set in __post_init__.
    # =============================
    Delta_cm: Optional[float] = None
    omega_A_cm: Optional[float] = None
    omega_B_cm: Optional[float] = None  # Specific to N_atoms = 2
    mu_A: Optional[float] = None
    mu_B: Optional[float] = None  # Specific to N_atoms = 2
    J_cm: Optional[float] = None  # Specific to N_atoms = 2

    # =============================
    # Decoherence and relaxation rates
    # Declare these as Optional fields. Their defaults will be set in __post_init__.
    # =============================
    gamma_0: Optional[float] = None
    gamma_phi: Optional[float] = None

    # Derived attribute, will be set in __post_init__
    psi_ini: Optional[Qobj] = None

    # =============================
    # Properties for all derived quantities
    # =============================
    def __post_init__(self):
        """
        Initialize derived parameters after the class is instantiated.
        This method is called automatically after __init__.
        """
        # Assert N_atoms value for the instance
        if self.N_atoms not in [1, 2]:
            raise ValueError(
                "N_atoms must be 1 or 2 for this SystemParameters instance."
            )

        # Initialize quantum states if they were left as None (though default_factory should handle this)
        if self.atom_g is None:
            self.atom_g = basis(2, 0)
        if self.atom_e is None:
            self.atom_e = basis(2, 1)

        if self.gamma_0 is None:
            self.gamma_0 = 1 / 300.0
        if self.gamma_phi is None:
            self.gamma_phi = (
                1 / 100.0
            )  # units different in N_atoms=1 and N_atoms=2 ?!?!?!? TODO

        _H0_temp = None  # temoporary Hamiltonian

        # Set N_atoms-dependent parameters if they were not provided by the user
        if self.N_atoms == 1:
            self.psi_ini = ket2dm(self.atom_g)

            if self.omega_A_cm is None:
                # Use the instance's omega_laser_cm, which might have been overridden by the user
                self.omega_A_cm = self.omega_laser_cm
            if self.mu_A is None:
                self.mu_A = 1.0

            # For N_atoms=1, omega_B_cm, mu_B, J_cm are not typically used,
            # but ensure they are None if not set, or handle as needed.
            if (
                self.omega_B_cm is not None
                or self.mu_B is not None
                or self.J_cm is not None
            ):
                # Or raise a warning/error if these are set for N_atoms=1
                pass

        elif self.N_atoms == 2:
            self.psi_ini = ket2dm(tensor(self.atom_g, self.atom_g))

            if self.omega_A_cm is None:
                self.omega_A_cm = self.omega_laser_cm + 360.0
            if self.omega_B_cm is None:
                self.omega_B_cm = self.omega_laser_cm - 360.0
            if self.mu_A is None:
                self.mu_A = 1.0
            if self.mu_B is None:  # Default mu_B to mu_A if not specified
                self.mu_B = self.mu_A
            if self.J_cm is None:
                self.J_cm = 0.0  # Default coupling

        # inhomogenity (necessary for the delayed photon effect to appear) TODO: I dont observe this!
        if self.Delta_cm is None:
            self.Delta_cm = 200.0

        # Store frequency values in fs units # Not needed!
        self._fs_values = {}

        # Find all attributes ending with '_cm' and convert them
        for attr_name in dir(self):
            if attr_name.endswith("_cm") and not attr_name.startswith("__"):
                try:
                    cm_value = getattr(self, attr_name)
                    if isinstance(cm_value, (int, float)) and not callable(cm_value):
                        fs_name = attr_name[:-3]  # Remove '_cm' suffix
                        self._fs_values[fs_name] = self.convert_cm_to_fs(cm_value)
                except (AttributeError, TypeError):
                    pass

        if self.N_atoms == 1:
            _H0_temp = self.Hamilton_tls()
        elif self.N_atoms == 2:
            _H0_temp = self.Hamilton_dimer_sys()

        if _H0_temp is not None:
            self.H0_undiagonalized = _H0_temp  # Store the original H0 if needed
            # The H0_diagonalized property will use H0_undiagonalized
            self.H0 = (
                self.H0_diagonalized
            )  # Access property, self.H0 should become a Qobj
        else:
            # Handle case where H0 could not be initialized
            raise ValueError("Hamiltonian H0 could not be initialized.")

    def convert_cm_to_fs(self, value):
        """
        Convert the wavenumber-frequencies from cm^-1 to angular frequency fs^-1

        Parameters:
            value (float): Value in cm^-1

        Returns:
            float: Value in fs^-1
        """
        return value * 2.998 * 2 * np.pi * 10**-5

    def Hamilton_tls(self) -> Qobj:
        """
        Returns:
            Qobj: Hamiltonian operator of the two-level system.
        """
        # in canonical basis

        H0 = self.hbar * self.omega_A * ket2dm(self.atom_e)
        return H0

    def Hamilton_dimer_sys(self) -> Qobj:
        """
        Hamiltonian of a dimer system (two coupled(J) tls').

        Returns:
            Qobj: Hamiltonian operator of the two-level system
        """
        # in canonical basis
        H = self.hbar * (
            self.omega_A * ket2dm(tensor(self.atom_e, self.atom_g))
            + self.omega_B * ket2dm(tensor(self.atom_g, self.atom_e))
            + self.J
            * (
                tensor(self.atom_e, self.atom_g)
                * tensor(self.atom_g, self.atom_e).dag()
                + tensor(self.atom_g, self.atom_e)
                * tensor(self.atom_e, self.atom_g).dag()
            )
            + (self.omega_A + self.omega_B) * ket2dm(tensor(self.atom_e, self.atom_e))
        )
        return H

    @property
    def eigenstates(self):
        """
        Calculate the eigenvalues and eigenstates of the H0_undiagonalized Hamiltonian.

        Returns:
            tuple: (Es, kets)
                Es (np.ndarray): NumPy array of eigenvalues.
                kets (list of Qobj): List of Qobj eigenstates (kets).
        """
        # self.H0_undiagonalized.eigenstates() from QuTiP returns a tuple:
        # (array_of_eigenvalues, list_of_qobj_eigenvectors)
        Es, kets = self.H0_undiagonalized.eigenstates()
        return Es, kets

    @property
    def H0_diagonalized(self):
        """
        Diagonalize the Hamiltonian and return the eigenvalues and eigenstates.

        Returns:
            tuple: Eigenvalues and eigenstates of the Hamiltonian.
        """
        Es, _ = self.eigenstates

        if self.RWA_laser:
            if self.N_atoms == 1:
                Es[1] -= self.omega_laser

            elif self.N_atoms == 2:
                Es[1] -= self.omega_laser
                Es[2] -= self.omega_laser
                Es[3] -= 2 * self.omega_laser

        H_diag = Qobj(np.diag(Es), dims=self.H0_undiagonalized.dims)
        return H_diag

    @property
    def omega_A(self):  # in fs
        return self.convert_cm_to_fs(self.omega_A_cm)

    @property
    def omega_B(self):  # in fs
        return self.convert_cm_to_fs(self.omega_B_cm)

    @property
    def J(self):  # in fs
        return self.convert_cm_to_fs(self.J_cm)

    @property
    def theta(self):
        return np.arctan(2 * self.J / (self.omega_A * self.omega_B)) / 2

    @property
    def omega_laser(self):  # in fs
        return self.convert_cm_to_fs(self.omega_laser_cm)

    @property
    def Delta(self):  # in fs
        return self.convert_cm_to_fs(self.Delta_cm)

    @property
    def Gamma(self):
        return self.gamma_0 / 2 + self.gamma_phi

    @property
    def rabi_0(self):
        return self.mu_A * self.E0 / self.hbar

    @property
    def delta_rabi(self):
        return self.omega_laser - self.omega_A

    @property
    def rabi_gen(self):
        return np.sqrt(self.rabi_0**2 + self.delta_rabi**2)

    @property
    def t_prd(self):
        return 2 * np.pi / self.rabi_gen if self.rabi_gen != 0 else 0.0

    @property
    def FWHMs(self):
        return [self.pulse_duration / 2] * 3

    @property
    def E_freqs(self):
        return [self.omega_laser] * 3

    @property
    def E_amps(self):
        return [self.E0, self.E0, 1e-1 * self.E0]

    @property
    def SM_op(self):
        if self.N_atoms == 1:
            SM_op = self.mu_A * (self.atom_g * self.atom_e.dag()).unit()
        elif (
            self.N_atoms == 2
        ):  # TODO THIS IS ONLY FOR THE COUPLED / DIAGONALIZED HAMILTONIAN
            C_A_1 = -np.sin(self.theta)
            C_A_2 = np.cos(self.theta)
            C_B_1 = C_A_2
            C_B_2 = -C_A_1

            mu_10 = self.mu_A * C_A_1 + self.mu_B * C_A_2
            mu_20 = self.mu_A * C_B_1 + self.mu_B * C_B_2
            mu_31 = self.mu_B * C_A_1 + self.mu_A * C_B_1
            mu_32 = self.mu_B * C_B_1 + self.mu_A * C_B_2
            _, eigenvecs = self.eigenstates

            sm_list = [
                mu_10 * (eigenvecs[0] * eigenvecs[1].dag()).unit(),
                mu_20 * (eigenvecs[0] * eigenvecs[2].dag()).unit(),
                mu_31 * (eigenvecs[1] * eigenvecs[3].dag()).unit(),
                mu_32 * (eigenvecs[2] * eigenvecs[3].dag()).unit(),
            ]
            SM_op = sum(sm_list)
        else:
            raise ValueError("Only N_atoms=1 or 2 are supported.")
        return SM_op

    @property
    def Dip_op(self):
        return self.SM_op + self.SM_op.dag()

    @property
    def Deph_op(self):
        if self.N_atoms == 1:
            Deph_op = ket2dm(self.atom_e)
        elif self.N_atoms == 2:
            cplng_ops_to_env = [
                ket2dm(tensor(self.atom_e, self.atom_g)),
                ket2dm(tensor(self.atom_g, self.atom_e)),
                ket2dm(tensor(self.atom_e, self.atom_e)),
            ]
            Deph_op = sum(cplng_ops_to_env)
        else:
            raise ValueError("Only N_atoms=1 or 2 are supported.")

        return Deph_op

    @property
    def e_ops_list(self):
        if self.N_atoms == 1:
            e_ops_list = [
                ket2dm(self.atom_g),
                self.atom_g * self.atom_e.dag(),
                self.atom_e * self.atom_g.dag(),
                ket2dm(self.atom_e),
            ]
        elif self.N_atoms == 2:
            """
            e_ops_list1 = [
                ket2dm(tensor(self.atom_g, self.atom_g)),
                ket2dm(tensor(self.atom_e, self.atom_g)),
                ket2dm(tensor(self.atom_g, self.atom_e)),
                ket2dm(tensor(self.atom_e, self.atom_e)),
            ]
            """
            e_ops_list2 = [ket2dm(state) for state in self.eigenstates[1]]

            e_ops_list = e_ops_list2

        else:
            raise ValueError("Only N_atoms=1 or 2 are supported.")

        return e_ops_list

    @property
    def e_ops_labels(self):
        if self.N_atoms == 1:
            e_ops_labels = ["gg", "ge", "eg", "ee"]
        elif self.N_atoms == 2:
            # e_ops_labels1 = [f"{i}" for i in range(len(self.eigenstates[1]))]
            e_ops_labels2 = ["0", "A", "B", "AB"]
            e_ops_labels = e_ops_labels2

        else:
            raise ValueError("Only N_atoms=1 or 2 are supported.")

        return e_ops_labels

    @property
    def c_ops_list(self):  # TODO not including temperature!
        Gamma = self.Gamma
        gamma_phi = self.gamma_phi

        if self.N_atoms == 1:
            c_ops_list = [
                np.sqrt(Gamma) * self.SM_op,
                np.sqrt(gamma_phi) * self.Deph_op,
            ]
        elif self.N_atoms == 2:
            c_ops_list = [self.Deph_op]
        else:
            raise ValueError("Only N_atoms=1 or 2 are supported.")

        return c_ops_list

    @property
    def cutoff(self):
        return self.cutoff_ * self.omega_A

    @property
    def args_bath(self):
        return {
            "g": self.gamma_phi,
            "cutoff": self.cutoff,
            "Boltzmann": self.Boltzmann,
            "hbar": self.hbar,
            "Temp": self.Temp,
        }

    @property
    def a_ops_list(self):
        env = BosonicEnvironment.from_spectral_density(
            lambda w: spectral_density_func_paper(w, self.args_bath),
            wMax=10 * self.cutoff,
            T=self.Temp,
        )
        if self.N_atoms == 1:
            a_ops_list = [
                [self.Deph_op, env.power_spectrum],
            ]  # TODO THIS WAS NOT IN THE PAPER!!!!

        elif self.N_atoms == 2:
            cplng_ops_to_env = [
                ket2dm(tensor(self.atom_e, self.atom_g)),  # atom A
                ket2dm(tensor(self.atom_g, self.atom_e)),  # atom B
                ket2dm(tensor(self.atom_e, self.atom_e)),  # double excited state
            ]
            a_ops_list = [
                [
                    cplng_ops_to_env[0],
                    env.power_spectrum,
                ],  # atom A with ohmic_spectrum
                [
                    cplng_ops_to_env[1],
                    env.power_spectrum,
                ],  # atom B with ohmic_spectrum
                [
                    cplng_ops_to_env[2],
                    lambda w: env.power_spectrum(2 * w),
                ],  # double excited state with 2 * ohmic_spectrum
            ]
        else:
            raise ValueError("Only N_atoms=1 or 2 are supported.")

        return a_ops_list

    def omega_ij(self, i: int, j: int) -> float:
        """
        Calculate the energy difference between two states.

        Parameters:
            i (int): Index of the first state.
            j (int): Index of the second state.

        Returns:
            float: Energy difference between the two states.
        """
        return self.eigenstates[0][i] - self.eigenstates[0][j]  # energy difference

    def gamma_small_ij(self, i: int, j: int) -> float:
        """
        Calculate the population relaxation rates.

        Parameters:
            i (int): Index of the first state.
            j (int): Index of the second state.

        Returns:
            float: Population relaxation rate.
        """

        w_ij = self.omega_ij(i, j)
        return np.sin(2 * self.theta) ** 2 * Power_spectrum_func_paper(
            w_ij, self.args_bath
        )

    def Gamma_big_ij(self, i: int, j: int) -> float:
        """
        Calculate the pure dephasing rates.

        Parameters:
            i (int): Index of the first state.
            j (int): Index of the second state.

        Returns:
            float: Pure dephasing rate.
        """
        # Pure dephasing rates helper
        P_0 = Power_spectrum_func_paper(0, self.args_bath)
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

    def summary(self):
        """
        Print a structured summary of system parameters, derived quantities, and solver-specific info.
        """
        print("=== SystemParameters Summary ===")

        # System Configuration
        print("\n# The system with:")
        print(f"    {'N_atoms':<20}: {self.N_atoms}")
        print(f"    {'ODE_Solver':<20}: {self.ODE_Solver}")
        print(f"    {'RWA_laser':<20}: {self.RWA_laser}")
        print("was analyzed.")

        # Simulation Parameters
        print("\n# With parameters for the SIMULATION:")
        print(f"    {'t_max':<20}: {self.t_max} fs")
        print(f"    {'dt':<20}: {self.dt} fs")
        print(f"    {'pulse_duration':<20}: {self.pulse_duration} fs")
        print(f"    {'omega_laser':<20}: {self.omega_laser_cm} cm^-1")
        print(f"    {'E0':<20}: {self.E0} (mu*E0, such that excitation is < 1%!)")

        # Atom/Energy Parameters
        print("\n# With parameters for the ATOMS:")
        print(f"    {'omega_A':<20}: {self.omega_A_cm} cm^-1")
        print(f"    {'mu_A':<20}: {self.mu_A}")
        if self.N_atoms == 1:
            if self.Delta_cm is not None:
                print(f"    {'Delta':<20}: {self.Delta_cm} cm^-1")
        elif self.N_atoms == 2:
            if self.omega_B_cm is not None:
                print(f"    {'omega_B':<20}: {self.omega_B_cm} cm^-1")
            if self.mu_B is not None:
                print(f"    {'mu_B':<20}: {self.mu_B}")
            if self.J_cm is not None:
                print(f"    {'J':<20}: {self.J_cm} cm^-1")

        # Bath Parameters
        print("\n# With parameters for the BATH:")
        if self.gamma_0 is not None:
            print(f"    {'gamma_0':<20}: {self.gamma_0:.4f} fs-1?")
        if self.gamma_phi is not None:
            print(f"    {'gamma_phi':<20}: {self.gamma_phi:.4f} fs-1?")
        print(f"    {'Temp':<20}: {self.Temp}")
        print(f"    {'cutoff':<20}: {self.cutoff / self.omega_A:.1f} omega_A")

        # Derived Quantities
        print("\n# Additional generated parameters are:")

        print(f"\n    {'psi_ini':<20}:")
        print(self.psi_ini)
        print(
            f"\n    {'System Hamiltonian (diagonalized)':<20}:"
        )  # H0 is H0_diagonalized
        print(self.H0)

        # Solver Specifics
        print("\n# Solver specific information:")
        if self.ODE_Solver == "ME":
            print(f"    {'c_ops_list':<20}:")
            for op in self.c_ops_list:
                print(f"        {op}")
        elif self.ODE_Solver == "BR":
            print(f"    {'a_ops_list':<20}:")
            for op_spec in self.a_ops_list:
                if isinstance(op_spec, list) and len(op_spec) == 2:
                    print(f"        Operator: {op_spec[0]}, Spectrum: {op_spec[1]}")
                else:
                    print(f"        {op_spec}")
        elif self.ODE_Solver == "Paper_BR":
            print(f"    {'Redfield tensor R_paper used (calculated by R_paper(self))'}")
            # Optionally print the R_paper matrix if it's not too large
            # print(R_paper(self))
        elif self.ODE_Solver == "Paper_eqs":
            print(
                f"    {'Custom ODE matrix used (calculated by matrix_ODE_paper(t, pulse_seq, self))'}"
            )

        # Operators
        print("\n# Dipole operator (Dip_op):")
        print(self.Dip_op)
        print("\n# Expectation operator labels (e_ops_labels):")
        print(self.e_ops_labels)
        print("\n=== End of Summary ===")


if __name__ == "__main__":
    """
    Test the SystemParameters class when run directly.
    """
    print("Testing SystemParameters class...")

    # Test with N_atoms=1
    print("\n=== Testing N_atoms=1 ===")
    system1 = SystemParameters(N_atoms=1)
    system1.summary()

    # Test with N_atoms=2
    print("\n=== Testing N_atoms=2 ===")
    system2 = SystemParameters(N_atoms=2)
    system2.summary()

    print("\n✅ SystemParameters tests completed successfully!")
