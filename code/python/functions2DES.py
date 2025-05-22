# =============================
# FUNCTIONS for overlapping pulses
# =============================

from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional
import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import os

# functions from test_baths.py
from test_baths import (
    spectral_density_func_paper,
    Power_spectrum_func_paper,
)


### Phase Cycling for Averaging
phases = [k * np.pi / 2 for k in range(4)]

# =============================
# SYSTEM PARAMETERS
# =============================


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
    cutoff_: float = 1.0
    # =============================
    # system size
    # =============================
    N_atoms: int = 1
    assert N_atoms == 1 or N_atoms == 2, "This code only works for 1 or 2 atoms"

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
    fine_spacing: float = 1.0  # in fs
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

            if self.Delta_cm is None:
                self.Delta_cm = 200.0
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

            # For N_atoms=2, Delta_cm is not typically used for the dimer model presented
            if self.Delta_cm is not None:
                # Or raise a warning/error if Delta_cm is set for N_atoms=2
                pass

        # Store frequency values in fs units
        self._fs_values = {}  # TODO TEST THIS!! might be the wrong position

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
        Convert values from cm^-1 to fs^-1 (angular frequency)

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
    def t_max_L(self):
        return 6 * 2 * np.pi / self.omega_laser if self.omega_laser != 0 else 0.0

    @property
    def t_prd(self):
        return 2 * np.pi / self.rabi_gen if self.rabi_gen != 0 else 0.0

    @property
    def Delta_ts(self):
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
        if self.N_atoms == 1:
            a_ops_list = [self.Deph_op]
        elif self.N_atoms == 2:
            env = BosonicEnvironment.from_spectral_density(
                lambda w: spectral_density_func_paper(w, self.args_bath),
                wMax=10 * self.cutoff,
                T=self.Temp,
            )
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

    @property
    def omega_ij(self, i: int, j: int) -> float:
        # TODO i think i can include this in the system class
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
        print(f"    {'fine_spacing':<20}: {self.fine_spacing} fs")
        print(f"    {'pulse_duration':<20}: {self.pulse_duration} fs")
        print(f"    {'omega_laser':<20}: {self.omega_laser_cm} cm^-1")

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
        derived_quantities = ["rabi_0", "delta_rabi", "rabi_gen", "t_prd"]
        for name in derived_quantities:
            value = getattr(self, name, "N/A")
            if isinstance(value, float):
                print(f"    {name:<20}: {value:.3f}")
            else:
                print(f"    {name:<20}: {value}")

        print(f"\n    {'Initial state (psi_ini)':<20}:")
        print(self.psi_ini)
        print(
            f"\n    {'System Hamiltonian (H0_diagonalized)':<20}:"
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


# =============================
# Pulse and PulseSequence classes for structured pulse handling
# =============================


@dataclass
class Pulse:
    pulse_start_time: float
    pulse_half_width: float
    pulse_phase: float
    pulse_amplitude: float
    pulse_freq: float


@dataclass
class PulseSequence:
    pulses: list = field(default_factory=list)  # List of Pulse objects

    @staticmethod
    def from_args(
        system: SystemParameters,
        curr: tuple,
        prev: tuple = None,
        preprev: tuple = None,
    ) -> "PulseSequence":
        """
        Factory method to create a PulseSequence from argument tuples and lists,
        using a single global pulse_freq and Delta_t for all pulses.

        Parameters:
            curr (tuple): (start_time, phase) for the current pulse
            prev (tuple, optional): (start_time, phase) for the previous pulse
            preprev (tuple, optional): (start_time, phase) for the earliest pulse
            pulse_freq (float): Frequency for all pulses
            Delta_t (float): Half-width for all pulses
            E_amps (list): List of amplitudes for each pulse

        Returns:
            PulseSequence: An instance containing up to three pulses
        """
        pulse_freq = system.omega_laser
        Delta_ts = system.Delta_ts
        E_amps = system.E_amps

        pulses = []

        # Add the earliest pulse if provided (preprev)
        if preprev is not None:
            t0_preprev, phi_preprev = preprev
            pulses.append(
                Pulse(
                    pulse_start_time=t0_preprev,
                    pulse_phase=phi_preprev,
                    pulse_half_width=Delta_ts[0],
                    pulse_amplitude=E_amps[0],
                    pulse_freq=pulse_freq,
                )
            )

        # Add the previous pulse if provided (prev)
        if prev is not None:
            t0_prev, phi_prev = prev
            idx = 1 if preprev is not None else 0
            pulses.append(
                Pulse(
                    pulse_start_time=t0_prev,
                    pulse_phase=phi_prev,
                    pulse_half_width=Delta_ts[1],
                    pulse_amplitude=E_amps[idx],
                    pulse_freq=pulse_freq,
                )
            )

        # Always add the current pulse (curr)
        t0_curr, phi_curr = curr
        if preprev is not None and prev is not None:
            idx = 2
        elif preprev is not None or prev is not None:
            idx = 1
        else:
            idx = 0
        pulses.append(
            Pulse(
                pulse_start_time=t0_curr,
                pulse_phase=phi_curr,
                pulse_half_width=Delta_ts[idx],
                pulse_amplitude=E_amps[idx],
                pulse_freq=pulse_freq,
            )
        )

        return PulseSequence(pulses=pulses)

    def as_dict(self) -> dict:
        """
        Convert to dictionary format compatible with legacy code.

        Returns:
            dict: Dictionary with key "pulses" and a list of pulse parameter dicts
        """
        return {"pulses": [pulse.__dict__ for pulse in self.pulses]}


def pulse_envelope(t: float, pulse_seq: PulseSequence) -> float:
    """
    Calculate the combined envelope of multiple pulses at time t using PulseSequence.
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    envelope = 0.0
    for pulse in pulse_seq.pulses:
        t0 = pulse.pulse_start_time
        Delta_width = pulse.pulse_half_width
        if Delta_width is None or Delta_width <= 0:
            continue
        if t0 is None:
            continue
        if t0 <= t <= t0 + 2 * Delta_width:
            arg = np.pi * (t - (t0 + Delta_width)) / (2 * Delta_width)
            envelope += np.cos(arg) ** 2
    return envelope


def E_pulse(t: float, pulse_seq: PulseSequence) -> complex:
    """
    Calculate the total electric field at time t for a set of pulses (envelope only, no carrier), using PulseSequence.
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    E_total = 0.0 + 0.0j
    for pulse in pulse_seq.pulses:
        phi = pulse.pulse_phase
        E0 = pulse.pulse_amplitude
        if phi is None or E0 is None:
            continue
        envelope = pulse_envelope(
            t, PulseSequence([pulse])
        )  # use pulse_envelope for each pulse
        E_total += E0 * envelope * np.exp(-1j * phi)
    return E_total / 2.0


def Epsilon_pulse(t: float, pulse_seq: PulseSequence) -> complex:
    """
    Calculate the total electric field at time t for a set of pulses, including carrier oscillation, using PulseSequence.
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    E_total = 0.0 + 0.0j
    for pulse in pulse_seq.pulses:
        omega = pulse.pulse_freq
        if omega is None:
            continue
        E_field = E_pulse(t, PulseSequence([pulse]))  # use E_pulse for each pulse
        E_total += E_field * np.exp(-1j * (omega * t))
    return E_total


def plot_pulse_envelope(times: np.ndarray, pulse_seq: PulseSequence, ax=None):
    """
    Plot the combined pulse envelope over time for up to three pulses using PulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (PulseSequence): PulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        ax (matplotlib.axes.Axes): Axes object with the plot.
    """
    # Calculate the combined envelope over time
    envelope = [pulse_envelope(t, pulse_seq) for t in times]

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot combined envelope
    ax.plot(
        times,
        envelope,
        label=r"$\text{Combined Envelope}$",
        linestyle="solid",
        alpha=0.8,
        color="C0",
    )

    # Styles for up to three pulses
    linestyles = ["dashed", "dashdot", "dotted"]
    colors = ["C1", "C2", "C3"]

    # Plot individual envelopes and annotations
    for idx, pulse in enumerate(pulse_seq.pulses[:3]):  # Up to 3 pulses
        t0 = pulse.pulse_start_time
        Delta_width = pulse.pulse_half_width

        # Compute individual pulse envelope
        individual_envelope = [
            (
                np.cos(np.pi * (t - (t0 + Delta_width)) / (2 * Delta_width)) ** 2
                if t0 <= t <= t0 + 2 * Delta_width
                else 0.0
            )
            for t in times
        ]

        ax.plot(
            times,
            individual_envelope,
            label=rf"$\text{{Pulse {idx + 1}}}$",
            linestyle=linestyles[idx % len(linestyles)],
            alpha=0.6,
            color=colors[idx % len(colors)],
        )

        # Annotate pulse key points
        ax.axvline(
            t0,
            linestyle="dotted",
            label=rf"$t_{{0, {idx + 1}}}$",
            alpha=0.4,
            color=colors[idx % len(colors)],
        )
        ax.axvline(
            t0 + Delta_width,
            linestyle="dashdot",
            label=rf"$t_{{0, {idx + 1}}} + \Delta_{{{idx + 1}}}$",
            alpha=0.6,
            color=colors[idx % len(colors)],
        )
        ax.axvline(
            t0 + 2 * Delta_width,
            linestyle="dotted",
            label=rf"$t_{{0, {idx + 1}}} + 2\Delta_{{{idx + 1}}}$",
            alpha=0.4,
            color=colors[idx % len(colors)],
        )

    # Final plot labeling
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Envelope Amplitude")
    ax.set_title(r"Pulse Envelopes for Up to Three Pulses")
    ax.legend(loc="upper right", fontsize="small")
    return ax


def El_field_3_pulses(times: np.ndarray, pulse_seq: PulseSequence, f=pulse_envelope):
    """
    Calculate the combined electric field for a PulseSequence.

    Parameters:
        times (np.ndarray): Time range for the pulses.
        pulse_seq (PulseSequence): PulseSequence object.
        f (function): Function to compute field (pulse_envelope, E_pulse, or Epsilon_pulse).

    Returns:
        np.ndarray: Electric field values.
    """
    # Calculate the electric field for each time
    E = np.array([f(t, pulse_seq) for t in times])
    # Normalize if not envelope
    if f != pulse_envelope and len(pulse_seq.pulses) > 0:
        E0 = pulse_seq.pulses[0].pulse_amplitude
        if E0 != 0:
            E *= 0.5 * E0
    return E


def H_int(
    t: float,
    pulse_seq: PulseSequence,
    system: SystemParameters,
) -> Qobj:
    """
    Define the interaction Hamiltonian for the system with multiple pulses using the PulseSequence class.

    Parameters:
        t (float): Time at which the interaction Hamiltonian is evaluated.
        pulse_seq (PulseSequence): PulseSequence object containing all pulse parameters.
        system (SystemParameters): System parameters.
        SM_op (Qobj): Lowering operator (system-specific).
        Dip_op (Qobj): Dipole operator (system-specific).

    Returns:
        Qobj: Interaction Hamiltonian at time t.
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    SM_op = system.SM_op
    Dip_op = system.Dip_op

    if system.RWA_laser:
        E_field = E_pulse(t, pulse_seq)  # Combined electric field under RWA
        H_int = -(
            SM_op.dag() * E_field + SM_op * np.conj(E_field)
        )  # RWA interaction Hamiltonian
    else:
        E_field = Epsilon_pulse(t, pulse_seq)  # Combined electric field with carrier
        H_int = -Dip_op * (E_field + np.conj(E_field))  # Full interaction Hamiltonian

    return H_int


def plot_positive_color_map(
    datas: tuple,
    T_wait: float = np.inf,
    space: str = "real",
    type: str = "real",
    output_dir: str = None,
    ODE_Solver: str = None,
    positive: bool = False,
    safe: bool = False,
    use_custom_colormap: bool = False,
    section: tuple = None,  # (x_min, x_max, y_min, y_max)
    system: SystemParameters = None,
):
    """
    Create a color plot of 2D functional data for positive x and y values only.

    Parameters:
        datas (tuple): (x, y, data) where x and y are 1D arrays and data is a 2D array.
        T_wait (float): waiting time to include in plot title and file name.
        space (str): Either 'real' or 'freq' specifying the space of the data.
        type (str): Type of data ('real', 'imag', 'abs', or 'phase'). Used only if space="freq".
        output_dir (str, optional): Directory to save the plot.
        ODE_Solver (str, optional): Solver name for filename.
        positive (bool): Whether to use ONLY positive values of x and y.
        safe (bool): If True, saves the plot to a file.
        use_custom_colormap (bool): Use custom colormap with white at zero.
        section (tuple, optional): (x_min, x_max, y_min, y_max) to zoom in.

    Returns:
        None
    """
    # =============================
    # Validate input
    # =============================
    if not isinstance(datas, tuple) or len(datas) != 3:
        raise ValueError("datas must be a tuple of (x, y, data)")

    x, y, data = datas

    x = np.real(x)
    y = np.real(y)

    data = np.array(data, dtype=np.complex128)
    if np.abs(data).max() == 0:
        raise ValueError("Data array is all zeros, cannot normalize.")
    data = data / np.abs(data).max()  # normalize

    if data.shape[1] != len(x):
        raise ValueError(
            f"Length of x ({len(x)}) must match the number of columns in data ({data.shape[1]})."
        )
    if data.shape[0] != len(y):
        raise ValueError(
            f"Length of y ({len(y)}) must match the number of rows in data ({data.shape[0]})."
        )

    # =============================
    # Set plot labels and colormap
    # =============================
    if space not in ("real", "freq"):
        raise ValueError("Invalid space. Must be 'real' or 'freq'.")
    if space == "real":
        colormap = "viridis"
        title = r"$\text{Real space}$"
        x_title = r"$t_{\text{det}}$ [fs]"
        y_title = r"$\tau_{\text{coh}}$ [fs]"
    else:
        colormap = "plasma"
        title = r"$\text{Freq space}$"
        x_title = r"$\omega_{t_{\text{det}}}$ [$10^4$ cm$^{-1}$]"
        y_title = r"$\omega_{\tau_{\text{coh}}}$ [$10^4$ cm$^{-1}$]"

    if type not in ("real", "imag", "abs", "phase"):
        raise ValueError("Invalid Type. Must be 'real', 'imag', 'abs', or 'phase'.")
    if type == "real":
        title += r"$\text{, Real 2D Spectrum}$"
        data = np.real(data)
    elif type == "imag":
        title += r"$\text{, Imag 2D Spectrum}$"
        data = np.imag(data)
    elif type == "abs":
        title += r"$\text{, Abs 2D Spectrum}$"
        data = np.abs(data)
        use_custom_colormap = False
    elif type == "phase":
        title += r"$\text{, Phase 2D Spectrum}$"
        data = np.angle(data)

    if T_wait != np.inf:
        title += rf"$\ \text{{at }} T = {T_wait:.2f}$"

    # =============================
    # Restrict to positive quadrant if requested
    # =============================
    if positive:
        rows, cols = data.shape
        if rows % 2 != 0:
            data = data[:-1, :]
            y = y[:-1]
        if cols % 2 != 0:
            data = data[:, :-1]
            x = x[:-1]
        mid_x = len(x) // 2
        mid_y = len(y) // 2
        q1 = data[mid_y:, mid_x:]
        q3 = data[:mid_y, :mid_x]
        averaged_data = (q1 + np.flip(q3, axis=(0, 1))) / 2
        x = x[mid_x:]
        y = y[mid_y:]
        data = averaged_data

    # =============================
    # Section cropping
    # =============================
    if section is not None:
        x_min, x_max, y_min, y_max = section

        # Validate coordinates are within data range
        x_min = max(x_min, np.min(x))
        x_max = min(x_max, np.max(x))
        y_min = max(y_min, np.min(y))
        y_max = min(y_max, np.max(y))

        x_indices = np.where((x >= x_min) & (x <= x_max))[0]
        y_indices = np.where((y >= y_min) & (y <= y_max))[0]

        x_indices = x_indices[x_indices < data.shape[1]]
        y_indices = y_indices[y_indices < data.shape[0]]

        data = data[np.ix_(y_indices, x_indices)]
        x = x[x_indices]
        y = y[y_indices]

    # =============================
    # Custom colormap for zero-centered data
    # =============================
    norm = None
    if use_custom_colormap:
        vmin = np.min(data)
        vmax = np.max(data)
        vcenter = 0
        cmap = plt.get_cmap("bwr")
        colors = cmap(np.linspace(0, 1, 256))
        mid = 128
        colors[mid] = [1, 1, 1, 1]  # white at center
        colormap = LinearSegmentedColormap.from_list("white_centered", colors)
        if vmin < vcenter < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            print(
                f"Warning: Cannot use TwoSlopeNorm with vmin={vmin}, vcenter={vcenter}, vmax={vmax}. Using default normalization."
            )

    cbarlabel = r"$\propto E_{\text{out}} / E_{0}$"

    # =============================
    # Plotting
    # =============================
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(
        x,
        y,
        data,
        shading="auto",
        cmap=colormap,
        norm=norm,
    )
    plt.colorbar(label=cbarlabel)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    # =============================
    # Save or show
    # =============================

    if safe and output_dir and system is not None:
        if not os.path.isdir(output_dir):
            raise ValueError(f"Output directory {output_dir} does not exist.")
        filename_parts = [
            f"N={system.N_atoms}",
            f"mua={system.mu_A:.0f}",
            f"E0={system.E0:.2e}",
            f"wa={system.omega_A:.2f}",
            f"wL={system.omega_laser / system.omega_A:.1f}wa",
            f"rabigen={system.rabi_gen:.2f}= sqrt({system.rabi_0:.2f}^2+{system.delta_rabi:.2f}^2)",
            f"pos={positive}",
            f"space={space}",
        ]
        if ODE_Solver == "Paper_eqs":
            filename_parts.append(f"Paper_eqs")
        if space == "freq":
            filename_parts.append(f"type_{type}")
        file_name_combined = "_".join(filename_parts) + ".svg"
        save_path_combined = os.path.join(output_dir, file_name_combined)
        plt.savefig(save_path_combined)
    else:
        print("Plot not saved. Ensure 'safe' is True and 'output_dir' is specified.")
    plt.show()


def get_tau_cohs_and_t_dets_for_T_wait(
    times: np.ndarray, T_wait: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the time arrays for tau_coh and t_det based on the waiting time T_wait and the time grid.

    Parameters:
        times (np.ndarray): 1D array of time points (must be sorted and equally spaced).
        T_wait (float): Waiting time.

    Returns:
        tuple: Arrays for coherence and detection times (tau_coh, t_det).
               Both have the same length.
    """
    # =============================
    # Validate input
    # =============================
    if times.size == 0:
        raise ValueError("Input 'times' array must not be empty.")
    if times.size == 1:
        return np.array([0.0]), np.array([0.0])
    spacing = times[1] - times[0]
    t_max = times[-1]

    # =============================
    # Check T_wait validity
    # =============================
    if T_wait > t_max:
        print("Warning: T_wait >= t_max, no valid tau_coh/t_det values.")
        return np.array([]), np.array([])
    if np.isclose(T_wait, t_max):
        return np.array([0.0]), np.array([0.0])

    # =============================
    # Calculate tau_coh and t_det arrays
    # =============================
    tau_coh_max = t_max - T_wait
    if tau_coh_max < 0:
        return np.array([]), np.array([])

    tau_coh = np.arange(
        0, tau_coh_max + spacing / 2, spacing
    )  # include endpoint if possible
    t_det = tau_coh + T_wait

    # =============================
    # Ensure t_det does not exceed t_max due to floating point
    # =============================
    valid_idx = t_det <= t_max + 1e-10
    tau_coh = tau_coh[valid_idx]
    t_det = t_det[valid_idx]

    return tau_coh, t_det


def Plot_example_evo(
    times_0: np.ndarray,
    times_1: np.ndarray,
    times_2: np.ndarray,
    datas: list,
    pulse_seq_f: PulseSequence,
    tau_coh: float,
    T_wait: float,
    system: SystemParameters,
):
    """
    Plot the evolution of the electric field and expectation values for a given tau_coh and T_wait.

    Parameters:
        times_0, times_1, times_2 (np.ndarray): Time ranges for the three pulses.
        datas (list): List of arrays of expectation values to plot.
        pulse_seq_f: PulseSequence object for the final pulse sequence.
        tau_coh (float): Coherence time.
        T_wait (float): Waiting time.
        system: System object containing all relevant parameters.

    Returns:
        None
    """
    # =============================
    # PREPARE TIME AXIS AND FIELD
    # =============================
    times_plot = np.concatenate([times_0, times_1, times_2])

    # Choose field function depending on RWA
    if getattr(system, "RWA_laser", False):
        field_func = E_pulse
    else:
        field_func = Epsilon_pulse

    # Compute total electric field using the pulse sequence
    E_total = np.zeros_like(times_plot, dtype=np.complex128)
    # =============================
    # Calculate total electric field for each pulse in the sequence
    # =============================
    E0 = pulse_seq_f.pulses[0].pulse_amplitude
    E_total = np.array([field_func(t, pulse_seq_f) / E0 for t in times_plot])

    # =============================
    # PLOTTING
    # =============================
    plt.figure(figsize=(14, 2 + 2 * len(datas)))

    # Plot electric field
    plt.subplot(len(datas) + 1, 1, 1)
    plt.plot(
        times_plot,
        np.real(E_total),
        color="C0",
        linestyle="solid",
        label=r"$\mathrm{Re}[E(t)]$",
    )
    plt.plot(
        times_plot,
        np.imag(E_total),
        color="C1",
        linestyle="dashed",
        label=r"$\mathrm{Im}[E(t)]$",
    )
    plt.ylabel(r"$E(t) / E_0$")
    plt.axvline(
        times_0[0] + system.Delta_ts[0],
        color="C2",
        linestyle="dashed",
        label=r"Pulse 1",
    )
    plt.axvline(
        times_1[0] + system.Delta_ts[1],
        color="C3",
        linestyle="dashdot",
        label=r"Pulse 2",
    )
    plt.axvline(
        times_2[0] + system.Delta_ts[2],
        color="C4",
        linestyle="dotted",
        label=r"Pulse 3",
    )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Plot expectation values
    for idx, data in enumerate(datas):
        plt.subplot(len(datas) + 1, 1, idx + 2)
        if hasattr(system, "e_ops_labels") and idx < len(system.e_ops_labels):
            label = (
                r"$\mathrm{Re}\langle"
                + " "
                + system.e_ops_labels[idx]
                + " "
                + r"\rangle$"
            )
        else:
            label = r"$\mathrm{Re}\langle \mu \rangle$"
        plt.plot(times_plot, data, color=f"C{(idx+5)%10}", linestyle="solid")
        plt.axvline(times_0[0] + system.Delta_ts[0], color="C2", linestyle="dashed")
        plt.axvline(times_1[0] + system.Delta_ts[1], color="C3", linestyle="dashdot")
        plt.axvline(times_2[0] + system.Delta_ts[2], color="C4", linestyle="dotted")
        plt.ylabel(label)
        # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.xlabel(r"$t\,/\,\mathrm{fs}$")
    plt.suptitle(
        rf"$\tau = {tau_coh:.2f}\,\mathrm{{fs}},\quad T = {T_wait:.2f}\,\mathrm{{fs}},\quad \mathrm{{Solver}}$: {system.ODE_Solver}"
    )
    plt.tight_layout()
    plt.show()


def extend_time_tau_axes(
    ts: np.ndarray,
    taus: np.ndarray,
    data: np.ndarray,
    pad_rows: tuple[int, int] = (0, 0),
    pad_cols: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extend the ts and taus axes and pad the data array accordingly.

    Parameters:
        ts (np.ndarray): Time axis (t).
        taus (np.ndarray): Tau axis (coherence time).
        data (np.ndarray): 2D data array.
        pad_rows (tuple): Padding for rows (before, after) along taus axis.
        pad_cols (tuple): Padding for columns (before, after) along ts axis.

    Returns:
        tuple: (extended_ts, extended_taus, padded_data)
    """
    # print(f"Pad rows: {pad_rows}, Pad cols: {pad_cols}", data.shape, flush=True)
    # Pad the data array
    padded_data = np.pad(data, (pad_rows, pad_cols), mode="constant", constant_values=0)

    # Compute steps
    dt = ts[1] - ts[0]
    dtau = taus[1] - taus[0]

    # Extend axes
    extended_ts = np.linspace(
        ts[0] - pad_cols[0] * dt, ts[-1] + pad_cols[1] * dt, padded_data.shape[1]
    )
    extended_taus = np.linspace(
        taus[0] - pad_rows[0] * dtau,
        taus[-1] + pad_rows[1] * dtau,
        padded_data.shape[0],
    )

    return extended_ts, extended_taus, padded_data


def compute_2d_fft_wavenumber(ts, taus, data):
    """
    Compute the 2D FFT of the data and convert axes to wavenumber units.

    Parameters:
        ts (np.ndarray): Time axis for detection.
        taus (np.ndarray): Time axis for coherence.
        data (np.ndarray): 2D data array.

    Returns:
        tuple: (nu_ts, nu_taus, s2d) where
            nu_ts (np.ndarray): Wavenumber axis for detection.
            nu_taus (np.ndarray): Wavenumber axis for coherence.
            s2d (np.ndarray): 2D FFT of the input data.
    """
    # Calculate frequency axes (cycle/fs)
    taufreqs = np.fft.fftshift(np.fft.fftfreq(len(taus), d=(taus[1] - taus[0])))
    tfreqs = np.fft.fftshift(np.fft.fftfreq(len(ts), d=(ts[1] - ts[0])))

    # Convert to wavenumber units [10^4 cm⁻¹]
    nu_taus = taufreqs / 2.998 * 10
    nu_ts = tfreqs / 2.998 * 10

    # 2D FFT: first over tau (axis=1), then over t (axis=0), take imaginary part

    if np.any(np.imag(data)):
        data_for_fft = np.imag(data)
    else:
        data_for_fft = np.real(data)

    # 2D FFT: first over tau (axis=1), then over t (axis=0)
    s2d = np.fft.fftshift(np.fft.fft(np.fft.fft(data_for_fft, axis=1), axis=0))

    return nu_ts, nu_taus, s2d


def apply_RWA_phase_factors(
    rho: Qobj, t: float, omega: float, system: SystemParameters
) -> Qobj:
    """
    Apply time-dependent phase factors to the density matrix entries.
    Dispatches to the appropriate implementation based on N_atoms.

    Parameters:
        rho (Qobj): Density matrix (Qobj) to modify.
        t (float): Current time.
        omega (float): Frequency of the phase factor.
        system (SystemParameters): System parameters.

    Returns:
        Qobj: Modified density matrix with phase factors applied.
    """
    if system.N_atoms == 1:
        return _apply_RWA_phase_factors_1atom(rho, t, omega)
    elif system.N_atoms == 2:
        return _apply_RWA_phase_factors_2atom(rho, t, omega)
    else:
        raise ValueError("Only N_atoms=1 or 2 are supported.")


def _apply_RWA_phase_factors_1atom(rho: Qobj, t: float, omega: float) -> Qobj:
    """
    Apply time-dependent phase factors to the density matrix entries.

    Parameters:
        rho (Qobj): Density matrix (Qobj) to modify.
        omega (float): Frequency of the phase factor.
        t (float): Current time.

    Returns:
        Qobj: Modified density matrix with phase factors applied.
    """
    # Extract the density matrix as a NumPy array
    rho_array = rho.full()
    # print(rho.isherm)

    # Apply the phase factors to the specified elements
    phase_1 = np.exp(-1j * omega * t)  # e^(-i * omega * t)

    # Modify the elements
    rho_array[1, 0] *= phase_1  # rho_alpha_0 = sigma_alpha_0 * e^(-i * omega * t)
    rho_array[0, 1] *= np.conj(phase_1)
    rho_result = Qobj(rho_array, dims=rho.dims)
    # print(rho_array[0, 1], rho_array[1,0])

    # assert rho_result.isherm, "The resulting density matrix is not Hermitian."

    return rho_result


def _apply_RWA_phase_factors_2atom(rho: Qobj, t: float, omega: float) -> Qobj:
    """
    Apply time-dependent phase factors to the density matrix entries.

    Parameters:
        rho (Qobj): Density matrix (Qobj) to modify.
        omega (float): Frequency of the phase factor.
        t (float): Current time.

    Returns:
        Qobj: Modified density matrix with phase factors applied.
    """
    # Extract the density matrix as a NumPy array
    rho_array = rho.full()
    # print(rho.isherm)

    # Apply the phase factors to the specified elements
    phase_1 = np.exp(-1j * omega * t)  # e^(-i * omega * t)
    phase_2 = np.exp(-1j * 2 * omega * t)  # e^(-i * 2 * omega * t)

    # Modify the elements
    bar_alpha = 3
    for alpha in range(1, 3):
        rho_array[
            alpha, 0
        ] *= phase_1  # rho_alpha_0 = sigma_alpha_0 * e^(-i * omega * t)
        rho_array[0, alpha] *= np.conj(phase_1)

        rho_array[
            bar_alpha, alpha
        ] *= phase_1  # rho_bar_alpha_alpha = sigma_bar_alpha_alpha * e^(-i * omega * t)
        rho_array[alpha, bar_alpha] *= np.conj(phase_1)

    rho_array[
        bar_alpha, 0
    ] *= phase_2  # rho_bar_alpha_0 = sigma_bar_alpha_0 * e^(-i * 2 * omega * t)
    rho_array[0, bar_alpha] *= np.conj(phase_2)

    rho_result = Qobj(rho_array, dims=rho.dims)
    # print(rho_array[0, 1], rho_array[1,0])

    # assert rho_result.isherm, "The resulting density matrix is not Hermitian."

    return rho_result


# =============================
# "Paper_eqs" OWN ODE SOLVER
# =============================
def matrix_ODE_paper(
    t: float, pulse_seq: PulseSequence, system: SystemParameters
) -> Qobj:
    """Dispatches to the appropriate implementation based on N_atoms."""
    if system.N_atoms == 1:
        return _matrix_ODE_paper_1atom(t, pulse_seq, system)
    elif system.N_atoms == 2:
        return _matrix_ODE_paper_2atom(t, pulse_seq, system)
    else:
        raise ValueError("Only N_atoms=1 or 2 are supported.")


def _matrix_ODE_paper_1atom(
    t: float, pulse_seq: PulseSequence, system: SystemParameters
) -> Qobj:
    """
    Constructs the matrix L(t) for the equation drho_dt = L(t) * rho,
    where rho is the flattened density matrix. Uses gamma values from the provided system.

    Parameters:
        t (float): Time at which to evaluate the matrix.
        pulse_seq (PulseSequence): PulseSequence object for the electric field.
        system (SystemParameters): System parameters containing Gamma, gamma_0, and mu_eg.

    Returns:
        Qobj: Liouvillian matrix as a Qobj.
    """
    # Calculate the electric field using the pulse sequence
    Et = E_pulse(t, pulse_seq)
    Et_conj = np.conj(Et)

    L = np.zeros((4, 4), dtype=complex)

    # Indices for the flattened density matrix:
    # 0: rho_gg, 1: rho_ge, 2: rho_eg, 3: rho_ee

    # --- d/dt rho_ee ---
    L[3, 3] = -system.gamma_0
    L[3, 1] = 1j * Et * system.mu_A
    L[3, 2] = -1j * Et_conj * system.mu_A

    # --- d/dt rho_gg ---
    # L[0, 1] = -1j * Et * system.mu_A
    # L[0, 2] = 1j * Et_conj * system.mu_A
    L[0, :] += -1 * np.sum(L[[3], :], axis=0)  # Enforce trace conservation

    # --- d/dt rho_eg --- and  --- d/dt rho_ge ---
    L[2, 0] = 1j * Et * system.mu_A
    L[2, 3] = -1j * Et * system.mu_A

    L[1, :] = np.conj(L[2, :])

    L[2, 2] = -system.Gamma  # Decay term for coherence
    L[1, 1] = -system.Gamma  # Decay term for coherence

    return Qobj(L, dims=[[[2], [2]], [[2], [2]]])


def _matrix_ODE_paper_2atom(
    t: float, pulse_seq: PulseSequence, system: SystemParameters
) -> Qobj:
    """including RWA.
    Constructs the matrix L(t) for the equation drho_dt = L(t) * rho,
    where rho is the flattened density matrix.
    """
    Et = E_pulse(t, pulse_seq)
    Et_conj = np.conj(Et)

    L = np.zeros((16, 16), dtype=complex)

    # Indices for the flattened density matrix:
    # 0: rho00, 1: rho01, 2: rho02, 3: rho03
    # 4: rho10, 5: rho11, 6: rho12, 7: rho13
    # 8: rho20, 9: rho21, 10: rho22, 11: rho23
    # 12: rho30, 13: rho31, 14: rho32, 15: rho33

    # --- d/dt rho_10 ---
    term = -1j * (system.omega_ij(1, 0) - system.omega_laser) - system.Gamma_big_ij(
        1, 0
    )
    L[4, 4] = term  # ρ₁₀ ← ρ₁₀
    L[4, 0] = 1j * Et * system.Dip_op[1, 0]  # ρ₁₀ ← ρ₀₀
    L[4, 5] = -1j * Et * system.Dip_op[1, 0]  # ρ₁₀ ← ρ₁₁
    L[4, 6] = -1j * Et * system.Dip_op[2, 0]  # ρ₁₀ ← ρ₁₂
    L[4, 12] = 1j * Et_conj * system.Dip_op[3, 1]  # ρ₁₀ ← ρ₃₀

    # --- d/dt rho_01 ---
    L[1, 1] = np.conj(term)  # ρ₀₁ ← ρ₀₁
    L[1, 0] = -1j * Et_conj * system.Dip_op[1, 0]  # ρ₀₁ ← ρ₀₀
    L[1, 5] = 1j * Et_conj * system.Dip_op[1, 0]  # ρ₀₁ ← ρ₁₁
    L[1, 9] = 1j * Et_conj * system.Dip_op[2, 0]  # ρ₀₁ ← ρ₂₁
    L[1, 3] = -1j * Et * system.Dip_op[3, 1]  # ρ₀₁ ← ρ₀₃

    # --- d/dt rho_20 ---
    term = -1j * (system.omega_ij(2, 0) - system.omega_laser) - system.Gamma_big_ij(
        2, 0
    )
    L[8, 8] = term  # ρ₂₀ ← ρ₂₀
    L[8, 0] = 1j * Et * system.Dip_op[2, 0]  # ρ₂₀ ← ρ₀₀
    L[8, 10] = -1j * Et * system.Dip_op[2, 0]  # ρ₂₀ ← ρ₂₂
    L[8, 9] = -1j * Et * system.Dip_op[1, 0]  # ρ₂₀ ← ρ₂₁
    L[8, 12] = 1j * Et_conj * system.Dip_op[3, 2]  # ρ₂₀ ← ρ₃₀

    # --- d/dt rho_02 ---
    L[2, 2] = np.conj(term)  # ρ₀₂ ← ρ₀₂
    L[2, 0] = -1j * Et_conj * system.Dip_op[2, 0]  # ρ₀₂ ← ρ₀₀
    L[2, 10] = 1j * Et_conj * system.Dip_op[2, 0]  # ρ₀₂ ← ρ₂₂
    L[2, 6] = 1j * Et_conj * system.Dip_op[1, 0]  # ρ₀₂ ← ρ₁₂
    L[2, 3] = -1j * Et * system.Dip_op[3, 2]  # ρ₀₂ ← ρ₀₃

    # --- d/dt rho_30 ---
    term = -1j * (system.omega_ij(3, 0) - 2 * system.omega_laser) - system.Gamma_big_ij(
        3, 0
    )
    L[12, 12] = term  # ρ₃₀ ← ρ₃₀
    L[12, 4] = 1j * Et * system.Dip_op[3, 1]  # ρ₃₀ ← ρ₁₀
    L[12, 8] = 1j * Et * system.Dip_op[3, 2]  # ρ₃₀ ← ρ₂₀
    L[12, 13] = -1j * Et * system.Dip_op[1, 0]  # ρ₃₀ ← ρ₃₁
    L[12, 14] = -1j * Et * system.Dip_op[2, 0]  # ρ₃₀ ← ρ₃₂

    # --- d/dt rho_03 ---
    L[3, 3] = np.conj(term)  # ρ₀₃ ← ρ₀₃
    L[3, 1] = -1j * Et_conj * system.Dip_op[3, 1]  # ρ₀₃ ← ρ₀₁
    L[3, 2] = -1j * Et_conj * system.Dip_op[3, 2]  # ρ₀₃ ← ρ₀₂
    L[3, 7] = 1j * Et_conj * system.Dip_op[1, 0]  # ρ₀₃ ← ρ₁₃
    L[3, 11] = 1j * Et_conj * system.Dip_op[2, 0]  # ρ₀₃ ← ρ₂₃

    # --- d/dt rho_12 ---
    term = -1j * system.omega_ij(1, 2) - system.Gamma_big_ij(1, 2)
    L[6, 6] = term  # ρ₁₂ ← ρ₁₂
    L[6, 2] = 1j * Et * system.Dip_op[1, 0]  # ρ₁₂ ← ρ₀₂
    L[6, 7] = -1j * Et * system.Dip_op[3, 2]  # ρ₁₂ ← ρ₁₃
    L[6, 14] = 1j * Et_conj * system.Dip_op[3, 1]  # ρ₁₂ ← ρ₃₂
    L[6, 4] = -1j * Et_conj * system.Dip_op[2, 0]  # ρ₁₂ ← ρ₁₀

    # --- d/dt rho_21 ---
    L[9, 9] = np.conj(term)  # ρ₂₁ ← ρ₂₁
    L[9, 8] = -1j * Et_conj * system.Dip_op[1, 0]  # ρ₂₁ ← ρ₂₀
    L[9, 13] = 1j * Et_conj * system.Dip_op[3, 2]  # ρ₂₁ ← ρ₃₁
    L[9, 9] = -1j * Et * system.Dip_op[3, 1]  # ρ₂₁ ← ρ₂₃
    L[9, 1] = 1j * Et * system.Dip_op[2, 0]  # ρ₂₁ ← ρ₀₁

    # --- d/dt rho_31 ---
    term = -1j * (system.omega_ij(3, 1) - system.omega_laser) - system.Gamma_big_ij(
        3, 1
    )
    L[13, 13] = term  # ρ₃₁ ← ρ₃₁
    L[13, 5] = 1j * Et * system.Dip_op[3, 1]  # ρ₃₁ ← ρ₁₁
    L[13, 9] = 1j * Et * system.Dip_op[3, 2]  # ρ₃₁ ← ρ₂₁
    L[13, 12] = -1j * Et_conj * system.Dip_op[1, 0]  # ρ₃₁ ← ρ₃₀

    # --- d/dt rho_13 ---
    L[7, 7] = np.conj(term)  # ρ₁₃ ← ρ₁₃
    L[7, 5] = -1j * Et_conj * system.Dip_op[3, 1]  # ρ₁₃ ← ρ₁₁
    L[7, 6] = -1j * Et_conj * system.Dip_op[3, 2]  # ρ₁₃ ← ρ₁₂
    L[7, 3] = 1j * Et * system.Dip_op[1, 0]  # ρ₁₃ ← ρ₀₃

    # --- d/dt rho_32 ---
    term = -1j * (system.omega_ij(3, 2) - system.omega_laser) - system.Gamma_big_ij(
        3, 2
    )
    L[14, 14] = term  # ρ₃₂ ← ρ₃₂
    L[14, 10] = 1j * Et * system.Dip_op[3, 2]  # ρ₃₂ ← ρ₂₂
    L[14, 6] = 1j * Et * system.Dip_op[3, 1]  # ρ₃₂ ← ρ₁₂
    L[14, 12] = -1j * Et_conj * system.Dip_op[2, 0]  # ρ₃₂ ← ρ₃₀

    # --- d/dt rho_23 ---
    L[11, 11] = np.conj(term)  # ρ₂₃ ← ρ₂₃
    L[11, 10] = -1j * Et * system.Dip_op[3, 2]  # ρ₂₃ ← ρ₂₂
    L[11, 9] = -1j * Et * system.Dip_op[3, 1]  # ρ₂₃ ← ρ₂₁
    L[11, 3] = 1j * Et_conj * system.Dip_op[2, 0]  # ρ₂₃ ← ρ₀₃

    ### Diagonals
    # --- d/dt rho_00 ---
    L[0, 1] = -1j * Et * system.Dip_op[1, 0]
    L[0, 2] = -1j * Et * system.Dip_op[2, 0]
    L[0, 4] = 1j * Et_conj * system.Dip_op[1, 0]
    L[0, 8] = 1j * Et_conj * system.Dip_op[2, 0]

    # --- d/dt rho_11 ---
    L[5, 5] = -1 * system.Gamma_big_ij(1, 1)
    L[5, 10] = system.gamma_small_ij(1, 2)
    L[5, 1] = 1j * Et * system.Dip_op[1, 0]
    L[5, 7] = -1j * Et * system.Dip_op[3, 1]
    L[5, 13] = 1j * Et_conj * system.Dip_op[3, 1]
    L[5, 4] = -1j * Et_conj * system.Dip_op[1, 0]

    # --- d/dt rho_22 ---
    L[10, 10] = -1 * system.Gamma_big_ij(2, 2)
    L[10, 5] = system.gamma_small_ij(2, 1)
    L[10, 2] = 1j * Et * system.Dip_op[2, 0]
    L[10, 11] = -1j * Et * system.Dip_op[3, 2]
    L[10, 14] = 1j * Et_conj * system.Dip_op[3, 2]
    L[10, 8] = -1j * Et_conj * system.Dip_op[2, 0]

    # --- d/dt rho_00 --- and  --- d/dt rho_33 (sum d/dt rho_ii = 0) (trace condition) ---
    L[15, :] = -1 * np.sum(
        L[[0, 5, 10], :], axis=0
    )  # TODO not mentioned in paper, i will assume it to conserve the trace
    # print("the trace d/dt (rho_00 + rho_11 + rho_22 + rho_33) = ", np.sum(L[[0, 5, 10, 15], :]), "should be 0")

    return Qobj(L, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]])


# only use the Redfield tensor as a matrix:
def R_paper(system: SystemParameters) -> Qobj:
    """Dispatches to the appropriate implementation based on N_atoms."""
    if system.N_atoms == 1:
        return _R_paper_1atom(system)
    elif system.N_atoms == 2:
        return _R_paper_2atom(system)
    else:
        raise ValueError("Only N_atoms=1 or 2 are supported.")


def _R_paper_1atom(system: SystemParameters) -> Qobj:
    """
    Constructs the Redfield Tensor R for the equation drho_dt = -i(Hrho - rho H) + R * rho,
    where rho is the flattened density matrix. Uses gamma values from the provided system.

    Parameters:
        system (SystemParameters): System parameters containing Gamma and gamma_0.

    Returns:
        Qobj: Redfield tensor as a Qobj.
    """
    R = np.zeros((4, 4), dtype=complex)  # Redfield tensor initialization

    # --- d/dt rho_eg ---
    R[2, 2] = -system.Gamma  # Decay term for coherence
    # --- d/dt rho_ge ---
    R[1, 1] = -system.Gamma

    # --- d/dt rho_ee ---
    R[3, 3] = -system.gamma_0  # Decay term for population
    # --- d/dt rho_gg ---
    R[0, 3] = system.gamma_0  # Ensures trace conservation

    return Qobj(R, dims=[[[2], [2]], [[2], [2]]])


def _R_paper_2atom(system: SystemParameters) -> Qobj:
    """
    including RWA
    Constructs the Redfield Tensor R for the equation drho_dt = -i(Hrho - rho H) + R * rho,
    where rho is the flattened density matrix.
    """
    R = np.zeros((16, 16), dtype=complex)

    # Indices for the flattened density matrix:
    # 0: rho00, 1: rho01, 2: rho02, 3: rho03
    # 4: rho10, 5: rho11, 6: rho12, 7: rho13
    # 8: rho20, 9: rho21, 10: rho22, 11: rho23
    # 12: rho30, 13: rho31, 14: rho32, 15: rho33

    # --- d/dt rho_10 ---
    term = -1j * (system.omega_ij(1, 0) - system.omega_laser) - system.Gamma_big_ij(
        1, 0
    )
    R[4, 4] = term

    # --- d/dt rho_01 ---
    R[1, 1] = np.conj(term)

    # --- d/dt rho_20 --- = ANSATZ = (d/dt s_20 - i omega_laser s_20) e^(-i omega_laser t)
    term = -1j * (system.omega_ij(2, 0) - system.omega_laser) - system.Gamma_big_ij(
        2, 0
    )
    R[8, 8] = term

    # --- d/dt rho_02 ---
    R[2, 2] = np.conj(term)

    # --- d/dt rho_30 ---
    term = -1j * (system.omega_ij(3, 0) - 2 * system.omega_laser) - system.Gamma_big_ij(
        3, 0
    )
    R[12, 12] = term

    # --- d/dt rho_03 ---
    R[3, 3] = np.conj(term)

    # --- d/dt rho_12 ---
    term = -1j * system.omega_ij(1, 2) - system.Gamma_big_ij(1, 2)
    R[6, 6] = term

    # --- d/dt rho_21 ---
    R[9, 9] = np.conj(term)

    # --- d/dt rho_31 ---
    term = -1j * (system.omega_ij(3, 1) - system.omega_laser) - system.Gamma_big_ij(
        3, 1
    )
    R[13, 13] = term

    # --- d/dt rho_13 ---
    R[7, 7] = np.conj(term)

    # --- d/dt rho_32 ---
    term = -1j * (system.omega_ij(3, 2) - system.omega_laser) - system.Gamma_big_ij(
        3, 2
    )
    R[14, 14] = term

    # --- d/dt rho_23 ---
    R[11, 11] = np.conj(term)

    ### Diagonals
    # --- d/dt rho_11 ---
    R[5, 5] = -system.Gamma_big_ij(1, 1)
    R[5, 10] = system.gamma_small_ij(1, 2)

    # --- d/dt rho_22 ---
    R[10, 10] = -system.Gamma_big_ij(2, 2)
    R[10, 5] = system.gamma_small_ij(2, 1)

    # NOW THERE IS NO POPULATION CHANGE in 3 || 1 goes to 2 and vice versa
    # --- d/dt rho_00 --- and  --- d/dt rho_33 (sum d/dt rho_ii = 0) (trace condition) ---
    # R[15, :] = -1 * np.sum(R[[0, 5, 10], :], axis=0)
    # R[0, :] = -1 * np.sum(R[[5, 10, 15], :], axis=0) # i think the ground state should get repopulated

    return Qobj(R, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]])


def compute_pulse_evolution(
    psi_ini: Qobj,
    times: np.ndarray,
    pulse_seq: PulseSequence,
    system: SystemParameters = None,
) -> qutip.Result:
    """
    Compute the evolution of the system for a given pulse sequence.

    Parameters:
        psi_ini (Qobj): Initial quantum state.
        times (np.ndarray): Time array for the evolution.
        pulse_seq (PulseSequence): PulseSequence object.
        system (SystemParameters): System parameters.

    Returns:
        qutip.Result: Result of the evolution.
    """
    # =============================
    # Use global parameters if not provided
    # =============================
    if system is None:
        raise ValueError("System parameters must be provided.")

    # Set solver options
    # progress_bar = "enhanced" if preprev != None and times[0] >= times[len(times) // 2] else ""
    options = {
        "store_states": True,
        "progress_bar": "",  # progress_bar,
        #   "nsteps": 10000,  # Increase max number of steps per integration interval
    }

    # =============================
    # Choose solver and compute the evolution
    # =============================
    if system.ODE_Solver not in ["ME", "BR", "Paper_eqs", "Paper_BR"]:
        raise ValueError(f"Unknown ODE solver: {system.ODE_Solver}")

    if system.ODE_Solver == "Paper_eqs":
        assert (
            system.RWA_laser == True
        ), "The equations of the paper only make sense with RWA"
        # You need to adapt Liouville to accept pulse_seq and system if needed
        Liouville = QobjEvo(lambda t, args=None: matrix_ODE_paper(t, pulse_seq, system))
        result = mesolve(
            Liouville,
            psi_ini,
            times,
            options=options,
        )
    else:
        # Build Hamiltonian
        H_free = system.H0_diagonalized  # already includes the RWA, if present!
        H_int_evo = H_free + QobjEvo(lambda t, args=None: H_int(t, pulse_seq, system))
        c_ops = []
        if system.ODE_Solver == "Paper_BR":
            c_ops = [R_paper(system)]

            result = mesolve(
                H_int_evo,
                psi_ini,
                times,
                c_ops=c_ops,
                options=options,
            )

        elif system.ODE_Solver == "ME":
            c_ops = system.c_ops_list

            result = mesolve(
                H_int_evo,
                psi_ini,
                times,
                c_ops=c_ops,
                options=options,
            )

        elif system.ODE_Solver == "BR":
            result = brmesolve(
                H_int_evo,
                psi_ini,
                times,
                a_ops=system.a_ops_list,
                options=options,
            )

        """
        # =============================
        # Split the evolution into two parts:
        # 1. With H_int for the pulse duration
        # 2. With H0 for the rest
        # =============================
        # Find the index where times = times[0] + 2*Delta_ts[0]
        last_pulse = pulse_seq.pulses[0]# MIGHT be wrong? -> -1
        t_pulse_end = last_pulse.pulse_start_time + 2 * last_pulse.pulse_half_width
        idx_split   = np.abs(times - t_pulse_end).argmin()

        times1 = times[:idx_split]
        if times1.size == 0:
            times1 = [last_pulse.pulse_start_time]
            

        result1 = mesolve(
            H_int_evo,
            psi_ini,
            times1,
            c_ops=[R_paper(system)],
            options=options,
        )

        # --- Second part: with H0 only ---
        # Use the last state as initial state for the second part
        psi_after_pulse = result1.states[-1]
        times2 = times[idx_split:]
        if len(times2) > 0:
            result2 = mesolve(
                H_free,
                psi_after_pulse,
                times2,
                c_ops=c_ops,
                options=options,
            )
            # Combine results
            all_states = list(result1.states) + list(result2.states)
            all_times = list(result1.times) + list(result2.times)
            options_full = options.copy()
            if "store_final_state" not in options_full:
                options_full["store_final_state"] = False
            if "store_states" not in options_full:
                options_full["store_states"] = True
            result = Result(e_ops=[], options=options_full)
            result.states = all_states
            result.times = all_times
            # Copy other attributes if needed
        else:
            result = result1
        """

    return result


def get_expect_vals_with_RWA(
    states: list[qutip.Qobj], times: np.array, system: SystemParameters
):
    """
    Calculate the expectation values in the result with RWA phase factors.

    Parameters:
        states= data.states (where data = qutip.Result): Results of the pulse evolution.
        times (list): Time points at which the expectation values are calculated.
        e_ops (list): the operators for which the expectation values are calculated
        omega (float): omega_laser (float): Frequency of the laser.
        RWA (bool): Whether to apply the RWA phase factors.
    Returns:
        list of lists: Expectation values for each operator of len(states).
    """
    omega = system.omega_laser
    e_ops = system.e_ops_list + [system.Dip_op]

    if system.RWA_laser:
        # Apply RWA phase factors to each state
        states = [
            apply_RWA_phase_factors(state, time, omega, system)
            for state, time in zip(states, times)
        ]
    updated_expects = [np.real(expect(states, e_op)) for e_op in e_ops]
    return updated_expects


# ##########################
# independent of system
# ##########################
def check_the_solver(
    times: np.ndarray, system: SystemParameters
) -> tuple[qutip.Result, float]:
    """
    Checks the solver within the compute_pulse_evolution function
    with the provided psi_ini, times, and system.

    Parameters:
        times (np.ndarray): Time array for the evolution.
        system (System): System object containing all relevant parameters, including e_ops_list.
        PulseSequence (type): The PulseSequence class to construct pulse sequences.

    Returns:
        qutip.Result: The result object from compute_pulse_evolution.
    """
    print(f"Checking '{system.ODE_Solver}' solver ")

    # =============================
    # INPUT VALIDATION
    # =============================
    if not hasattr(system, "ODE_Solver"):
        raise AttributeError("system must have attribute 'ODE_Solver'")
    if not hasattr(system, "e_ops_list"):
        raise AttributeError("system must have attribute 'e_ops_list'")
    if not isinstance(system.psi_ini, qutip.Qobj):
        raise TypeError("psi_ini must be a qutip.Qobj")
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be a numpy.ndarray")
    if not isinstance(system.e_ops_list, list) or not all(
        isinstance(op, qutip.Qobj) for op in system.e_ops_list
    ):
        raise TypeError("system.e_ops_list must be a list of qutip.Qobj")
    if len(times) < 2:
        raise ValueError("times must have at least two elements")

    # =============================
    # CONSTRUCT PULSE SEQUENCE (refactored)
    # =============================

    # Define pulse parameters
    phi_0 = np.pi / 2
    phi_1 = np.pi / 4
    phi_2 = 0
    t_start_pulse0 = times[0]
    t_start_pulse1 = times[-1] / 2
    t_start_2 = times[-1] / 1.1

    # Use the from_args static method to construct the sequence
    pulse_seq = PulseSequence.from_args(
        system=system,
        curr=(t_start_2, phi_2),
        prev=(t_start_pulse1, phi_1),
        preprev=(t_start_pulse0, phi_0),
    )
    result = compute_pulse_evolution(system.psi_ini, times, pulse_seq, system=system)
    # =============================
    # CHECK THE RESULT
    # =============================
    if not isinstance(result, qutip.Result):
        raise TypeError("Result must be a qutip.Result object")
    if list(result.times) != list(times):
        raise ValueError("Result times do not match input times")
    if len(result.states) != len(times):
        raise ValueError("Number of output states does not match number of time points")

    # =============================
    # CHECK DENSITY MATRIX PROPERTIES
    # =============================
    strg = ""
    global time_cut
    omega = system.omega_laser
    time_cut = np.inf  # time after which the checks failed
    for index, state in enumerate(result.states):
        # Apply RWA phase factors if needed
        if getattr(system, "RWA_laser", False):
            state = apply_RWA_phase_factors(state, times[index], omega, system)
        time = times[index]
        if not state.isherm:
            strg += f"Density matrix is not Hermitian after t = {time}.\n"
            print(state)
        eigvals = state.eigenenergies()
        if not np.all(
            eigvals >= -1e-3
        ):  # allow for small numerical negative eigenvalues
            strg += f"Density matrix is not positive semidefinite after t = {time}: The lowest eigenvalue is {eigvals.min()}.\n"
            time_cut = time
        if not np.isclose(state.tr(), 1.0):
            strg += f"Density matrix is not trace-preserving after t = {time}: The trace is {state.tr()}.\n"
            time_cut = time
        if strg:
            strg += "Adjust your parameters!"
            print(strg)
            break
    else:
        print(
            "Checks passed. Solver appears to be called correctly, and density matrix remains Hermitian and positive."
        )

    return result, time_cut


def compute_two_dimensional_polarization(
    T_wait: float,
    phi_0: float,
    phi_1: float,
    times: np.ndarray,
    system: SystemParameters,
    **kwargs: dict,
):
    """
    Compute the two-dimensional polarization for a given waiting time (T_wait) and
    the phases of the first and second pulses (phi_0, phi_1).

    Parameters:
        T_wait (float): Waiting time between the second and third pulses.
        phi_0 (float): Phase of the first pulse.
        phi_1 (float): Phase of the second pulse.
        times (np.ndarray): Time array.
        system: System object containing all relevant parameters.
        **kwargs: Additional keyword arguments.
                  Can include 'plot_example' (bool, optional): Whether to plot an example evolution.

    Returns:
        tuple: (t_det_vals, tau_coh_vals, data)
    """

    plot_example = kwargs.get("plot_example", False)

    # get the symmetric times, tau_coh, t_det
    tau_coh_vals, t_det_vals = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait=T_wait)

    # initialize the time domain Spectroscopy data tr(Dip_op * rho_final(tau_coh, t_det))
    data = np.zeros((len(tau_coh_vals), len(t_det_vals)), dtype=np.complex64)

    idx_end_pulse0 = 0
    t_start_pulse0 = times[idx_end_pulse0]
    idx_end_pulse0 = np.abs(times - (system.Delta_ts[0])).argmin()
    idx_start_pulse1_max = np.abs(
        times - (tau_coh_vals[-1] - system.Delta_ts[1])
    ).argmin()
    times_0 = times[: idx_start_pulse1_max + 1]
    if times_0.size == 0:
        times_0 = times[idx_end_pulse0 : idx_end_pulse0 + 1]

    # First pulse
    pulse_0 = (t_start_pulse0, phi_0)
    # Instead of directly constructing PulseSequence, use from_args:
    pulse_seq_0 = PulseSequence.from_args(
        system=system,
        curr=pulse_0,
    )
    data_0 = compute_pulse_evolution(
        system.psi_ini, times_0, pulse_seq_0, system=system
    )

    for tau_idx, tau_coh in enumerate(tau_coh_vals):
        idx_start_pulse1 = np.abs(times - (tau_coh - system.Delta_ts[1])).argmin()
        t_start_pulse1 = times[idx_start_pulse1]
        idx_end_pulse1 = np.abs(times - (tau_coh + system.Delta_ts[1])).argmin()
        rho_1 = data_0.states[idx_start_pulse1]

        idx_start_2 = np.abs(times - (tau_coh + T_wait - system.Delta_ts[2])).argmin()
        idx_end_pulse2 = np.abs(
            times - (tau_coh + T_wait + system.Delta_ts[2])
        ).argmin()
        t_start_2 = times[idx_start_2]

        times_1 = times[idx_start_pulse1 : idx_start_2 + 1]
        if times_1.size == 0:
            times_1 = times[idx_start_pulse1 : idx_end_pulse1 + 1]

        pulse_1 = (t_start_pulse1, phi_1)
        pulse_seq_1 = PulseSequence.from_args(
            system=system,
            curr=pulse_1,
            prev=pulse_0,
        )
        data_1 = compute_pulse_evolution(rho_1, times_1, pulse_seq_1, system=system)

        idx_start_2_in_times_1 = np.abs(times_1 - t_start_2).argmin()
        rho_2 = data_1.states[idx_start_2_in_times_1]

        times_2 = times[idx_start_2:]
        if times_2.size == 0:
            times_2 = times[idx_start_2 : idx_end_pulse2 + 1]

        phi_2 = 0
        pulse_f = (t_start_2, phi_2)
        pulse_seq_f = PulseSequence.from_args(
            system=system,
            curr=pulse_f,
            prev=pulse_1,
            preprev=pulse_0,
        )
        data_f = compute_pulse_evolution(rho_2, times_2, pulse_seq_f, system=system)

        for t_idx, t_det in enumerate(t_det_vals):
            actual_det_time = t_start_2 + system.Delta_ts[2] + t_det

            if actual_det_time < system.t_max and actual_det_time < time_cut:
                t_idx_in_times_2 = np.abs(times_2 - actual_det_time).argmin()

                rho_f = data_f.states[t_idx_in_times_2]
                if system.RWA_laser:
                    rho_f = apply_RWA_phase_factors(
                        rho_f,
                        times_2[t_idx_in_times_2],
                        omega=system.omega_laser,
                        system=system,
                    )
                value = expect(system.Dip_op, rho_f)
                data[tau_idx, t_idx] = np.real(value)

                if t_idx == 0 and tau_idx == len(tau_coh_vals) // 3 and plot_example:
                    print(system.RWA_laser)
                    data_1_expects = get_expect_vals_with_RWA(
                        data_0.states[: idx_start_pulse1 + 1],
                        data_0.times[: idx_start_pulse1 + 1],
                        system,
                    )
                    data_2_expects = get_expect_vals_with_RWA(
                        data_1.states[: idx_start_2_in_times_1 + 1],
                        data_1.times[: idx_start_2_in_times_1 + 1],
                        system,
                    )
                    data_f_expects = get_expect_vals_with_RWA(
                        data_f.states, data_f.times, system
                    )
                    data_expectations = [
                        np.concatenate(
                            [
                                data_1_expects[idx],
                                data_2_expects[idx],
                                data_f_expects[idx],
                            ]
                        )
                        for idx in range(len(system.e_ops_list) + 1)
                    ]

                    Plot_example_evo(
                        times_0[: idx_start_pulse1 + 1],
                        times_1,
                        times_2,
                        data_expectations,
                        pulse_seq_f,
                        tau_coh,
                        T_wait,
                        system=system,
                    )

    return (
        t_det_vals,
        tau_coh_vals,
        1j * data,
    )  # because E ~ i*P # TODO: also export T_wait


# ##########################
# dependent of system
# ##########################
def sigma(E: np.ndarray, Delta: float, E0: float = 0.0) -> np.ndarray:
    """
    Compute the normalized Gaussian function σ(E - E0) with given FWHM.

    Parameters
    ----------
    E : np.ndarray
        Energy value(s) at which to evaluate σ(E - E0).
    Delta : float
        Full Width at Half Maximum (FWHM) of the Gaussian.
    E0 : float, optional
        Center energy (default: 0.0).

    Returns
    -------
    np.ndarray
        The value(s) of σ(E - E0) at E.

    Notes
    -----
    The function is normalized such that
        ∫σ(E - E0) dE = 1
    for all Delta.
    """
    # =============================
    # Compute normalized Gaussian
    # =============================
    ln2 = np.log(2)  # natural logarithm of 2
    sigma_val = Delta / (2 * np.sqrt(2 * ln2))  # standard deviation from FWHM
    norm = 1.0 / (sigma_val * np.sqrt(2 * np.pi))  # normalization factor
    exponent = -0.5 * ((E - E0) / sigma_val) ** 2  # Gaussian exponent

    return norm * np.exp(exponent)


def sample_from_sigma(
    N: int, Delta: float, E0: float, E_range: float = 10.0
) -> np.ndarray:
    """
    Sample N values from the normalized σ(E) distribution using rejection sampling.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    Delta : float
        Full Width at Half Maximum (FWHM) of the Gaussian.
    E0 : float
        Center energy of the distribution.
    E_range : float, optional
        Range (in units of Delta) to sample from around E0 (default: 10).

    Returns
    -------
    np.ndarray
        Array of sampled energy values.
    """
    # =============================
    # Define the sampling range and maximum
    # =============================
    E_min = E0 - E_range * Delta
    E_max = E0 + E_range * Delta
    E_vals = np.linspace(E_min, E_max, 10000)
    sigma_vals = sigma(E_vals, Delta, E0)
    sigma_max = np.max(sigma_vals)

    # =============================
    # Rejection sampling
    # =============================
    samples = []
    while len(samples) < N:
        E_try = np.random.uniform(E_min, E_max)
        y_try = np.random.uniform(0, sigma_max)
        if y_try < sigma(E_try, Delta, E0):
            samples.append(E_try)
    return np.array(samples)


def compute_many_polarizations(
    T_wait: float,
    phi_0: float,
    phi_1: float,
    N: int,
    E0: float,
    Delta: float,
    times: np.ndarray,
    system: SystemParameters,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calls compute_two_dimensional_polarization N times for different omega_ats sampled from σ.

    Parameters
    ----------
    T_wait : float
        Delay between pulses.
    phi_0 : float
        Phase of the first pulse.
    phi_1 : float
        Phase of the second pulse.
    N : int
        Number of simulations (samples).
    E0 : float
        Center of the frequency distribution.
    Delta : float
        FWHM for sigma distribution.
    times : np.ndarray
        Time grid for simulation.
    **kwargs : dict
        Additional arguments for compute_two_dimensional_polarization.

    Returns
    -------
    tuple
        (t_det_vals, tau_coh_vals, data_avg, omega_ats)
    """
    # =============================
    # Sample omega_ats from sigma distribution
    # =============================
    omega_ats = sample_from_sigma(N, Delta, E0)
    results = []

    # =============================
    # Run simulations for each sampled omega
    # =============================
    for omega in omega_ats:
        system_new = copy.deepcopy(system)  # create a copy of the system
        system_new.omega_A_cm = omega
        print("new omega_A", system_new.omega_A)
        data = compute_two_dimensional_polarization(
            T_wait=T_wait,
            phi_0=phi_0,
            phi_1=phi_1,
            times=times,
            system=system_new,
            **kwargs,
        )

        results.append(data[2])  # only store the data array

    data_avg = np.mean(results, axis=0)  # average over all samples
    t_det_vals = data[0]  # detection times from last run
    tau_coh_vals = data[1]  # coherence times from last run

    return t_det_vals, tau_coh_vals, data_avg, omega_ats


# ##########################
# functions for parallel processing
# ##########################
def for_one_time_calc_phase_comb(
    T_wait: float,
    phases: list,
    times: np.ndarray,
    system: SystemParameters,
    **kwargs: dict,
) -> np.ndarray:
    """
    Compute and average the 2D polarization for all phase combinations for one T_wait.

    Parameters
    ----------
    T_wait : float
        Waiting time.
    phases : list
        List of phase values.
    times : np.ndarray
        Time grid for computation.
    system : SystemParameters
        System parameters object.
    **kwargs: Additional keyword arguments.
                Can include 'plot_example' (bool, optional): Whether to plot an example evolution.

    Returns
    -------
    np.ndarray
        Averaged 2D polarization data for this T_wait.
    """
    results = []
    for phi1 in phases:
        for phi2 in phases:
            _, _, data = compute_two_dimensional_polarization(
                T_wait, phi1, phi2, times=times, system=system, **kwargs
            )
            results.append(data)
    averaged_data = np.mean(np.stack(results), axis=0)

    return averaged_data


def parallel_process_all_combinations(
    phases: list,
    times_T: np.ndarray,
    times: np.ndarray,
    system: SystemParameters,
    **kwargs: dict,
) -> list:
    """
    Compute the averaged 2D polarization for all T_wait in times_T using all phase combinations.

    Parameters
    ----------
    phases : list
        List of phase values.
    times_T : np.ndarray
        Array of T_wait values.
    times : np.ndarray
        Time grid for computation.
    system : SystemParameters
        System parameters object.
    **kwargs: Additional keyword arguments.
                Can include 'plot_example' (bool, optional): Whether to plot an example evolution.

    Returns
    -------
    list
        List of averaged 2D polarization arrays for each T_wait.
    """
    results = [None] * len(times_T)

    # =============================
    # Parallelize over T_waits using ThreadPoolExecutor
    # =============================

    with ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(
                for_one_time_calc_phase_comb, T_wait, phases, times, system, **kwargs
            ): idx
            for idx, T_wait in enumerate(times_T)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                averaged_data = future.result()
                # print(f" at index {idx} completed.", averaged_data.shape)
            except Exception as exc:
                # print(f"T_wait index {idx} generated an exception: {exc}")
                averaged_data = None
            results[idx] = averaged_data
            # print(len(results), results[idx].shape)
    return results


def parallel_process_all_omega_ats(
    omega_ats: list,
    phases: list,
    times_T: np.ndarray,
    times: np.ndarray,
    system: SystemParameters,
    **kwargs: dict,
) -> list:
    """
    Compute the averaged 2D polarization for all omega_ats and all T_wait in times_T using all phase combinations.

    Parameters
    ----------
    omega_ats : list or np.ndarray
        List of omega_A values to use.
    phases : list
        List of phase values.
    times_T : np.ndarray
        Array of T_wait values.
    times : np.ndarray
        Time grid for computation.
    system : SystemParameters
        System parameters object.
    **kwargs: Additional keyword arguments.
                Can include 'plot_example' (bool, optional): Whether to plot an example evolution.

    Returns
    -------
    list
        List of lists, each containing averaged 2D polarization arrays for each T_wait at each omega_at.
    """
    all_results = []
    for omega_at in omega_ats:
        system_new = copy.deepcopy(system)
        system_new.omega_A_cm = omega_at
        results = parallel_process_all_combinations(
            phases, times_T, times, system=system_new, **kwargs
        )
        all_results.append(results)

    # Average over omega_ats for each T_wait
    averaged_results = []
    for t_idx in range(len(times_T)):
        data_for_t = [all_results[o_idx][t_idx] for o_idx in range(len(omega_ats))]
        if not data_for_t:  # If all simulations failed for this T_wait
            averaged_results.append(None)
            continue
        averaged_data = np.mean(np.stack(data_for_t), axis=0)
        averaged_results.append(averaged_data)

    return averaged_results


def run_parallel_for_sampled_omegas(
    N: int,
    Delta: float,
    E0: float,
    phases: list,
    times_T: np.ndarray,
    times: np.ndarray,
    system: SystemParameters,
    E_range: float = 10.0,
    **kwargs: dict,
) -> list:
    """
    Sample omega_ats using sample_from_sigma and run parallel_process_all_omega_ats.

    Parameters
    ----------
    N : int
        Number of samples.
    Delta : float
        Energy splitting parameter.
    E0 : float
        Central energy.
    phases : list
        List of phase values.
    times_T : np.ndarray
        Array of T_wait values.
    times : np.ndarray
        Time grid for computation.
    system : SystemParameters
        System parameters object.
    E_range : float, optional
        Range for sampling. Default is 10.
    **kwargs: Additional keyword arguments.
                Can include 'plot_example' (bool, optional): Whether to plot an example evolution.

    Returns
    -------
    list
        Averaged 2D polarization arrays for each T_wait, averaged over sampled omegas.
    """
    omega_ats = sample_from_sigma(N, Delta, E0, E_range=E_range)
    averaged_results = parallel_process_all_omega_ats(
        omega_ats, phases, times_T, times, system=system, **kwargs
    )
    return averaged_results


def extend_and_plot_results(
    averaged_results: list[np.ndarray],
    times_T: np.ndarray,
    times: np.ndarray,
    extend_for: tuple[int, int] = None,
    **plot_args_freq: dict,
) -> None:
    """
    Extend and plot the results for a set of 2D spectra averaged over phase/inhomogeneous broadening.

    Parameters
    ----------
    averaged_results : list of np.ndarray
        List of 2D arrays (each shape: [len(taus), len(ts)]) for each T_wait.
    times_T : np.ndarray
        Array of T_wait values.
    times : np.ndarray
        Time grid used for simulation.
    extend_for : tuple[int, int], optional
        Padding for (rows, columns) as (before, after) for both axes.
    **plot_args_freq : dict
        Additional keyword arguments for frequency-domain plotting.

    Returns
    -------
    None
    """
    if not averaged_results:
        print("No results to plot")
        return

    global_ts, global_taus = get_tau_cohs_and_t_dets_for_T_wait(times, times_T[0])
    global_data_time = np.zeros((len(global_taus), len(global_ts)), dtype=np.complex64)

    if extend_for is not None:
        global_ts, global_taus, global_data_time = extend_time_tau_axes(
            global_ts,
            global_taus,
            global_data_time,
            pad_rows=extend_for,
            pad_cols=extend_for,
        )

    global_nu_ts, global_nu_taus, _ = compute_2d_fft_wavenumber(
        global_ts, global_taus, np.zeros((len(global_ts), len(global_taus)))
    )
    global_data_freq = np.zeros((len(global_taus), len(global_ts)), dtype=np.complex64)

    # =============================
    # Combine all data arrays into global arrays for time and frequency domains
    # =============================

    # Initialize global arrays with zeros
    # global_ts and global_taus are the largest axes (from the first T_wait)
    # global_data_time and global_data_freq are already initialized above

    def find_closest_index(local_vals, global_vals):
        """Find indices in global_vals closest to each value in local_vals."""
        return [np.argmin(np.abs(global_vals - v)) for v in local_vals]

    for i, data in enumerate(averaged_results):
        T_wait = times_T[i]
        ts, taus = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait)

        if extend_for is not None:
            ts, taus, data = extend_time_tau_axes(
                ts, taus, data, pad_rows=extend_for, pad_cols=extend_for
            )

        nu_ts, nu_taus, data_freq = compute_2d_fft_wavenumber(ts, taus, data)

        # Map local data into the global arrays
        tau_indices = find_closest_index(taus, global_taus)
        t_indices = find_closest_index(ts, global_ts)
        for local_tau_idx, global_tau_idx in enumerate(tau_indices):
            for local_t_idx, global_t_idx in enumerate(t_indices):
                # Check bounds before assignment to avoid IndexError
                if (
                    global_tau_idx < global_data_time.shape[0]
                    and global_t_idx < global_data_time.shape[1]
                ):
                    global_data_time[global_tau_idx, global_t_idx] += data[
                        local_tau_idx, local_t_idx
                    ]
                    global_data_freq[global_tau_idx, global_t_idx] += data_freq[
                        local_tau_idx, local_t_idx
                    ]
                else:
                    # Print debug info if index is out of bounds
                    print(
                        f"IndexError: global_tau_idx={global_tau_idx}, global_t_idx={global_t_idx}, shape={global_data_time.shape}"
                    )
        """plot_positive_color_map(
            (ts, taus, data), 
            times_T[i], 
            type="imag",
            use_custom_colormap=True,
            )

        plot_positive_color_map(
            (nu_ts, nu_taus, data_freq),
            times_T[i],
            **plot_args_freq
        )"""

    # Normalize by number of T_waits
    global_data_time /= len(averaged_results)
    global_data_freq /= len(averaged_results)

    # Plot the global results
    """
    plot_positive_color_map(
        (global_ts, global_taus, global_data_time),
        type="imag",
        use_custom_colormap=True,
    )
    """
    plot_positive_color_map(
        (global_nu_ts, global_nu_taus, global_data_freq),
        **plot_args_freq,
    )
