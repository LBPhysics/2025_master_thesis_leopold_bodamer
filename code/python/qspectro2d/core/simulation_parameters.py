# =============================
# DEFINE THE SIMULATION PARAMETERS CLASS
# =============================
from dataclasses import dataclass, asdict  # for the class definiton
import numpy as np
import qspectro2d
from qspectro2d.core.pulse_sequences import PulseSequence
from qspectro2d.core.system_parameters import SystemParameters
from qspectro2d.core.coupling_class import SystemBathCoupling, SystemLaserCoupling
from qutip import Qobj, ket2dm
from qspectro2d.core.pulse_functions import Epsilon_pulse, E_pulse


@dataclass
class SimulationConfig:
    # =============================
    # Solver and model control
    # =============================
    """
    "Paper_eqs" (solve the EOMs from the paper) or
    "Paper_BR" do d/dt rho = -i/hbar * [H0 - Dip * E, rho] + R(rho)
    "ME" (solve the master equation) or
    "BR" (solve the Bloch-Redfield equations)
    """
    ODE_Solver: str = "Paper_BR"
    RWA_SL: bool = True
    #  CAN ONLY HANDLE TRUE For Paper_eqs
    #  only valid for omega_laser ~ omega_A

    # Other Simulation parameters
    # times
    dt: float = 0.1  # time step in fs
    tau_coh: float = 100.0  # coherence time in fs
    t_wait: float = 0.0  # wait time before the first pulse in fs
    t_det_max: float = 100.0
    # phase cycling -> 16 parallel jobs
    n_phases: int = 4
    # inhomogeneous broadening
    n_freqs: int = 100
    Delta_cm: float = 300.0

    def summary(self) -> str:
        return (
            f"SimulationConfig Summary:\n"
            f"--------------------------\n"
            f"Solver Type     : {self.ODE_Solver}\n"
            f"Use RWA_SL      : {self.RWA_SL}\n\n"
            f"Time Step (dt)  : {self.dt} fs\n"
            f"Coherence Time  : {self.tau_coh} fs\n"
            f"Wait Time       : {self.t_wait} fs\n"
            f"Max Det. Time   : {self.t_det_max} fs\n\n"
            f"Phase Cycles    : {self.n_phases}\n"
            f"Inhom. Points   : {self.n_freqs}\n"
            f"Delta (cm⁻¹)     : {self.Delta_cm}\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        return cls(**data)

    def __str__(self) -> str:
        return self.summary()


@dataclass
class SimulationModel:
    simulation: SimulationConfig

    system: SystemParameters
    laser: PulseSequence
    SB_coupling: SystemBathCoupling
    SL_coupling: SystemLaserCoupling

    keep_track: str = "eigenstates"  # alternative "basis" determines the "observables"

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

        if self.simulation.RWA_SL:
            if self.system.N_atoms == 1:
                Es[1] -= self.laser.omega_laser

            elif self.system.N_atoms == 2:
                Es[1] -= self.laser.omega.omega_laser
                Es[2] -= self.laser.omega.omega_laser
                Es[3] -= 2 * self.laser.omega.omega_laser

        H_diag = Qobj(np.diag(Es), dims=self.H0_undiagonalized.dims)
        return H_diag

    def H_int(self, t: float) -> Qobj:
        """
        Define the interaction Hamiltonian for the system with multiple pulses using the PulseSequence class.

        Parameters:
            t (float): Time at which the interaction Hamiltonian is evaluated.

        Returns:
            Qobj: Interaction Hamiltonian at time t.
        """
        SM_op = self.system.SM_op  # Lowering operator
        Dip_op = self.system.Dip_op

        if self.simulation.RWA_SL:
            E_field_RWA = E_pulse(t, self.laser)  # Combined electric field under RWA
            H_int = -(
                SM_op.dag() * E_field_RWA + SM_op * np.conj(E_field_RWA)
            )  # RWA interaction Hamiltonian
        else:
            E_field = Epsilon_pulse(
                t, self.laser
            )  # Combined electric field with carrier
            H_int = -Dip_op * (
                E_field + np.conj(E_field)
            )  # Full interaction Hamiltonian

        return H_int

    @property
    def observable_ops(self):
        if self.N_atoms == 1:
            observable_ops = [
                ket2dm(self.atom_g),  # |gxg|
                self.atom_g * self.atom_e.dag(),  # |gxe|
                self.atom_e * self.atom_g.dag(),  # |exg|
                ket2dm(self.atom_e),  # |exe|
            ]
        elif self.keep_track == "basis":
            observable_ops = [
                ket2dm(self.system.basis[i]) for i in range(self.system.N_atoms)
            ]
        elif self.keep_track == "eigenstates":
            observable_ops = [ket2dm(state) for state in self.system.eigenstates[1]]

        return observable_ops

    @property
    def observable_strs(self):
        if self.system.N_atoms == 1:
            observable_strs = ["gxg", "gxe", "exg", "exe"]
        elif self.keep_track == "basis":
            observable_strs = [f"basis({i})" for i in range(self.system.N_atoms)]
            # observable_strs1 = ["0", "A", "B", "AB"]
        elif self.keep_track == "eigenstates":
            observable_strs = [
                f"eigenstate({i})" for i in range(len(self.system.eigenstates[1]))
            ]

        return observable_strs

    def __post_init__(self):
        self.H0_undiagonalized = self.system.H0_undiagonalized
        self.H_sim = self.H0_diagonalized

    def summary(self):
        print("\n# Summary of System Parameters:")
        for key, value in self.__dict__.items():
            print(f"    {key:<20}: {value}")

        # Solver Specifics
        print("\n# Solver specific information:")
        if self.ODE_Solver == "ME":
            print(f"    {'me_decay_channels':<20}:")
            for op in self.me_decay_channels:
                print(f"        {op}")
        elif self.ODE_Solver == "BR":
            print(f"    {'br_decay_channels':<20}:")
            for op_spec in self.br_decay_channels:
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


def test_simulation_model():
    """
    Test function for SimulationModel class functionality.
    Tests H_int method with different system configurations.
    """
    print("\n" + "=" * 60)
    print("TESTING SIMULATION MODEL")
    print("=" * 60)

    # =============================
    # SETUP TEST CONFIGURATIONS
    # =============================

    ### Create simulation configuration
    sim_config = SimulationConfig(
        ODE_Solver="Paper_BR", RWA_SL=True, dt=0.1, tau_coh=50.0
    )

    ### Create test pulse sequence
    from qspectro2d.baths.bath_parameters import BathParameters
    from qspectro2d.core.pulse_sequences import PulseSequence

    test_pulse_seq = PulseSequence()
    test_pulse_seq.from_general_specs(
        pulse_peak_times=[0.0, 50.0],
        pulse_phases=[0.0, 1.57],  # 0, π/2
        pulse_freqs=[1.0, 1.0],
        pulse_fwhms=[15.0, 15.0],
        pulse_amplitudes=[0.05, 0.05],
        pulse_types="gaussian",
    )

    # =============================
    # TEST 1: Single atom system
    # =============================
    print("\n--- Test 1: Single Atom System ---")

    try:
        ### Create 1-atom system

        system_1 = SystemParameters(N_atoms=1)
        bath = BathParameters()

        ### Create dummy coupling objects (minimal setup for testing)
        sb_coupling = SystemBathCoupling(
            system=system_1,  # System for which the coupling is defined
            bath=bath,
        )
        sl_coupling = SystemLaserCoupling(
            system=system_1,  # System for which the coupling is defined
            laser=test_pulse_seq,
        )

        ### Create simulation model
        sim_model_1 = SimulationModel(
            simulation=sim_config,
            system=system_1,
            laser=test_pulse_seq,
            SB_coupling=sb_coupling,
            SL_coupling=sl_coupling,
        )

        ### Test H_int method
        test_time = 1.0
        H_int_1 = sim_model_1.H_int(test_time)

        print(f"✓ Single atom H_int created successfully")
        print(f"  - Type: {type(H_int_1)}")
        print(f"  - Dimensions: {H_int_1.dims}")
        print(f"  - Is Hermitian: {H_int_1.isherm}")
        print(f"  - Matrix shape: {H_int_1.shape}")

    except Exception as e:
        print(f"✗ Error in single atom test: {e}")

    # =============================
    # TEST 2: Two atom system
    # =============================
    print("\n--- Test 2: Two Atom System ---")

    try:
        ### Create 2-atom system
        system_2 = SystemParameters(N_atoms=2)

        ### Create simulation model
        sim_model_2 = SimulationModel(
            simulation=sim_config,
            system=system_2,
            laser=test_pulse_seq,
            SB_coupling=sb_coupling,
            SL_coupling=sl_coupling,
        )

        ### Test H_int method
        H_int_2 = sim_model_2.H_int(test_time)

        print(f"✓ Two atom H_int created successfully")
        print(f"  - Type: {type(H_int_2)}")
        print(f"  - Dimensions: {H_int_2.dims}")
        print(f"  - Is Hermitian: {H_int_2.isherm}")
        print(f"  - Matrix shape: {H_int_2.shape}")

    except Exception as e:
        print(f"✗ Error in two atom test: {e}")

    # =============================
    # TEST 3: Time evolution of H_int
    # =============================
    print("\n--- Test 3: Time Evolution ---")

    try:
        test_times = np.linspace(0, 100, 5)  # Test at different times

        print(f"Testing H_int at different times for single atom system:")
        for t in test_times:
            H_t = sim_model_1.H_int(t)
            max_element = np.max(np.abs(H_t.full()))
            print(f"  t = {t:6.1f} fs: max|H_int| = {max_element:.2e}")

    except Exception as e:
        print(f"✗ Error in time evolution test: {e}")

    # =============================
    # TEST 4: RWA vs non-RWA comparison
    # =============================
    print("\n--- Test 4: RWA vs Non-RWA ---")

    try:
        ### Create non-RWA configuration
        sim_config_no_rwa = SimulationConfig(
            ODE_Solver="Paper_BR", RWA_SL=False, dt=0.1, tau_coh=100.0  # No RWA
        )

        sim_model_no_rwa = SimulationModel(
            simulation=sim_config_no_rwa,
            system=system_1,
            laser=test_pulse_seq,
            SB_coupling=sb_coupling,
            SL_coupling=sl_coupling,
        )

        ### Compare H_int with and without RWA
        H_rwa = sim_model_1.H_int(test_time)
        H_no_rwa = sim_model_no_rwa.H_int(test_time)

        print(f"✓ RWA comparison successful")
        print(f"  - RWA H_int max element: {np.max(np.abs(H_rwa.full())):.2e}")
        print(f"  - Non-RWA H_int max element: {np.max(np.abs(H_no_rwa.full())):.2e}")

    except Exception as e:
        print(f"✗ Error in RWA comparison test: {e}")

    print("\n" + "=" * 60)
    print("SIMULATION MODEL TESTING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    ### Run the test function when script is executed directly
    test_simulation_model()
