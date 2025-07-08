# =============================
# DEFINE THE simulation_config PARAMETERS CLASS
# =============================
from dataclasses import dataclass, asdict  # for the class definiton
import numpy as np
from qspectro2d.core.laser_system.laser_class import LaserPulseSystem
from qspectro2d.core.atomic_system.system_class import AtomicSystem
from core.system_bath_class import SystemBathCoupling
from core.system_laser_class import SystemLaserCoupling
from qutip import Qobj, ket2dm
from qspectro2d.core.laser_system.laser_fcts import Epsilon_pulse, E_pulse

SUPPORTED_SOLVERS = ["ME", "BR", "Paper_eqs", "Paper_BR"]


@dataclass
class SimulationConfigClass:
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

    # Other simulation_config parameters
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

    def __post_init__(self):
        # Validate ODE_Solver
        if self.ODE_Solver not in SUPPORTED_SOLVERS:
            raise ValueError(
                f"Invalid ODE_Solver: {self.ODE_Solver}. Supported solvers are: "
                f"ME, BR, Paper_eqs, Paper_BR."
            )

        # Validate RWA_SL with Paper_eqs -> must be True
        if self.RWA_SL == False and self.ODE_Solver == "Paper_eqs":
            print("WARNING: RWA_SL must be True for Paper_eqs. Setting RWA_SL to True.")
            self.RWA_SL = True
        # Validate other parameters
        if self.dt <= 0:
            raise ValueError("dt must be a positive value.")
        if self.tau_coh < 0:
            raise ValueError("tau_coh must be a positive value.")
        if self.t_wait < 0:
            raise ValueError("t_wait must be non-negative.")
        if self.t_det_max <= 0:
            raise ValueError("t_det_max must be a positive value.")
        if self.n_phases <= 0:
            raise ValueError("n_phases must be a positive integer.")
        if self.n_freqs <= 0:
            raise ValueError("n_freqs must be a positive integer.")

        self.t_max = self.tau_coh + self.t_wait + self.t_det_max
        if self.t_max <= self.t_wait:
            raise ValueError("t_wait is greater than t_max.")

    def summary(self) -> str:
        return (
            f"SimulationConfigClass Summary:\n"
            f"--------------------------\n"
            f"Solver Type        : {self.ODE_Solver}\n"
            f"Use RWA_SL         : {self.RWA_SL}\n\n"
            f"Time Step (dt)     : {self.dt} fs\n"
            f"Coherence Time     : {self.tau_coh} fs\n"
            f"Wait Time          : {self.t_wait} fs\n"
            f"Max Det. Time      : {self.t_det_max} fs\n\n"
            f"Phase Cycles       : {self.n_phases}\n"
            f"Inhom. Points      : {self.n_freqs}\n"
            f"Delta (cm⁻¹)       : {self.Delta_cm}\n"
            f"Total Time (t_max) : {self.t_max} fs\n"
            f"--------------------------\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfigClass":
        return cls(**data)

    def __str__(self) -> str:
        return self.summary()


@dataclass
class SimClassOQS:
    # Default class attributes for the simulation model
    simulation_config: SimulationConfigClass

    system: AtomicSystem
    laser: LaserPulseSystem
    SB_coupling: SystemBathCoupling
    SL_coupling: SystemLaserCoupling

    keep_track: str = "eigenstates"  # alternative "basis" determines the "observables"

    @property
    def H0_diagonalized(self):
        """
        Diagonalize the Hamiltonian and return the eigenvalues and eigenstates.
        WITH RWA

        Returns:
            tuple: Eigenvalues and eigenstates of the Hamiltonian.
        """
        Es, _ = self.system.eigenstates

        if self.simulation_config.RWA_SL:
            if self.system.N_atoms == 1:
                Es[1] -= self.laser.omega_laser

            elif self.system.N_atoms == 2:
                Es[1] -= self.laser.omega.omega_laser
                Es[2] -= self.laser.omega.omega_laser
                Es[3] -= 2 * self.laser.omega.omega_laser
            else:
                print(
                    "TODO extend the H_diag to N_atoms > 2 ?Es[i] -= self.laser.omega_laser?"
                )

        H_diag = Qobj(np.diag(Es), dims=self.H0_undiagonalized.dims)
        return H_diag

    def H_int(self, t: float) -> Qobj:
        """
        Define the interaction Hamiltonian for the system with multiple pulses using the LaserPulseSystem class.

        Parameters:
            t (float): Time at which the interaction Hamiltonian is evaluated.

        Returns:
            Qobj: Interaction Hamiltonian at time t.
        """
        SM_op = self.system.SM_op  # Lowering operator
        RWA_SL = self.simulation_config.RWA_SL
        laser = self.laser

        return H_int_(
            t=t,
            SM_op=SM_op,
            RWA_SL=RWA_SL,
            laser=laser,
        )

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
        # Hamiltonians
        self.H0_undiagonalized = self.system.H0_undiagonalized
        self.H_sim = self.H0_diagonalized

        # TIME GRID FOR SIMULATION
        # =============================
        self.t0 = -self.laser.pulse_fwhms[0]
        self.t_max = self.simulation_config.t_max
        self.dt = self.simulation_config.dt
        n_steps = (
            int(np.round((self.t_max - self.t0) / self.dt)) + 1
        )  # ensure inclusion of t_max
        self.times = np.linspace(self.t0, self.t_max, n_steps)  # include t0 and t_max

        # define the decay channels based on the ODE_Solver
        ODE_Solver = self.simulation_config.ODE_Solver
        if ODE_Solver == "ME":
            # Master equation solver
            self.decay_channels = self.SB_coupling.me_decay_channels
        elif ODE_Solver == "BR":
            # Bloch-Redfield solver
            self.decay_channels = self.SB_coupling.br_decay_channels
        elif ODE_Solver == "Paper_eqs":
            self.simulation_config.RWA_SL = True  # RWA is required for Paper_eqs
            self.decay_channels = []

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


def H_int_(
    t: float,
    SM_op: Qobj,
    RWA_SL: bool,
    laser: LaserPulseSystem,
) -> Qobj:
    """
    Define the interaction Hamiltonian for the system with multiple pulses using the LaserPulseSystem class.

    Parameters:
        t (float): Time at which the interaction Hamiltonian is evaluated.
        SM_op (Qobj): Lowering operator of the system.
        RWA_SL (bool): Whether to use the rotating wave approximation (RWA) for single-laser coupling.
        laser (LaserPulseSystem): Laser pulse system containing the pulse parameters.

    Returns:
        Qobj: Interaction Hamiltonian at time t.
    """

    if RWA_SL:
        E_field_RWA = E_pulse(t, laser)  # Combined electric field under RWA
        H_int = -(
            SM_op.dag() * E_field_RWA + SM_op * np.conj(E_field_RWA)
        )  # RWA interaction Hamiltonian
    else:
        Dip_op = SM_op * SM_op.dag()
        E_field = Epsilon_pulse(t, laser)  # Combined electric field with carrier
        H_int = -Dip_op * (E_field + np.conj(E_field))  # Full interaction Hamiltonian

    return H_int


def test_simulation_model():
    """
    Test function for SimClassOQS class functionality.
    Tests H_int method with different system configurations.
    """
    print("\n" + "=" * 60)
    print("TESTING simulation_config MODEL")
    print("=" * 60)

    # =============================
    # SETUP TEST CONFIGURATIONS
    # =============================

    ### Create simulation_config configuration
    sim_config = SimulationConfigClass(
        ODE_Solver="Paper_BR", RWA_SL=True, dt=0.1, tau_coh=50.0
    )

    ### Create test pulse sequence
    from qspectro2d.core.bath_system.bath_class import BathClass
    from qspectro2d.core.laser_system.laser_class import LaserPulseSystem

    test_pulse_seq = LaserPulseSystem()
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

        system_1 = AtomicSystem(N_atoms=1)
        bath = BathClass()

        ### Create dummy coupling objects (minimal setup for testing)
        sb_coupling = SystemBathCoupling(
            system=system_1,  # System for which the coupling is defined
            bath=bath,
        )
        sl_coupling = SystemLaserCoupling(
            system=system_1,  # System for which the coupling is defined
            laser=test_pulse_seq,
        )

        ### Create simulation_config model
        sim_model_1 = SimClassOQS(
            simulation_config=sim_config,
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
        system_2 = AtomicSystem(N_atoms=2)

        ### Create simulation_config model
        sim_model_2 = SimClassOQS(
            simulation_config=sim_config,
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
        sim_config_no_rwa = SimulationConfigClass(
            ODE_Solver="Paper_BR", RWA_SL=False, dt=0.1, tau_coh=100.0  # No RWA
        )

        sim_model_no_rwa = SimClassOQS(
            simulation_config=sim_config_no_rwa,
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
    print("simulation_config MODEL TESTING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    ### Run the test function when script is executed directly
    test_simulation_model()
