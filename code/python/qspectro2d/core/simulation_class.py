# =============================
# DEFINE THE simulation_config PARAMETERS CLASS
# =============================
from dataclasses import dataclass, asdict, field  # for the class definiton
import numpy as np
from qspectro2d.core.laser_system.laser_class import LaserPulseSystem
from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.core.bath_system.bath_class import BathClass
from qspectro2d.core.system_bath_class import SystemBathCoupling
from qspectro2d.core.system_laser_class import SystemLaserCoupling
from qutip import Qobj, QobjEvo, ket2dm, liouvillian, stacked_index
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
    #  only valid for omega_laser ~ omega_atomic
    RWA_SL: bool = True
    #  CAN ONLY HANDLE TRUE For Paper_eqs
    keep_track: str = "eigenstates"  # alternative "basis" determines the "observables"

    # Other simulation_config parameters
    # times
    dt: float = 0.1  # time step in fs
    tau_coh: float = 100.0  # coherence time in fs
    t_wait: float = 0.0  # wait time before the first pulse in fs
    t_det_max: float = 100.0
    # phase cycling
    n_phases: int = 4  # -> 4*4=16 parallel jobs
    # inhomogeneous broadening
    n_freqs: int = 1

    # additional parameters
    max_workers: int = 1  # Number of parallel workers for the simulation
    simulation_type: str = "1d"  # Type of simulation, e.g., "1d", "2d"
    #
    apply_ift: bool = (
        True  # Apply inverse Fourier transform to get the photon echo signal
    )

    @property
    def combinations(self) -> int:
        """
        Calculate the total number of combinations based on phase cycles and inhomogeneous points.
        """
        return self.n_phases * self.n_phases * self.n_freqs

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
            f"-------------------------------\n"
            f"{self.simulation_type} ELECTRONIC SPECTROSCOPY SIMULATION\n"
            f"Time Parameters:\n"
            f"Coherence Time     : {self.tau_coh} fs\n"
            f"Wait Time          : {self.t_wait} fs\n"
            f"Max Det. Time      : {self.t_det_max} fs\n\n"
            f"Total Time (t_max) : {self.t_max} fs\n"
            f"Time Step (dt)     : {self.dt} fs\n"
            f"-------------------------------\n"
            f"Solver Type        : {self.ODE_Solver}\n"
            f"Use RWA_SL         : {self.RWA_SL}\n\n"
            f"-------------------------------\n"
            f"Phase Cycles       : {self.n_phases}\n"
            f"Inhom. Points      : {self.n_freqs}\n"
            f"Total Combinations : {self.combinations}\n"
            f"Max Workers        : {self.max_workers}\n"
            f"-------------------------------\n"
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
    bath: BathClass
    sl_coupling: SystemLaserCoupling = field(init=False)
    sb_coupling: SystemBathCoupling = field(init=False)

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
                Es[1] -= self.laser.omega_laser
                Es[2] -= self.laser.omega_laser
                Es[3] -= 2 * self.laser.omega_laser
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
        if self.system.N_atoms == 1:
            observable_ops = [
                ket2dm(self.system.atom_g),  # |gxg|
                self.system.atom_g * self.system.atom_e.dag(),  # |gxe|
                self.system.atom_e * self.system.atom_g.dag(),  # |exg|
                ket2dm(self.system.atom_e),  # |exe|
            ]
        elif self.simulation_config.keep_track == "basis":
            observable_ops = [
                ket2dm(self.system.basis[i]) for i in range(self.system.N_atoms)
            ]
        elif self.simulation_config.keep_track == "eigenstates":
            observable_ops = [ket2dm(state) for state in self.system.eigenstates[1]]

        return observable_ops

    @property
    def observable_strs(self):
        if self.system.N_atoms == 1:
            observable_strs = ["gxg", "gxe", "exg", "exe"]
        elif self.simulation_config.keep_track == "basis":
            observable_strs = [f"basis({i})" for i in range(self.system.N_atoms)]
            # observable_strs1 = ["0", "A", "B", "AB"]
        elif self.simulation_config.keep_track == "eigenstates":
            observable_strs = [
                f"eigenstate({i})" for i in range(len(self.system.eigenstates[1]))
            ]

        return observable_strs

    def __post_init__(self):
        # Generate the coupling objects
        self.sb_coupling = SystemBathCoupling(self.system, self.bath)
        self.sl_coupling = SystemLaserCoupling(self.system, self.laser)

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
        n_steps_det = int(np.round(self.simulation_config.t_det_max / self.dt)) + 1
        self.times_det = np.linspace(0, self.simulation_config.t_det_max, n_steps_det)

        # define the decay channels and EVOlution objectsbased on the ODE_Solver
        ODE_Solver = self.simulation_config.ODE_Solver
        if ODE_Solver == "ME":
            # Master equation solver
            self.decay_channels = self.sb_coupling.me_decay_channels
            self.Evo_obj_free = self.H0_undiagonalized
            self.Evo_obj_int = lambda t: self.H_int(t)
        elif ODE_Solver == "BR":
            # Bloch-Redfield solver
            self.decay_channels = self.sb_coupling.br_decay_channels
            self.Evo_obj_free = self.H0_undiagonalized
            self.Evo_obj_int = lambda t: self.H_int(t)
        elif ODE_Solver == "Paper_eqs":
            self.simulation_config.RWA_SL = True  # RWA is required for Paper_eqs
            self.decay_channels = []
            custom = lambda t: matrix_ODE_paper(t, self)
            self.Evo_obj_free = custom
            self.Evo_obj_int = custom
        elif ODE_Solver == "Paper_BR":
            custom_free = liouvillian(self.H0_diagonalized) + R_paper(
                self
            )  # TODO somehow contains 2 RWAs for N_atoms == 2.
            custom_int = custom_free + liouvillian(QobjEvo(self.H_int))
            self.Evo_obj_free = custom_free
            self.Evo_obj_int = custom_int

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


# =============================
# THE PAPER SOLVER FUNCTIONS
# =============================
# =============================
# "Paper_eqs" OWN ODE SOLVER
# =============================
def matrix_ODE_paper(t: float, sim_oqs: SimClassOQS) -> Qobj:
    """
    Dispatches to the appropriate implementation based on N_atoms.
    Solves the equation drho_dt = L(t) * rho,
    in natural units: L = -i/hbar(Hrho - rho H) + R * rho,  with [hbar] = 1 and [R] = [1] = [power Spectrum S(w)] = [all the Gammas: like gamma_phi].
    """
    N_atoms = sim_oqs.system.N_atoms
    if N_atoms == 1:
        return _matrix_ODE_paper_1atom(t, sim_oqs)
    elif N_atoms == 2:
        return _matrix_ODE_paper_2atom(t, sim_oqs)
    else:
        raise ValueError("Only N_atoms=1 or 2 are supported.")


def _matrix_ODE_paper_1atom(t: float, sim_oqs: SimClassOQS) -> Qobj:
    """
    Constructs the matrix L(t) for the equation
    drho_dt = L(t) · vec(rho),   QuTiP-kompatibel (column stacking).
    Uses gamma values from the provided system.

    Parameters:
        t (float): Time at which to evaluate the matrix.
        pulse_seq (LaserPulseSystem): LaserPulseSystem object for the electric field.
        system (AtomicSystem): System parameters containing Gamma, gamma_0, and mu_eg.

    Returns:
        Qobj: Liouvillian matrix as a Qobj.
    """
    pulse_seq = sim_oqs.laser
    Et = E_pulse(t, pulse_seq)
    Et_conj = np.conj(Et)
    μ = sim_oqs.system.dip_moments[0]

    gamma0 = sim_oqs.bath.gamma_0
    Γ = sim_oqs.bath.Gamma  # dephasing

    size = 2  # 2 states |g>, |e>

    idx_00 = stacked_index(size, row=0, col=0)  # ρ_gg
    idx_01 = stacked_index(size, row=0, col=1)  # ρ_ge
    idx_10 = stacked_index(size, row=1, col=0)  # ρ_eg
    idx_11 = stacked_index(size, row=1, col=1)  # ρ_ee

    L = np.zeros((4, 4), dtype=complex)

    # ----- d/dt ρ_gg
    L[idx_00, idx_11] = gamma0
    L[idx_00, idx_01] = -1j * Et * μ
    L[idx_00, idx_10] = +1j * Et_conj * μ

    # ----- d/dt ρ_ee
    L[idx_11, :] = -L[idx_00, :]  # Ensures trace conservation

    # ----- d/dt ρ_eg
    L[idx_10, idx_00] = +1j * Et * μ  # ρ_gg
    L[idx_10, idx_11] = -1j * Et * μ  # ρ_ee

    L[idx_10, idx_10] = -Γ  # Decay term for coherence

    # ----- d/dt ρ_ge  – complex conjugate
    L[idx_01, idx_00] = -1j * Et_conj * μ  # ρ_gg
    L[idx_01, idx_11] = +1j * Et_conj * μ  # ρ_ee

    L[idx_01, idx_01] = -Γ  # Decay term for coherence

    return Qobj(L, dims=[[[2], [2]], [[2], [2]]])  # 'super' wird aus den Dims erkannt


# carefull i changed this function with GPT
def _matrix_ODE_paper_2atom(
    t: float,
    sim_oqs: SimClassOQS,
) -> Qobj:
    """
    Column-stacked Liouvillian L(t) such that         d/dt vec(rho) = L(t) · vec(rho)

    Index-Konvention (column stacking, wie in QuTiP):
        vec(rho)[ i + 4*j ]   ↔   rho_{ij}     für i,j = 0…3
    """
    # --------------------------------------------------------------
    # Helpers & short-hands
    # --------------------------------------------------------------
    pulse_seq = sim_oqs.laser
    Et = E_pulse(t, pulse_seq)
    Et_conj = np.conj(Et)
    omega_laser = sim_oqs.laser.omega_laser

    size = 4  # 4 states |0>, |1>, |2>, |3>

    # Define all indices using stacked_index
    idx_00 = stacked_index(size, row=0, col=0)  # ρ_00
    idx_01 = stacked_index(size, row=0, col=1)  # ρ_01
    idx_02 = stacked_index(size, row=0, col=2)  # ρ_02
    idx_03 = stacked_index(size, row=0, col=3)  # ρ_03
    idx_10 = stacked_index(size, row=1, col=0)  # ρ_10
    idx_11 = stacked_index(size, row=1, col=1)  # ρ_11
    idx_12 = stacked_index(size, row=1, col=2)  # ρ_12
    idx_13 = stacked_index(size, row=1, col=3)  # ρ_13
    idx_20 = stacked_index(size, row=2, col=0)  # ρ_20
    idx_21 = stacked_index(size, row=2, col=1)  # ρ_21
    idx_22 = stacked_index(size, row=2, col=2)  # ρ_22
    idx_23 = stacked_index(size, row=2, col=3)  # ρ_23
    idx_30 = stacked_index(size, row=3, col=0)  # ρ_30
    idx_31 = stacked_index(size, row=3, col=1)  # ρ_31
    idx_32 = stacked_index(size, row=3, col=2)  # ρ_32
    idx_33 = stacked_index(size, row=3, col=3)  # ρ_33

    L = np.zeros((size * size, size * size), dtype=complex)

    # --------------------------------------------------------------
    # 1) Off-diagonal one-excited-coherences
    # --------------------------------------------------------------
    # ρ_10   (|1⟩⟨0|)
    term = -1j * (
        sim_oqs.system.omega_ij(1, 0) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(1, 0)
    L[idx_10, idx_10] = term
    L[idx_10, idx_00] = 1j * Et * sim_oqs.system.Dip_op[1, 0]
    L[idx_10, idx_11] = -1j * Et * sim_oqs.system.Dip_op[1, 0]
    L[idx_10, idx_12] = -1j * Et * sim_oqs.system.Dip_op[2, 0]
    L[idx_10, idx_30] = 1j * Et_conj * sim_oqs.system.Dip_op[3, 1]

    # ρ_01   = (ρ_10)†
    L[idx_01, idx_01] = np.conj(term)
    L[idx_01, idx_00] = np.conj(L[idx_10, idx_00])
    L[idx_01, idx_11] = np.conj(L[idx_10, idx_11])
    L[idx_01, idx_21] = np.conj(L[idx_10, idx_12])
    L[idx_01, idx_03] = np.conj(L[idx_10, idx_30])

    # ρ_20   (|2⟩⟨0|)
    term = -1j * (
        sim_oqs.system.omega_ij(2, 0) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(2, 0)
    L[idx_20, idx_20] = term
    L[idx_20, idx_00] = 1j * Et * sim_oqs.system.Dip_op[2, 0]
    L[idx_20, idx_22] = -1j * Et * sim_oqs.system.Dip_op[2, 0]
    L[idx_20, idx_21] = -1j * Et * sim_oqs.system.Dip_op[1, 0]
    L[idx_20, idx_30] = 1j * Et_conj * sim_oqs.system.Dip_op[3, 2]

    # ρ_02
    L[idx_02, idx_02] = np.conj(term)
    L[idx_02, idx_00] = np.conj(L[idx_20, idx_00])
    L[idx_02, idx_22] = np.conj(L[idx_20, idx_22])
    L[idx_02, idx_12] = np.conj(L[idx_20, idx_21])
    L[idx_02, idx_03] = np.conj(L[idx_20, idx_30])

    # --------------------------------------------------------------
    # 2) double-excited-coherences
    # --------------------------------------------------------------
    # ρ_30   (|3⟩⟨0|)
    term = -1j * (
        sim_oqs.system.omega_ij(3, 0) - 2 * omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(3, 0)
    L[idx_30, idx_30] = term
    L[idx_30, idx_10] = 1j * Et * sim_oqs.system.Dip_op[3, 1]
    L[idx_30, idx_20] = 1j * Et * sim_oqs.system.Dip_op[3, 2]
    L[idx_30, idx_31] = -1j * Et * sim_oqs.system.Dip_op[1, 0]
    L[idx_30, idx_32] = -1j * Et * sim_oqs.system.Dip_op[2, 0]

    # ρ_03
    L[idx_03, idx_03] = np.conj(term)
    L[idx_03, idx_01] = np.conj(L[idx_30, idx_10])
    L[idx_03, idx_02] = np.conj(L[idx_30, idx_20])
    L[idx_03, idx_13] = np.conj(L[idx_30, idx_31])
    L[idx_03, idx_23] = np.conj(L[idx_30, idx_32])

    # --------------------------------------------------------------
    # 3) cross-coherences inside one excitation manifold
    # --------------------------------------------------------------
    # ρ_12   (|1⟩⟨2|)
    term = -1j * sim_oqs.system.omega_ij(1, 2) - sim_oqs.sb_coupling.Gamma_big_ij(1, 2)
    L[idx_12, idx_12] = term
    L[idx_12, idx_02] = 1j * Et * sim_oqs.system.Dip_op[1, 0]
    L[idx_12, idx_13] = -1j * Et * sim_oqs.system.Dip_op[3, 2]
    L[idx_12, idx_32] = 1j * Et_conj * sim_oqs.system.Dip_op[3, 1]
    L[idx_12, idx_10] = -1j * Et_conj * sim_oqs.system.Dip_op[2, 0]

    # ρ_21
    L[idx_21, idx_21] = np.conj(term)
    L[idx_21, idx_20] = np.conj(L[idx_12, idx_02])
    L[idx_21, idx_31] = np.conj(L[idx_12, idx_13])
    L[idx_21, idx_23] = np.conj(L[idx_12, idx_32])
    L[idx_21, idx_01] = np.conj(L[idx_12, idx_10])

    # ρ_31   (|3⟩⟨1|)
    term = -1j * (
        sim_oqs.system.omega_ij(3, 1) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(3, 1)
    L[idx_31, idx_31] = term
    L[idx_31, idx_11] = 1j * Et * sim_oqs.system.Dip_op[3, 1]
    L[idx_31, idx_21] = 1j * Et * sim_oqs.system.Dip_op[3, 2]
    L[idx_31, idx_30] = -1j * Et_conj * sim_oqs.system.Dip_op[1, 0]

    # ρ_13
    L[idx_13, idx_13] = np.conj(term)
    L[idx_13, idx_11] = np.conj(L[idx_31, idx_11])
    L[idx_13, idx_12] = np.conj(L[idx_31, idx_21])
    L[idx_13, idx_03] = np.conj(L[idx_31, idx_30])

    # ρ_32   (|3⟩⟨2|)
    term = -1j * (
        sim_oqs.system.omega_ij(3, 2) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(3, 2)
    L[idx_32, idx_32] = term
    L[idx_32, idx_22] = 1j * Et * sim_oqs.system.Dip_op[3, 2]
    L[idx_32, idx_12] = 1j * Et * sim_oqs.system.Dip_op[3, 1]
    L[idx_32, idx_30] = -1j * Et_conj * sim_oqs.system.Dip_op[2, 0]

    # ρ_23
    L[idx_23, idx_23] = np.conj(term)
    L[idx_23, idx_22] = np.conj(L[idx_32, idx_22])
    L[idx_23, idx_21] = np.conj(L[idx_32, idx_12])
    L[idx_23, idx_03] = np.conj(L[idx_32, idx_30])

    # --------------------------------------------------------------
    # 4) Populations (diagonals)
    # --------------------------------------------------------------
    # ρ_00
    L[idx_00, idx_01] = -1j * Et * sim_oqs.system.Dip_op[1, 0]
    L[idx_00, idx_02] = -1j * Et * sim_oqs.system.Dip_op[2, 0]
    L[idx_00, idx_10] = 1j * Et_conj * sim_oqs.system.Dip_op[1, 0]
    L[idx_00, idx_20] = 1j * Et_conj * sim_oqs.system.Dip_op[2, 0]

    # ρ_11
    L[idx_11, idx_11] = -sim_oqs.sb_coupling.Gamma_big_ij(1, 1)
    L[idx_11, idx_22] = sim_oqs.sb_coupling.gamma_small_ij(1, 2)
    L[idx_11, idx_01] = 1j * Et * sim_oqs.system.Dip_op[1, 0]
    L[idx_11, idx_13] = -1j * Et * sim_oqs.system.Dip_op[3, 1]
    L[idx_11, idx_31] = 1j * Et_conj * sim_oqs.system.Dip_op[3, 1]
    L[idx_11, idx_10] = -1j * Et_conj * sim_oqs.system.Dip_op[1, 0]

    # ρ_22
    L[idx_22, idx_22] = -sim_oqs.sb_coupling.Gamma_big_ij(2, 2)
    L[idx_22, idx_11] = sim_oqs.sb_coupling.gamma_small_ij(2, 1)
    L[idx_22, idx_02] = 1j * Et * sim_oqs.system.Dip_op[2, 0]
    L[idx_22, idx_23] = -1j * Et * sim_oqs.system.Dip_op[3, 2]
    L[idx_22, idx_32] = 1j * Et_conj * sim_oqs.system.Dip_op[3, 2]
    L[idx_22, idx_20] = -1j * Et_conj * sim_oqs.system.Dip_op[2, 0]

    # ρ_33  – Spurbedingung: dρ_00 + dρ_11 + dρ_22 + dρ_33 = 0
    L[idx_33, :] = -L[idx_00, :] - L[idx_11, :] - L[idx_22, :]

    return Qobj(L, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]])


# only use the Redfield tensor as a matrix:
def R_paper(sim_oqs: SimClassOQS) -> Qobj:
    """Dispatches to the appropriate implementation based on N_atoms."""
    N_atoms = sim_oqs.system.N_atoms
    if N_atoms == 1:
        return _R_paper_1atom(sim_oqs)
    elif N_atoms == 2:
        return _R_paper_2atom(sim_oqs)
    else:
        raise ValueError("Only N_atoms=1 or 2 are supported.")


def _R_paper_1atom(sim_oqs: SimClassOQS) -> Qobj:
    """
    Constructs the Redfield Tensor R for the equation drho_dt = -i(Hrho - rho H) + R * rho,
    where rho is the flattened density matrix. Uses gamma values from the provided system.

    Parameters:
        system (AtomicSystem): System parameters containing Gamma and gamma_0.

    Returns:
        Qobj: Redfield tensor as a Qobj.
    """
    R = np.zeros((4, 4), dtype=complex)  # Redfield tensor initialization
    Gamma = sim_oqs.bath.Gamma  # dephasing
    gamma_0 = sim_oqs.bath.gamma_0  # population decay
    # --- d/dt rho_eg ---
    R[2, 2] = -Gamma  # Decay term for coherence
    # --- d/dt rho_ge ---
    R[1, 1] = -Gamma

    # --- d/dt rho_ee ---
    R[3, 3] = -gamma_0  # Decay term for population
    # --- d/dt rho_gg ---
    R[0, 3] = gamma_0  # Ensures trace conservation

    return Qobj(R, dims=[[[2], [2]], [[2], [2]]])


def _R_paper_2atom(sim_oqs: SimClassOQS) -> Qobj:
    """
    including RWA
    Constructs the Redfield Tensor R for the equation drho_dt = -i(Hrho - rho H) + R * rho,
    where rho is the flattened density matrix.
    """
    size = 4  # 4 states |0>, |1>, |2>, |3>

    # Define all indices using stacked_index
    idx_00 = stacked_index(size, row=0, col=0)  # ρ_00
    idx_01 = stacked_index(size, row=0, col=1)  # ρ_01
    idx_02 = stacked_index(size, row=0, col=2)  # ρ_02
    idx_03 = stacked_index(size, row=0, col=3)  # ρ_03
    idx_10 = stacked_index(size, row=1, col=0)  # ρ_10
    idx_11 = stacked_index(size, row=1, col=1)  # ρ_11
    idx_12 = stacked_index(size, row=1, col=2)  # ρ_12
    idx_13 = stacked_index(size, row=1, col=3)  # ρ_13
    idx_20 = stacked_index(size, row=2, col=0)  # ρ_20
    idx_21 = stacked_index(size, row=2, col=1)  # ρ_21
    idx_22 = stacked_index(size, row=2, col=2)  # ρ_22
    idx_23 = stacked_index(size, row=2, col=3)  # ρ_23
    idx_30 = stacked_index(size, row=3, col=0)  # ρ_30
    idx_31 = stacked_index(size, row=3, col=1)  # ρ_31
    idx_32 = stacked_index(size, row=3, col=2)  # ρ_32
    idx_33 = stacked_index(size, row=3, col=3)  # ρ_33

    R = np.zeros((size * size, size * size), dtype=complex)
    omega_laser = sim_oqs.laser.omega_laser
    # --------------------------------------------------------------
    # 1) Off-diagonal one-excited-coherences
    # --------------------------------------------------------------
    # --- d/dt rho_10 ---
    term = -1j * (
        sim_oqs.system.omega_ij(1, 0) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(1, 0)
    R[idx_10, idx_10] = term  # ρ₁₀ ← ρ₁₀

    # --- d/dt rho_01 ---
    R[idx_01, idx_01] = np.conj(term)  # ρ₀₁ ← ρ₀₁

    # --- d/dt rho_20 --- = ANSATZ = (d/dt s_20 - i omega_laser s_20) e^(-i omega_laser t)
    term = -1j * (
        sim_oqs.system.omega_ij(2, 0) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(2, 0)
    R[idx_20, idx_20] = term  # ρ₂₀ ← ρ₂₀

    # --- d/dt rho_02 ---
    R[idx_02, idx_02] = np.conj(term)  # ρ₀₂ ← ρ₀₂

    # --------------------------------------------------------------
    # 2) double-excited-coherences
    # --------------------------------------------------------------
    # --- d/dt rho_30 ---
    term = -1j * (
        sim_oqs.system.omega_ij(3, 0) - 2 * omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(3, 0)
    R[idx_30, idx_30] = term  # ρ₃₀ ← ρ₃₀

    # --- d/dt rho_03 ---
    R[idx_03, idx_03] = np.conj(term)  # ρ₀₃ ← ρ₀₃

    # --------------------------------------------------------------
    # 3) cross-coherences inside one excitation manifold
    # --------------------------------------------------------------
    # --- d/dt rho_12 ---
    term = -1j * sim_oqs.system.omega_ij(1, 2) - sim_oqs.sb_coupling.Gamma_big_ij(1, 2)
    R[idx_12, idx_12] = term  # ρ₁₂ ← ρ₁₂

    # --- d/dt rho_21 ---
    R[idx_21, idx_21] = np.conj(term)  # ρ₂₁ ← ρ₂₁

    # --- d/dt rho_31 ---
    term = -1j * (
        sim_oqs.system.omega_ij(3, 1) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(3, 1)
    R[idx_31, idx_31] = term  # ρ₃₁ ← ρ₃₁

    # --- d/dt rho_13 ---
    R[idx_13, idx_13] = np.conj(term)  # ρ₁₃ ← ρ₁₃

    # --- d/dt rho_32 ---
    term = -1j * (
        sim_oqs.system.omega_ij(3, 2) - omega_laser
    ) - sim_oqs.sb_coupling.Gamma_big_ij(3, 2)
    R[idx_32, idx_32] = term  # ρ₃₂ ← ρ₃₂

    # --- d/dt rho_23 ---
    R[idx_23, idx_23] = np.conj(term)  # ρ₂₃ ← ρ₂₃

    # --------------------------------------------------------------
    # 4) populations (diagonals)
    # --------------------------------------------------------------
    # --- d/dt rho_11 ---
    R[idx_11, idx_11] = -sim_oqs.sb_coupling.Gamma_big_ij(1, 1)
    R[idx_11, idx_22] = sim_oqs.sb_coupling.gamma_small_ij(
        1, 2
    )  # for the coupled dimer: pop transer

    # --- d/dt rho_22 ---
    R[idx_22, idx_22] = -sim_oqs.sb_coupling.Gamma_big_ij(2, 2)
    R[idx_22, idx_11] = sim_oqs.sb_coupling.gamma_small_ij(
        2, 1
    )  # for the coupled dimer: pop transer

    # NOW THERE IS NO POPULATION CHANGE in 3 || 1 goes to 2 and vice versa
    # --- d/dt rho_00 --- and  --- d/dt rho_33 (sum d/dt rho_ii = 0) (trace condition) ---
    R[idx_33, :] = -R[idx_00, :] - R[idx_11, :] - R[idx_22, :]

    return Qobj(R, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]])


def main():
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

    test_pulse_seq = LaserPulseSystem.from_general_specs(
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

    ### Create 1-atom system

    system_1 = AtomicSystem(N_atoms=1)
    bath = BathClass()

    ### Create simulation_config model
    sim_model_1 = SimClassOQS(
        simulation_config=sim_config,
        system=system_1,
        laser=test_pulse_seq,
        bath=bath,
    )
    try:
        ### Create 1-atom system

        system_1 = AtomicSystem(N_atoms=1)
        bath = BathClass()

        ### Create simulation_config model
        sim_model_1 = SimClassOQS(
            simulation_config=sim_config,
            system=system_1,
            laser=test_pulse_seq,
            bath=bath,
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
        system_2 = AtomicSystem(
            N_atoms=2, freqs_cm=[16000.0, 16100.0], dip_moments=[1.0, 2.0]
        )

        ### Create simulation_config model
        sim_model_2 = SimClassOQS(
            simulation_config=sim_config,
            system=system_2,
            laser=test_pulse_seq,
            bath=bath,
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
            bath=bath,
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
    main()
