# =============================
# DEFINE THE simulation_config PARAMETERS CLASS
# =============================
from dataclasses import dataclass, asdict, field
from os import times
import re  # for the class definiton
import numpy as np
from zmq import has
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.core.bath_system.bath_class import BathSystem
from qspectro2d.core.system_bath_class import SystemBathCoupling
from qspectro2d.core.system_laser_class import SystemLaserCoupling
from qutip import Qobj, QobjEvo, ket2dm, liouvillian, stacked_index
from qspectro2d.core.laser_system.laser_fcts import Epsilon_pulse, E_pulse

SUPPORTED_SOLVERS = ["ME", "BR", "Paper_eqs", "Paper_BR"]


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
    #  only valid for omega_laser ~ omega_atomic
    RWA_SL: bool = True
    #  CAN ONLY HANDLE TRUE For Paper_eqs
    keep_track: str = "eigenstates"  # alternative "basis" determines the "observables"

    # Other simulation_config parameters
    # times
    dt: float = 0.1  # time step in fs
    t_coh: float = 100.0  # coherence time in fs
    t_wait: float = 0.0  # wait time before the first pulse in fs
    t_det_max: float = 100.0
    # phase cycling
    n_phases: int = 4  # -> 4*4=16 parallel jobs
    # inhomogeneous broadening
    n_freqs: int = 1

    # additional parameters
    max_workers: int = 1  # Number of parallel workers for the simulation
    simulation_type: str = "1d"  # Type of simulation, e.g., "1d", "2d"

    # IFT parameters
    IFT_component: tuple = (
        1,
        -1,
        0,
    )  # classical average || (1, -1, 0) == photon echo signal

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
        if self.t_coh < 0:
            raise ValueError("t_coh must be a positive value.")
        if self.t_wait < 0:
            raise ValueError("t_wait must be non-negative.")
        if self.t_det_max <= 0:
            raise ValueError("t_det_max must be a positive value.")
        if self.n_phases <= 0:
            raise ValueError("n_phases must be a positive integer.")
        if self.n_freqs <= 0:
            raise ValueError("n_freqs must be a positive integer.")

        if self.t_coh < self.t_det_max:
            self.t_max = self.t_wait + 2 * self.t_det_max
        else:
            self.t_max = self.t_coh + self.t_wait + self.t_det_max

    def summary(self) -> str:
        return (
            f"SimulationConfig Summary:\n"
            f"-------------------------------\n"
            f"{self.simulation_type} ELECTRONIC SPECTROSCOPY SIMULATION\n"
            f"Time Parameters:\n"
            f"Coherence Time     : {self.t_coh} fs\n"
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
    def from_dict(cls, data: dict) -> "SimulationConfig":
        return cls(**data)

    def __str__(self) -> str:
        return self.summary()


@dataclass
class SimulationModuleOQS:
    # Default class attributes for the simulation model
    simulation_config: SimulationConfig

    system: AtomicSystem
    laser: LaserPulseSequence
    bath: BathSystem
    sl_coupling: SystemLaserCoupling = field(init=False)
    sb_coupling: SystemBathCoupling = field(init=False)

    def __post_init__(self):
        # Generate the coupling objects
        self.sb_coupling = SystemBathCoupling(self.system, self.bath)
        self.sl_coupling = SystemLaserCoupling(self.system, self.laser)

        # define the decay channels based on the ODE_Solver
        ODE_Solver = self.simulation_config.ODE_Solver
        if ODE_Solver == "ME":
            self.decay_channels = self.sb_coupling.me_decay_channels
        elif ODE_Solver == "BR":
            self.decay_channels = self.sb_coupling.br_decay_channels
        elif ODE_Solver == "Paper_eqs":
            self.simulation_config.RWA_SL = True  # RWA is required for Paper_eqs
            self.decay_channels = []
        elif ODE_Solver == "Paper_BR":
            self.decay_channels = []
            # Redfield tensor, holds decay_channels

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

        H_diag = Qobj(np.diag(Es), dims=self.system.H0_N_canonical.dims)
        return H_diag

    def H_int_sl(self, t: float) -> Qobj:
        """
        Define the interaction Hamiltonian for the system with multiple pulses using the LaserPulseSequence class.

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
                ket2dm(self.system._atom_g),  # |gxg|
                self.system._atom_g * self.system._atom_e.dag(),  # |gxe|
                self.system._atom_e * self.system._atom_g.dag(),  # |exg|
                ket2dm(self.system._atom_e),  # |exe|
            ]
        elif self.simulation_config.keep_track == "basis":
            observable_ops = [
                ket2dm(self.system.basis[i]) for i in range(len(self.system.basis))
            ]
        elif self.simulation_config.keep_track == "eigenstates":
            observable_ops = [ket2dm(state) for state in self.system.eigenstates[1]]

        return observable_ops

    @property
    def observable_strs(self):
        if self.system.N_atoms == 1:
            observable_strs = [
                r" g \times g ",
                r" g \times e ",
                r" e \times g ",
                r" e \times e ",
            ]
        elif self.simulation_config.keep_track == "basis":
            observable_strs = [
                rf"\text{{basis}}({i})" for i in range(self.system.N_atoms)
            ]
            # observable_strs1 = ["0", "A", "B", "AB"]
        elif self.simulation_config.keep_track == "eigenstates":
            observable_strs = [
                rf"\text{{eigenstate}}({i})"
                for i in range(len(self.system.eigenstates[1]))
            ]

        return observable_strs

    def paper_evo_obj(self, t: float) -> Qobj:
        return matrix_ODE_paper(t, self)

    # TIME GRID FOR SIMULATION
    @property
    def times_global(self):
        if not hasattr(self, "_times_global"):
            # Compute if not manually set
            t0 = -self.laser.pulse_fwhms[0]
            t_max = self.simulation_config.t_max
            dt = self.simulation_config.dt

            n_steps = int(np.round((t_max - t0) / dt)) + 1  # ensure inclusion of t_max
            self._times_global = np.linspace(t0, t_max, n_steps)  # include t0 and t_max
        return self._times_global

    @times_global.setter
    def times_global(self, value):
        self._times_global = value

    @property
    def times_local(self):
        if hasattr(self, "_times_local_manual"):
            return self._times_local_manual  # Manual override

        times_global = self.times_global
        # Automatically compute based on current config
        t_coh = self.simulation_config.t_coh
        t_wait = self.simulation_config.t_wait
        t_max_curr = t_coh + t_wait + self.simulation_config.t_det_max
        idx = np.abs(times_global - t_max_curr).argmin()

        return self.times_global[: idx + 1]

    @times_local.setter
    def times_local(self, value):
        self._times_local_manual = value  # Explicitly set value

    def reset_times_local(self):
        """Clear manual override, return to auto-computed behavior."""
        if hasattr(self, "_times_local_manual"):
            del self._times_local_manual

    @property
    def times_det(self):
        dt = self.simulation_config.dt
        t_det_max = self.simulation_config.t_det_max

        n_steps_det = int(np.round(t_det_max / dt)) + 1
        times_det = np.linspace(0, t_det_max, n_steps_det)
        self._times_det = times_det  # Store in the instance for later use
        return self._times_det

    @property
    def times_det_actual(self):
        """Returns the actual detection times."""
        self.reset_times_local()
        times_det_actual = self.times_local[-len(self.times_det) :]
        return times_det_actual

    @property
    def Evo_obj_free(self):
        ODE_Solver = self.simulation_config.ODE_Solver
        if ODE_Solver == "ME":
            return self.H0_diagonalized
        elif ODE_Solver == "BR":
            return self.H0_diagonalized
        elif ODE_Solver == "Paper_eqs":
            return self.paper_evo_obj
        elif ODE_Solver == "Paper_BR":
            return liouvillian(self.H0_diagonalized) + R_paper(self)
        else:
            raise ValueError(f"Unknown ODE_Solver: {ODE_Solver}")

    @property
    def Evo_obj_int(self):
        ODE_Solver = self.simulation_config.ODE_Solver
        if ODE_Solver == "ME":
            return self.H0_diagonalized + QobjEvo(self.H_int_sl)
        elif ODE_Solver == "BR":
            return self.H0_diagonalized + QobjEvo(self.H_int_sl)
        elif ODE_Solver == "Paper_eqs":
            return self.paper_evo_obj
        elif ODE_Solver == "Paper_BR":  # TODO somehow contains 2 RWAs for N_atoms == 2.
            custom_free = liouvillian(self.H0_diagonalized) + R_paper(self)
            return custom_free + liouvillian(QobjEvo(self.H_int_sl))
        else:
            raise ValueError(f"Unknown ODE_Solver: {ODE_Solver}")

    def summary(self):
        print("\n# Summary of System Parameters:")
        for key, value in self.__dict__.items():
            print(f"    {key:<20}: {value}")

        # Solver Specifics
        ODE_Solver = self.simulation_config.ODE_Solver
        print(f"    {'ODE_Solver':<20}: {ODE_Solver}")
        print(f"        decay_channels: {self.decay_channels}")
        if ODE_Solver == "Paper_BR":
            print(f"    {'Redfield tensor R_paper used (calculated by R_paper(self))'}")
        if ODE_Solver == "Paper_eqs":
            print(
                f"    {'Custom ODE matrix used (calculated by matrix_ODE_paper(t, pulse_seq, self))'}"
            )

    def to_dict(self) -> dict:
        """
        Converts the SimulationModuleOQS instance and its nested objects into a dictionary.
        Handles nested dataclasses and provides custom handling for non-dataclass
        objects like Qobj and callables.
        """
        # Start with asdict for the main structure and nested dataclasses
        data = asdict(self)
        data["system"] = self.system.to_dict()
        data["laser"] = self.laser.to_dict()
        data["bath"] = self.bath.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationModuleOQS":
        """
        Creates a SimulationModuleOQS instance from a dictionary.
        Reconstructs nested objects from their dictionary representations.

        Parameters:
            data (dict): Dictionary containing all necessary data to reconstruct the object.

        Returns:
            SimulationModuleOQS: Reconstructed simulation model instance.
        """
        # Reconstruct nested objects from their dictionary representations
        simulation_config = SimulationConfig.from_dict(data["simulation_config"])
        system = AtomicSystem.from_dict(data["system"])
        laser = LaserPulseSequence.from_dict(data["laser"])
        bath = BathSystem.from_dict(data["bath"])

        return cls(
            simulation_config=simulation_config, system=system, laser=laser, bath=bath
        )


def H_int_(
    t: float,
    SM_op: Qobj,
    RWA_SL: bool,
    laser: LaserPulseSequence,
) -> Qobj:
    """
    Define the interaction Hamiltonian for the system with multiple pulses using the LaserPulseSequence class.

    Parameters:
        t (float): Time at which the interaction Hamiltonian is evaluated.
        SM_op (Qobj): Lowering operator of the system.
        RWA_SL (bool): Whether to use the rotating wave approximation (RWA) for single-laser coupling.
        laser (LaserPulseSequence): Laser pulse system containing the pulse parameters.

    Returns:
        Qobj: Interaction Hamiltonian at time t.
    """

    if RWA_SL:
        E_field_RWA = E_pulse(t, laser)  # Combined electric field under RWA
        H_int_sl = -(
            SM_op.dag() * E_field_RWA + SM_op * np.conj(E_field_RWA)
        )  # RWA interaction Hamiltonian
    else:
        Dip_op = SM_op * SM_op.dag()
        E_field = Epsilon_pulse(t, laser)  # Combined electric field with carrier
        H_int_sl = -Dip_op * (
            E_field + np.conj(E_field)
        )  # Full interaction Hamiltonian

    return H_int_sl


# =============================
# THE PAPER SOLVER FUNCTIONS
# =============================
def matrix_ODE_paper(t: float, sim_oqs: SimulationModuleOQS) -> Qobj:
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


def _matrix_ODE_paper_1atom(t: float, sim_oqs: SimulationModuleOQS) -> Qobj:
    """
    Constructs the matrix L(t) for the equation
    drho_dt = L(t) · vec(rho),   QuTiP-kompatibel (column stacking).
    Uses gamma values from the provided system.

    Parameters:
        t (float): Time at which to evaluate the matrix.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object for the electric field.
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
    sim_oqs: SimulationModuleOQS,
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
def R_paper(sim_oqs: SimulationModuleOQS) -> Qobj:
    """Dispatches to the appropriate implementation based on N_atoms."""
    N_atoms = sim_oqs.system.N_atoms
    if N_atoms == 1:
        return _R_paper_1atom(sim_oqs)
    elif N_atoms == 2:
        return _R_paper_2atom(sim_oqs)
    else:
        raise ValueError("Only N_atoms=1 or 2 are supported.")


def _R_paper_1atom(sim_oqs: SimulationModuleOQS) -> Qobj:
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


def _R_paper_2atom(sim_oqs: SimulationModuleOQS) -> Qobj:
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
    Test function for SimulationModuleOQS class functionality.
    Tests H_int_sl method with different system configurations.
    """
    print("\n" + "=" * 80)
    print("TESTING simulation_config MODEL")

    # =============================
    # SETUP TEST CONFIGURATIONS
    # =============================

    ### Create simulation_config configuration
    sim_config = SimulationConfig(
        ODE_Solver="Paper_BR", RWA_SL=True, dt=0.1, t_coh=50.0
    )

    ### Create test pulse sequence
    from qspectro2d.core.bath_system.bath_class import BathSystem
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence

    test_pulse_seq = LaserPulseSequence.from_general_specs(
        pulse_peak_times=[0.0, 50.0],
        pulse_phases=[0.0, 1.57],  # 0, π/2
        pulse_freqs=[1.0, 1.0],
        pulse_fwhms=[15.0, 15.0],
        pulse_amplitudes=[0.05, 0.05],
        envelope_types="gaussian",
    )

    # =============================
    # TEST 1: Single atom system
    # =============================
    print("\n--- Test 1: Single Atom System ---")

    ### Create 1-atom system

    system_1 = AtomicSystem(N_atoms=1)
    bath = BathSystem()

    ### Create simulation_config model
    sim_model_1 = SimulationModuleOQS(
        simulation_config=sim_config,
        system=system_1,
        laser=test_pulse_seq,
        bath=bath,
    )
    try:
        ### Create 1-atom system

        system_1 = AtomicSystem(N_atoms=1)
        bath = BathSystem()

        ### Create simulation_config model
        sim_model_1 = SimulationModuleOQS(
            simulation_config=sim_config,
            system=system_1,
            laser=test_pulse_seq,
            bath=bath,
        )

        ### Test H_int_sl method
        test_time = 1.0
        H_int_1 = sim_model_1.H_int_sl(test_time)

        print(f"✓ Single atom H_int_sl created successfully")
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
        sim_model_2 = SimulationModuleOQS(
            simulation_config=sim_config,
            system=system_2,
            laser=test_pulse_seq,
            bath=bath,
        )

        ### Test H_int_sl method
        H_int_2 = sim_model_2.H_int_sl(test_time)

        print(f"✓ Two atom H_int_sl created successfully")
        print(f"  - Type: {type(H_int_2)}")
        print(f"  - Dimensions: {H_int_2.dims}")
        print(f"  - Is Hermitian: {H_int_2.isherm}")
        print(f"  - Matrix shape: {H_int_2.shape}")

    except Exception as e:
        print(f"✗ Error in two atom test: {e}")

    # =============================
    # TEST 3: Time evolution of H_int_sl
    # =============================
    print("\n--- Test 3: Time Evolution ---")

    try:
        test_times = np.linspace(0, 100, 5)  # Test at different times

        print(f"Testing H_int_sl at different times for single atom system:")
        for t in test_times:
            H_t = sim_model_1.H_int_sl(t)
            max_element = np.max(np.abs(H_t.full()))
            print(f"  t = {t:6.1f} fs: max|H_int_sl| = {max_element:.2e}")

    except Exception as e:
        print(f"✗ Error in time evolution test: {e}")

    # =============================
    # TEST 4: RWA vs non-RWA comparison
    # =============================
    print("\n--- Test 4: RWA vs Non-RWA ---")

    try:
        ### Create non-RWA configuration
        sim_config_no_rwa = SimulationConfig(
            ODE_Solver="Paper_BR", RWA_SL=False, dt=0.1, t_coh=100.0  # No RWA
        )

        sim_model_no_rwa = SimulationModuleOQS(
            simulation_config=sim_config_no_rwa,
            system=system_1,
            laser=test_pulse_seq,
            bath=bath,
        )

        ### Compare H_int_sl with and without RWA
        H_rwa = sim_model_1.H_int_sl(test_time)
        H_no_rwa = sim_model_no_rwa.H_int_sl(test_time)

        print(f"✓ RWA comparison successful")
        print(f"  - RWA H_int_sl max element: {np.max(np.abs(H_rwa.full())):.2e}")
        print(
            f"  - Non-RWA H_int_sl max element: {np.max(np.abs(H_no_rwa.full())):.2e}"
        )

    except Exception as e:
        print(f"✗ Error in RWA comparison test: {e}")

    # =============================
    # TEST 5: to_dict and from_dict methods
    # =============================
    print("\n--- Test 5: Dictionary Serialization/Deserialization ---")

    try:
        ### Test SimulationConfig to_dict and from_dict
        print("Testing SimulationConfig:")

        # Create original config
        original_config = SimulationConfig(
            ODE_Solver="Paper_BR",
            RWA_SL=True,
            dt=0.05,
            t_coh=150.0,
            t_wait=10.0,
            t_det_max=200.0,
            n_phases=8,
            n_freqs=3,
            max_workers=4,
            simulation_type="2d",
        )

        # Convert to dict and back
        config_dict = original_config.to_dict()
        reconstructed_config = SimulationConfig.from_dict(config_dict)

        # Compare attributes
        config_match = True
        for attr in [
            "ODE_Solver",
            "RWA_SL",
            "dt",
            "t_coh",
            "t_wait",
            "t_det_max",
            "n_phases",
            "n_freqs",
            "max_workers",
            "simulation_type",
        ]:
            if getattr(original_config, attr) != getattr(reconstructed_config, attr):
                config_match = False
                print(
                    f"  ✗ Mismatch in {attr}: {getattr(original_config, attr)} != {getattr(reconstructed_config, attr)}"
                )

        if config_match:
            print(f"  ✓ SimulationConfig serialization successful")
            print(f"    - Dictionary keys: {list(config_dict.keys())}")
            print(f"    - All attributes match after reconstruction")

        ### Test SimulationModuleOQS to_dict and from_dict
        print("\nTesting SimulationModuleOQS:")

        # Use the existing sim_model_1 for testing
        original_model = sim_model_1

        # Convert to dict and back
        model_dict = original_model.to_dict()
        reconstructed_model = SimulationModuleOQS.from_dict(model_dict)

        # Compare key attributes
        model_match = True
        comparison_results = []

        # Check simulation_config
        if (
            original_model.simulation_config.ODE_Solver
            != reconstructed_model.simulation_config.ODE_Solver
        ):
            model_match = False
            comparison_results.append(f"ODE_Solver mismatch")

        # Check system attributes
        if original_model.system.N_atoms != reconstructed_model.system.N_atoms:
            model_match = False
            comparison_results.append(f"N_atoms mismatch")

        if not np.allclose(
            original_model.system.freqs_cm, reconstructed_model.system.freqs_cm
        ):
            model_match = False
            comparison_results.append(f"freqs_cm mismatch")

        # Check laser attributes
        if len(original_model.laser.pulses) != len(reconstructed_model.laser.pulses):
            model_match = False
            comparison_results.append(f"pulse count mismatch")

        # Check bath attributes
        if abs(original_model.bath.gamma_0 - reconstructed_model.bath.gamma_0) > 1e-10:
            model_match = False
            comparison_results.append(f"gamma_0 mismatch")

        # Test H_int_sl method functionality
        test_time_dict = 5.0
        try:
            H_int_original = original_model.H_int_sl(test_time_dict)
            H_int_reconstructed = reconstructed_model.H_int_sl(test_time_dict)

            if not np.allclose(H_int_original.full(), H_int_reconstructed.full()):
                model_match = False
                comparison_results.append(f"H_int_sl method results differ")
            else:
                comparison_results.append(f"H_int_sl methods produce identical results")

        except Exception as e:
            model_match = False
            comparison_results.append(f"H_int_sl method error: {e}")

        if model_match:
            print(f"  ✓ SimulationModuleOQS serialization successful")
            print(f"    - Dictionary structure: simulation_config, system, laser, bath")
            print(f"    - All core attributes match after reconstruction")
            for result in comparison_results:
                print(f"    - {result}")
        else:
            print(f"  ✗ SimulationModuleOQS serialization issues:")
            for result in comparison_results:
                print(f"    - {result}")

        ### Test dictionary structure and content
        print(f"\nDictionary structure analysis:")
        print(f"  - SimulationConfig dict keys: {list(config_dict.keys())}")
        print(f"  - SimulationModuleOQS dict keys: {list(model_dict.keys())}")
        print(
            f"  - Dict size (SimulationModuleOQS): ~{len(str(model_dict))} characters"
        )

        ### Test with different system configurations
        print(f"\nTesting with 2-atom system:")

        # Create 2-atom system for testing
        system_2_test = AtomicSystem(
            N_atoms=2, freqs_cm=[16000.0, 16100.0], dip_moments=[1.0, 2.0]
        )

        sim_model_2_test = SimulationModuleOQS(
            simulation_config=sim_config,
            system=system_2_test,
            laser=test_pulse_seq,
            bath=bath,
        )

        # Test serialization for 2-atom system
        model_2_dict = sim_model_2_test.to_dict()
        reconstructed_model_2 = SimulationModuleOQS.from_dict(model_2_dict)

        ### Validate reconstructed 2-atom model with all essential attributes
        model_2_match = True
        comparison_2_results = []

        # Check simulation_config attributes
        config_attrs = ["ODE_Solver", "RWA_SL", "dt", "t_coh", "t_wait", "t_det_max"]
        for attr in config_attrs:
            orig_val = getattr(sim_model_2_test.simulation_config, attr)
            recon_val = getattr(reconstructed_model_2.simulation_config, attr)
            if orig_val != recon_val:
                model_2_match = False
                comparison_2_results.append(f"Config {attr}: {orig_val} != {recon_val}")

        # Check system attributes
        if sim_model_2_test.system.N_atoms != reconstructed_model_2.system.N_atoms:
            model_2_match = False
            comparison_2_results.append(
                f"N_atoms: {sim_model_2_test.system.N_atoms} != {reconstructed_model_2.system.N_atoms}"
            )

        if not np.allclose(
            sim_model_2_test.system.freqs_cm, reconstructed_model_2.system.freqs_cm
        ):
            model_2_match = False
            comparison_2_results.append(f"freqs_cm arrays differ")

        if not np.allclose(
            sim_model_2_test.system.dip_moments,
            reconstructed_model_2.system.dip_moments,
        ):
            model_2_match = False
            comparison_2_results.append(f"dip_moments arrays differ")

        # Check laser pulse attributes
        orig_pulses = sim_model_2_test.laser.pulses
        recon_pulses = reconstructed_model_2.laser.pulses
        if len(orig_pulses) != len(recon_pulses):
            model_2_match = False
            comparison_2_results.append(
                f"Pulse count: {len(orig_pulses)} != {len(recon_pulses)}"
            )
        else:
            for i, (orig_pulse, recon_pulse) in enumerate(
                zip(orig_pulses, recon_pulses)
            ):
                pulse_attrs = [
                    "pulse_peak_time",
                    "pulse_phase",
                    "pulse_freq",
                    "pulse_fwhm",
                    "pulse_amplitude",
                ]
                for attr in pulse_attrs:
                    if (
                        abs(getattr(orig_pulse, attr) - getattr(recon_pulse, attr))
                        > 1e-10
                    ):
                        model_2_match = False
                        comparison_2_results.append(f"Pulse {i} {attr} differs")

        # Check bath attributes
        bath_attrs = ["bath", "gamma_0", "Gamma", "cutoff_", "Temp"]
        for attr in bath_attrs:
            orig_val = getattr(sim_model_2_test.bath, attr)
            recon_val = getattr(reconstructed_model_2.bath, attr)

            # Handle different data types appropriately
            if isinstance(orig_val, str):
                # For string attributes like 'bath'
                if orig_val != recon_val:
                    model_2_match = False
                    comparison_2_results.append(
                        f"Bath {attr}: {orig_val} != {recon_val}"
                    )
            else:
                # For numeric attributes
                if abs(orig_val - recon_val) > 1e-10:
                    model_2_match = False
                    comparison_2_results.append(
                        f"Bath {attr}: {orig_val} != {recon_val}"
                    )

        # Test H_int_sl method functionality for 2-atom system
        test_time_2atom = 10.0
        try:
            H_int_orig_2 = sim_model_2_test.H_int_sl(test_time_2atom)
            H_int_recon_2 = reconstructed_model_2.H_int_sl(test_time_2atom)

            if not np.allclose(H_int_orig_2.full(), H_int_recon_2.full(), rtol=1e-12):
                model_2_match = False
                comparison_2_results.append(
                    f"H_int_sl methods produce different results"
                )
                max_diff = np.max(np.abs(H_int_orig_2.full() - H_int_recon_2.full()))
                comparison_2_results.append(f"Max H_int_sl difference: {max_diff:.2e}")
            else:
                comparison_2_results.append(
                    f"H_int_sl methods produce identical results"
                )

        except Exception as e:
            model_2_match = False
            comparison_2_results.append(f"H_int_sl method error: {e}")

        # Test dimensions and matrix properties
        try:
            if H_int_orig_2.dims != H_int_recon_2.dims:
                model_2_match = False
                comparison_2_results.append(
                    f"H_int_sl dimensions differ: {H_int_orig_2.dims} != {H_int_recon_2.dims}"
                )

            if H_int_orig_2.shape != H_int_recon_2.shape:
                model_2_match = False
                comparison_2_results.append(
                    f"H_int_sl shapes differ: {H_int_orig_2.shape} != {H_int_recon_2.shape}"
                )

        except Exception as e:
            model_2_match = False
            comparison_2_results.append(f"Dimension comparison error: {e}")

        # Test observable operators consistency
        try:
            orig_obs_ops = sim_model_2_test.observable_ops
            recon_obs_ops = reconstructed_model_2.observable_ops

            if len(orig_obs_ops) != len(recon_obs_ops):
                model_2_match = False
                comparison_2_results.append(f"Observable operators count differs")
            else:
                for i, (orig_op, recon_op) in enumerate(
                    zip(orig_obs_ops, recon_obs_ops)
                ):
                    if not np.allclose(orig_op.full(), recon_op.full(), rtol=1e-12):
                        model_2_match = False
                        comparison_2_results.append(f"Observable operator {i} differs")
                        break
                else:
                    comparison_2_results.append(f"All observable operators match")

        except Exception as e:
            model_2_match = False
            comparison_2_results.append(f"Observable operators error: {e}")

        # Print results for 2-atom system
        if model_2_match:
            print(f"  ✓ 2-atom model serialization successful")
            print(f"    - All essential attributes match after reconstruction")
            print(f"    - Dictionary size: ~{len(str(model_2_dict))} characters")
            for result in comparison_2_results:
                print(f"    - {result}")
        else:
            print(f"  ✗ 2-atom model serialization issues:")
            for result in comparison_2_results:
                print(f"    - {result}")

        ### Additional comprehensive tests
        print(f"\nComprehensive serialization tests:")

        # Test edge cases with different configurations
        test_configs = [
            SimulationConfig(ODE_Solver="ME", RWA_SL=True, dt=0.05, t_coh=25.0),
            SimulationConfig(ODE_Solver="BR", RWA_SL=False, dt=0.2, t_coh=200.0),
            SimulationConfig(ODE_Solver="Paper_eqs", RWA_SL=True, dt=0.1, t_coh=50.0),
        ]

        print(f"\n✓ to_dict and from_dict testing completed successfully!")

    except Exception as e:
        print(f"✗ Error in dictionary serialization test: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
    print("All tests completed.")
