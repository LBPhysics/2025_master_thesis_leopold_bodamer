# =============================
# FUNCTIONS for overlapping pulses
# =============================

from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Any, Optional
import matplotlib.pyplot as plt
from qutip.solver import Result
from qutip import *
import numpy as np


### Phase Cycling for Averaging
phases = [k * np.pi / 2 for k in range(4)]

# =============================
# SYSTEM PARAMETERS     (**changeable**)
# =============================


@dataclass
class SystemParameters:
    # =============================
    # Fundamental constants and system size
    # =============================
    hbar: float = 1.0
    N_atoms: int = 1  # Set the number of atoms only works for 1

    # =============================
    # Solver and model control
    # =============================
    ODE_Solver: str = (
        "Paper_BR"  # "Paper_eqs" (solve the EOMs from the paper) or "Paper_BR" do d/dt rho = -i/hbar * [H0 - Dip * E, rho] + R(rho)
    )
    RWA_laser: bool = (
        True  #  CAN ONLY HANDLE TRUE For MY solver (paper)   only valid for omega_laser ~ omega_A
    )

    # =============================
    # Energy and transition parameters in cm-1
    # =============================
    # values from the paper: [cm-1] linear wavenumber, usually used in spectroscopy
    Delta_cm: float = 200.0
    omega_A_cm: float = 16000.0
    mu_eg_cm: float = 1.0
    omega_laser_cm: float = 16000.0

    #   omega_B_cm: float = 16000.0
    #   mu_B_cm: float = 1.0

    # =============================
    # Laser field parameters
    # =============================
    E0: float = 0.1

    # =============================
    # Pulse and time grid parameters
    # =============================
    pulse_duration: float = 15.0  # in fs
    t_max: float = 15.0  # in fs
    fine_spacing: float = 1.0  # in fs

    # =============================
    # Decoherence and relaxation rates
    # =============================
    gamma_0: float = 1 / 300  # in fs-1
    T2: float = 100.0  # in fs

    # =============================
    # Quantum states (initialized as None, set by method)
    # =============================
    atom_g: Optional[Any] = None
    atom_e: Optional[Any] = None
    psi_ini: Optional[Any] = None

    def __post_init__(self):
        """
        Initialize quantum states for the given number of atoms.
        """
        self.init_quantum_states()

    def init_quantum_states(self):
        """
        Initialize the ground and excited states and the initial density matrix.
        Supports N_atoms = 1 or 2.
        """
        if self.N_atoms == 1:
            self.atom_g = basis(2, 0)
            self.atom_e = basis(2, 1)
            self.psi_ini = ket2dm(self.atom_g)
        elif self.N_atoms == 2:
            x = 1
            # TODO: Add a proper case for 2-atom initialization
        else:
            raise ValueError("Only N_atoms=1 or 2 are supported.")

    # =============================
    # Properties for all derived quantities
    # =============================

    @property
    def omega_A(self):  # in fs
        return self.omega_A_cm * 2.998 * 2 * np.pi * 10**-5

    @property
    def omega_laser(self):  # in fs
        return self.omega_laser_cm * 2.998 * 2 * np.pi * 10**-5

    @property
    def Delta(self):  # in fs
        return self.Delta_cm * 2.998 * 2 * np.pi * 10**-5

    @property
    def gamma_phi(self):
        return 1 / self.T2

    @property
    def Gamma(self):
        return self.gamma_0 / 2 + self.gamma_phi

    @property
    def rabi_0(self):
        return self.mu_eg_cm * self.E0 / self.hbar

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
            return self.mu_eg_cm * (self.atom_g * self.atom_e.dag()).unit()
        elif self.N_atoms == 2:
            g1, e1 = basis(2, 0), basis(2, 1)
            # TODO REALLY implement this
            return self.mu_eg_cm * tensor(g1, e1)
        else:
            raise ValueError("Only N_atoms=1 or 2 are supported.")

    @property
    def Dip_op(self):
        return self.SM_op + self.SM_op.dag()

    @property
    def e_ops_list(self):
        if self.N_atoms == 1:
            return [
                ket2dm(self.atom_g),
                self.atom_g * self.atom_e.dag(),
                self.atom_e * self.atom_g.dag(),
                ket2dm(self.atom_e),
            ]
        elif self.N_atoms == 2:
            g1, e1 = basis(2, 0), basis(2, 1)
            g2, e2 = basis(2, 0), basis(2, 1)
            return [
                ket2dm(tensor(g1, g2)),
                tensor(g1, g2) * tensor(e1, e2).dag(),
                tensor(e1, e2) * tensor(g1, g2).dag(),
                ket2dm(tensor(e1, e2)),
            ]
        else:
            raise ValueError("Only N_atoms=1 or 2 are supported.")

    @property
    def e_ops_labels(self):
        if self.N_atoms == 1:
            return ["gg", "ge", "eg", "ee"]
        elif self.N_atoms == 2:
            # TODO REALLY implement this
            return ["gg", "ge", "eg", "ee"]

    @property
    def c_ops_list(self):
        Gamma = self.Gamma
        gamma_phi = self.gamma_phi
        if self.N_atoms == 1:
            SM_op = self.SM_op
            return [
                np.sqrt(Gamma) * SM_op if Gamma > 0 else 0 * SM_op,
                (
                    np.sqrt(gamma_phi) * ket2dm(self.atom_e)
                    if gamma_phi > 0
                    else 0 * ket2dm(self.atom_e)
                ),
            ]
        elif self.N_atoms == 2:
            g1, e1 = basis(2, 0), basis(2, 1)
            # TODO REALLY implement this
            SM_op = self.SM_op
            return [
                np.sqrt(Gamma) * SM_op if Gamma > 0 else 0 * SM_op,
                (
                    np.sqrt(gamma_phi) * ket2dm(tensor(e1, e1))
                    if gamma_phi > 0
                    else 0 * ket2dm(tensor(e1, e1))
                ),
            ]
        else:
            raise ValueError("Only N_atoms=1 or 2 are supported.")

    def set_N_atoms(self, N_atoms: int):
        """
        Update the number of atoms and re-initialize quantum states.
        """
        if N_atoms not in [1, 2]:
            raise ValueError("Only N_atoms=1 or 2 are supported.")
        self.N_atoms = N_atoms
        self.init_quantum_states()

    def summary(self):
        """
        Print a summary of all parameters and selected derived quantities,
        skipping atom_g and atom_e, but including Dip_op and e_ops_labels.
        """
        print("=== SystemParameters Summary ===")
        for k, v in self.__dict__.items():
            if k not in ["hbar", "atom_g", "atom_e"]:
                print(f"{k:20}: {v}")
        print("--- Derived Quantities ---")
        for name in [
            "rabi_0",
            "delta_rabi",
            "rabi_gen",
            "t_prd",
            "fine_spacing",
            "t_max",
        ]:
            print(f"{name:20}: {getattr(self, name)}")
        print("\nDipole operator (Dip_op):")
        print(self.Dip_op)
        print("\nExpectation operator labels (e_ops_labels):")
        print(self.e_ops_labels)


# =============================
# END OF CLASS
# =============================

# This class is robust, modular, and ready for extension.
# - Only base parameters are stored as attributes.
# - All derived quantities are generated on demand.
# - N_atoms=1 and N_atoms=2 are both supported.
# - All logic is grouped and commented for clarity.


# =============================
# Define Pulse and PulseSequence classes for structured pulse handling
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


# ##########################
# independent of system
# ##########################
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
    if safe and output_dir is not None:
        if not os.path.isdir(output_dir):
            raise ValueError(f"Output directory {output_dir} does not exist.")
        filename_parts = [
            f"M={system.N_atoms}",
            f"mua={system.mu_eg_cm:.0f}",
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


def extend_time_tau_axes(ts, taus, data, pad_rows=(0, 0), pad_cols=(0, 0)):
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
