# FUNCTIONS for overlapping pulses
# =============================

from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.pyplot as plt
from qutip.solver import Result
from qutip import *
import numpy as np
import itertools
import os

# Import the outsourced plot settings
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# =============================
# DUMMY PARAMETERS FOR FUNCTION INDEPENDENCE
# =============================
# System and pulse parameters (dummy values)
N_atoms = 2  # Number of atoms
mu_eg = 1.0  # Dipole moment
E0 = 1.0  # Electric field amplitude
omega_A = 1.0  # Atomic transition frequency
omega_laser = 1.0  # Laser frequency
rabi_gen = 1.0  # Generalized Rabi frequency
rabi_0 = 1.0  # Rabi frequency
delta_rabi = 0.0  # Rabi detuning

# Pulse parameters (dummy arrays)
E_freqs = [1.0, 1.0, 1.0]  # Pulse frequencies
Delta_ts = [1.0, 1.0, 1.0]  # Pulse half-widths
E_amps = [1.0, 1.0, 1.0]  # Pulse amplitudes


# Dummy operators for a two-level system
SM_op = destroy(2)
Dip_op = sigmax()

# Dummy expectation operators and labels
e_ops_list = [sigmax(), sigmay()]
e_ops_labels = [r"$\sigma_x$", r"$\sigma_y$"]

# Dummy initial state
psi_ini = basis(2, 0)

# Dummy ODE solver name
ODE_Solver = "dummy_solver"

# Dummy RWA flag
RWA_laser = False


# Dummy function for applying RWA phase factors
def apply_RWA_phase_factors(state, time, omega):
    return state


# Dummy global time_cut
time_cut = float("inf")


# Dummy compute_pulse_evolution function
def compute_pulse_evolution(
    psi_ini, times, curr, prev=None, preprev=None, omega_at=None
):
    """Dummy function. Should be overwritten by the main script."""
    raise NotImplementedError(
        "This function must be overwritten with an N_atoms-specific implementation."
    )


############## Bath parameters ##############
global Boltzmann, hbar, Temp
Boltzmann = 1.0
hbar = 1.0

# Matplotlib settings according to LaTeX caption formatting
plt.rcParams.update(
    {
        "text.usetex": True,  # Enable LaTeX for text rendering
        "font.family": "serif",  # Use a serif font family
        "font.serif": "Palatino",  # or [], Set Palatino or standard latex font
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 20,  # Font size for general text
        "axes.titlesize": 20,  # Font size for axis titles
        "axes.labelsize": 20,  # Font size for axis labels
        "xtick.labelsize": 20,  # Font size for x-axis tick labels
        "ytick.labelsize": 20,  # Font size for y-axis tick labels
        "legend.fontsize": 20,  # Font size for legends
        "figure.figsize": [8, 6],  # Size of the plot (width x height)
        "figure.autolayout": True,  # Automatic layout adjustment
        "savefig.format": "svg",  # Default format for saving figures
        "figure.facecolor": "none",  # Make the figure face color transparent
        "axes.facecolor": "none",  # Make the axes face color transparent
        "savefig.transparent": True,  # Save figures with transparent background
    }
)

# Define the output directory relative to the main directory of the repository
repo_root_dir = os.path.abspath(
    os.path.join(os.getcwd(), "../../")
)  # Navigate to the main directory
output_dir = os.path.join(
    repo_root_dir, "figures", "figures_from_python"
)  # Define the output folder path
os.makedirs(output_dir, exist_ok=True)

# mpl.use("Agg")  # Use a non-interactive backend / SAVE figures to svg files
mpl.use(
    "module://matplotlib_inline.backend_inline"
)  # Use inline backend for Jupyter notebooks
# mpl.use('TkAgg')  # open each plot in interactive window


# ##########################
# independent of system
# ##########################
def plot_positive_color_map(
    datas,
    T_wait=np.inf,
    space="real",
    type="real",
    output_dir=None,
    ODE_Solver=None,
    positive=False,
    safe=False,
    use_custom_colormap=False,
    section=None,  # tuple of (x_min, x_max, y_min, y_max)
):
    """
    Create a color plot of 2D functional data for positive x and y values only.

    Parameters:
        datas (tuple): Tuple containing (x, y, data) where x and y are 1D arrays and data is a 2D array.
        T_wait (float): waiting time to include in plot title and file name.
        space (str): Either 'real' or 'freq' specifying the space of the data.
        type (str): Type of data ('real', 'imag', 'abs', or 'phase'). Used only if space="freq".
        positive (bool): Whether to use ONLY positive values of x and y.
        safe (bool): If True, saves the plot to a file.
        output_dir (str, optional): Directory to save the plot.

    Returns:
        None
    """
    # Ensure datas is a tuple with (x, y, data)
    if not isinstance(datas, tuple) or len(datas) != 3:
        raise ValueError("datas must be a tuple of (x, y, data)")

    x, y, data = datas  # corresponds to ts, taus, data

    # Validate that x, y, and data are numpy arrays
    x = np.real(x)
    y = np.real(y)
    data = np.array(data) / np.abs(data).max()

    # Check if data dimensions match x and y
    if data.shape[1] != len(x):
        raise ValueError(
            f"Length of x ({len(x)}) must match the number of columns in data ({data.shape[1]})."
        )
    if data.shape[0] != len(y):
        raise ValueError(
            f"Length of y ({len(y)}) must match the number of rows in data ({data.shape[0]})."
        )

    if space != "real" and space != "freq":
        raise ValueError("Invalid space. Must be 'real' or 'freq'.")
    elif space == "real":
        colormap = "viridis"
        title = r"$\text{Real space}$"
        x_title = r"$t_{\text{det}}$ [fs]"
        y_title = r"$\tau_{\text{coh}}$ [fs]"
    elif space == "freq":
        colormap = "plasma"
        title = r"$\text{Freq space}$"
        x_title = r"$\omega_{t_{\text{det}}}$ [$10^4$ cm$^{-1}$]"
        y_title = r"$\omega_{\tau_{\text{coh}}}$ [$10^4$ cm$^{-1}$]"

    if type != "real" and type != "imag" and type != "abs" and type != "phase":
        raise ValueError("Invalid Type. Must be 'real', 'imag', 'abs', or 'phase'.")
    elif type == "real":
        title += r"$\text{, Real 2D Spectrum}$"
        data = np.real(data).astype(np.float64)
    elif type == "imag":
        title += r"$\text{, Imag 2D Spectrum}$"
        data = np.imag(data).astype(np.float64)
    elif type == "abs":
        title += r"$\text{, Abs 2D Spectrum}$"
        data = np.abs(data).astype(np.float64)
        use_custom_colormap = False
    elif type == "phase":
        title += r"$\text{, Phase 2D Spectrum}$"
        data = np.angle(data).astype(np.float64)

    if T_wait != np.inf:
        title += rf"$\ \text{{at }} T = {T_wait:.2f}$"

    # Filter for positive x and y values if requested
    if positive:
        # Ensure the data array has even dimensions
        rows, cols = data.shape
        if rows % 2 != 0:
            data = data[:-1, :]  # Remove the last row if the number of rows is odd
        if cols % 2 != 0:
            data = data[
                :, :-1
            ]  # Remove the last column if the number of columns is odd

        # Recalculate the midpoints after adjusting dimensions
        mid_x = len(x) // 2
        mid_y = len(y) // 2

        # Extract quadrants
        q1 = data[mid_y:, mid_x:]  # First quadrant
        q2 = data[mid_y:, :mid_x]  # Second quadrant
        q3 = data[:mid_y, :mid_x]  # Third quadrant
        q4 = data[:mid_y, mid_x:]  # Fourth quadrant

        # Average the quadrants
        averaged_data = (
            q1 + np.flip(q3, axis=(0, 1))
        ) / 2  #  np.flip(q2, axis=1) +  + np.flip(q4, axis=0)

        # Update x, y, and data to only include the first quadrant
        x = x[mid_x:]
        y = y[mid_y:]
        data = averaged_data  # plot only a small portion of the data
    if section is not None:
        x_min, x_max, y_min, y_max = section
        x_indices = np.where((x >= x_min) & (x <= x_max))[0]
        y_indices = np.where((y >= y_min) & (y <= y_max))[0]
        # Ensure indices are within bounds
        x_indices = x_indices[x_indices < data.shape[1]]
        y_indices = y_indices[y_indices < data.shape[0]]
        data = data[np.ix_(y_indices, x_indices)]
        x = x[x_indices]
        y = y[y_indices]

    norm = None
    if use_custom_colormap:
        vmin = np.min(data)
        vmax = np.max(data)
        vcenter = 0

        # Create a colormap that forces zero values to white
        cmap = plt.get_cmap("bwr")
        colors = cmap(np.linspace(0, 1, 256))
        mid = 128  # Middle index
        colors[mid] = [1, 1, 1, 1]  # Pure white in RGBA
        colormap = LinearSegmentedColormap.from_list("white_centered", colors)

        # Ensure vmin, vcenter, and vmax are in ascending order
        if vmin < vcenter < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            print(
                f"Warning: Cannot use TwoSlopeNorm with vmin={vmin}, vcenter={vcenter}, vmax={vmax}. Using default normalization."
            )
    # Set the colorbar label
    cbarlabel = r"$\propto E_{\text{out}} / E_{0}$"

    # Plot the color map
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

    # Save the plot if safe is True and output_dir is provided
    if safe and output_dir is not None:
        assert os.path.isdir(
            output_dir
        ), f"Output directory {output_dir} does not exist."

        filename_parts = [
            f"M={N_atoms}",
            f"mua={mu_eg:.0f}",
            f"E0={E0:.2e}",
            f"wa={omega_A:.2f}",
            f"wL={omega_laser / omega_A:.1f}wa",
            f"rabigen={rabi_gen:.2f}= sqrt({rabi_0:.2f}^2+{delta_rabi:.2f}^2)",
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


def get_expect_vals_with_RWA(states, times, e_ops=e_ops_list, omega=omega_laser):
    """
    Calculate the expectation values in the result with RWA phase factors.

    Parameters:
        states= data.states (where data = qutip.Result): Results of the pulse evolution.
        times (list): Time points at which the expectation values are calculated.
        e_ops (list): the operators for which the expectation values are calculated
        omega (float): omega_laser (float): Frequency of the laser.
    Returns:
        list of lists: Expectation values for each operator of len(states).
    """
    if RWA_laser:
        # Apply RWA phase factors to each state
        states = [
            apply_RWA_phase_factors(state, time, omega)
            for state, time in zip(states, times)
        ]

    updated_expects = [np.real(expect(states, e_op)) for e_op in e_ops]
    return updated_expects


############## PULSE FUNCTIONS
def get_pulse_args(curr, prev=None, preprev=None):
    """
    Generate pulse arguments for multiple pulses based on the current time, phase, and pulse index.

    Parameters:
        curr = (t0_cur, phi_curr) (tuple of floats): starting time and phase kick of the current pulse
        prev = (t0_prev, phi_prev)  (tuple of floats, optional): Start time and phase kick of the previous pulse. Defaults to None.
        preprev = (t0_preprev, phi_preprev) (tuple of floats, optional): Start time and phase kick of the first pulse. Defaults to None.

    Returns:
        dict: Dictionary containing pulse parameters for the current and potentially previous.
    """
    # i (int): Index of the current pulse (0, 1, or 2).
    args = []
    i = 0

    if prev != None:
        i = 1
        t0_prev, phi_prev = prev
        if preprev != None:
            i = 2
            t0_preprev, phi_preprev = preprev
            args.append(
                {
                    "pulse_phase": phi_preprev,
                    "pulse_start_time": t0_preprev,
                    "pulse_freq": E_freqs[i - 2],
                    "pulse_half_width": Delta_ts[i - 2],
                    "pulse_amplitude": E_amps[i - 2],
                }
            )
        args.append(
            {
                "pulse_phase": phi_prev,
                "pulse_start_time": t0_prev,
                "pulse_freq": E_freqs[i - 1],
                "pulse_half_width": Delta_ts[i - 1],
                "pulse_amplitude": E_amps[i - 1],
            }
        )

    t0_curr, phi_curr = curr
    args.append(
        {
            "pulse_phase": phi_curr,
            "pulse_start_time": t0_curr,
            "pulse_freq": E_freqs[i],
            "pulse_half_width": Delta_ts[i],
            "pulse_amplitude": E_amps[i],
        }
    )
    return {"pulses": args}


def pulse_envelope(t, args):
    """
    Calculate the combined envelope of multiple pulses.

    Parameters:
        t (float): Time at which the envelope is evaluated.
        args (dict): Dictionary containing a list of pulse parameters under the key "pulses".

    Returns:
        float: Combined envelope value at time t.
    """
    envelope = 0.0
    for params in args["pulses"]:
        t0 = params["pulse_start_time"]  # Start time of the pulse
        Delta_width = params["pulse_half_width"]  # Half-width of the pulse

        # Add the envelope contribution if within the pulse range
        if t0 <= t <= t0 + 2 * Delta_width:
            envelope += (
                np.cos(np.pi * (t - (t0 + Delta_width)) / (2 * Delta_width)) ** 2
            )
    # Example: check for zero or negative pulse width
    if Delta_width == 0:
        raise ValueError("Pulse half-width Delta_width is zero!")

    # Example: check for NaN/inf in envelope
    if not np.isfinite(envelope):
        raise ValueError(f"Envelope is not finite at t={t}, args={args}")
    return envelope


def E_pulse(t, args):
    """
    Define a time-dependent electric field pulse with combined envelopes.

    Parameters:
        t (float): Time at which the electric field is evaluated.
        args (dict): Dictionary containing pulse parameters:
            - "pulses": List of dictionaries for each pulse, each containing:
                - "pulse_start_time": Start time of the pulse.
                - "pulse_half_width": Half-width of the pulse.
                - "pulse_phase": Phase of the pulse.
                - "pulse_amplitude": Amplitude of the pulse.

    Returns:
        complex: Combined electric field value at time t.
    """
    E_total = 0.0
    for pulse in args["pulses"]:
        phi = pulse["pulse_phase"]
        E0 = pulse["pulse_amplitude"]
        # Calculate the envelope for this pulse
        Env = pulse_envelope(t, {"pulses": [pulse]})
        # Calculate the electric field for this pulse
        E = E0 * Env * np.exp(-1j * phi)  # Phase factor
        E_total += E  # Add the contribution of this pulse
    return E_total / 2


def Epsilon_pulse(t, args):
    """
    Define a time-dependent electric field pulse with combined envelopes and rotation,
    using the existing pulse_envelope() function.

    Parameters:
        t (float): Time at which the electric field is evaluated.
        args (dict): Dictionary containing pulse parameters:
            - "pulses": List of dicts for each pulse, each containing:
                - "pulse_start_time"
                - "pulse_half_width"
                - "pulse_freq"
                - "pulse_phase"
                - "pulse_amplitude"

    Returns:
        complex: Combined electric field value at time t.
    """
    omega = args["pulses"][0]["pulse_freq"]
    E_total = E_pulse(t, args) * np.exp(-1j * omega * t)
    return E_total


def plot_pulse_envelope(times, args, ax=None):
    """
    Plot the combined pulse envelope over time for up to three pulses.

    Parameters:
        times (array): Array of time values.
        args (dict): Dictionary containing a list of pulse parameters under the key "pulses".
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        ax (matplotlib.axes.Axes): Axes object with the plot.
    """
    # Calculate the combined envelope over time
    envelope = [pulse_envelope(t, args) for t in times]

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
    for idx, params in enumerate(args["pulses"][:3]):  # Up to 3 pulses
        t0 = params["pulse_start_time"]
        Delta_width = params["pulse_half_width"]

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


def El_field_3_pulses(times, args, f=Epsilon_pulse):
    """
    Calculate the combined electric field and time ranges for three pulses.

    Parameters:
        times (array): Time range for the three pulses.
        args (dict): all arguments of the three pulses
    Returns:
        tuple: Combined time range (times_plot) and electric field (E_total).
    """
    ### Calculate the electric fields for each time range

    E0 = args["pulses"][0]["pulse_amplitude"]
    E = np.array([f(t, args) for t in times])
    if f != pulse_envelope:
        E /= 0.5 * E0
    return E


def get_tau_cohs_and_t_dets_for_T_wait(times, T_wait):
    """
    Calculate the time arrays for tau_coh and t_det based on the waiting time T_wait and spacing.

    Parameters:
        T_wait (float): Waiting time.
        spacing (float): Time spacing.
        end (float): the final time

    Returns:
        tuple: Arrays for coherence and detections times tau_coh and t_det.
        both have the same len(_)
    """
    ### Calculate the first and last entries for tau_coh times
    if times.size != 0:
        spacing = times[1] - times[0]
        t_max = times[-1]
    else:
        raise (ValueError)

    if T_wait > t_max:
        print("there are no tau_coh / t_det vals because T_wait >= t_max")
    if T_wait == t_max:
        return np.array([0]), np.array([0])
    time_0 = 0  # pulses can overlap!!
    time_max = t_max - T_wait
    times_ = np.arange(time_0, time_max, spacing)  # time_max is not included!

    return times_, times_


def H_int(t, args):
    """
    Define the interaction Hamiltonian for the system with multiple pulses.

    Parameters:
        t (float): Time at which the interaction Hamiltonian is evaluated.
        args (dict): Dictionary containing pulse parameters:
            - "pulses": List of dictionaries for each pulse.

    Returns:
        Qobj: Interaction Hamiltonian at time t.
    """
    # Calculate the electric field based on whether RWA is applied
    if RWA_laser:
        E_field = E_pulse(t, args)  # Use the combined electric field under RWA
        H_int = -(
            SM_op.dag() * E_field + SM_op * np.conj(E_field)
        )  # Interaction Hamiltonian in RWA
    else:
        E_field = Epsilon_pulse(
            t, args
        )  # Use the combined electric field with rotation
        H_int = -Dip_op * (E_field + np.conj(E_field))  # Full interaction Hamiltonian

    return H_int


def check_the_solver(psi_ini, times, e_ops_list):
    """
    Checks the solver within the compute_pulse_evolution function
    with the provided psi_ini, times, and e_ops_list.

    Parameters:
        psi_ini (qutip.Qobj): Initial quantum state.
        times (numpy.ndarray): Time array for the evolution.
        e_ops_list (list of qutip.Qobj): List of expectation value operators.

    Returns:
        None: Prints the results of the checks.  Raises exceptions on errors.
    """
    print(f"Checking '{ODE_Solver}' solver ")

    # 1. Input Validation
    # ----------------------
    if not isinstance(psi_ini, qutip.Qobj):
        raise TypeError("psi_ini must be a qutip.Qobj")
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be a numpy.ndarray")
    if not isinstance(e_ops_list, list) or not all(
        isinstance(op, qutip.Qobj) for op in e_ops_list
    ):
        raise TypeError("e_ops_list must be a list of qutip.Qobj")

    if len(times) < 2:
        raise ValueError("times must have at least two elements")

    # 2. Call the function to be checked
    # ----------------------------------
    phi_0 = np.pi / 2
    phi_1 = np.pi / 4
    t_start_0 = 0
    t_start_1 = 10
    t_start_2 = 15
    pulse_0 = (t_start_0, phi_0)
    pulse_1 = (t_start_1, phi_1)
    phi_2 = 0
    pulse_f = (t_start_2, phi_2)
    result = compute_pulse_evolution(
        psi_ini, times, curr=pulse_f, prev=pulse_1, preprev=pulse_0
    )

    # 3. Check the result (basic checks)
    # ----------------------------------
    if not isinstance(result, qutip.Result):
        raise TypeError("Result must be a qutip.Result object")

    # Add more specific checks as needed for your implementation
    if result.times != list(times):
        raise ValueError("Result times do not match input times")

    if len(result.states) != len(times):
        raise ValueError("Number of output states does not match number of time points")

    # 4. Check if density matrix is Hermitian, positive, and trace-preserving
    # ------------------------------------------------------
    strg = ""
    global time_cut
    for index, state in enumerate(result.states):
        if RWA_laser:
            state = apply_RWA_phase_factors(state, times[index], omega=omega_laser)
        time = times[index]
        if not state.isherm:
            strg += f"Density matrix is not Hermitian after t = {time}.\n"
            print(state)
        eigvals = state.eigenenergies()
        # print(eigvals)
        time_cut = (
            np.inf
        )  # time after which the evolution becomes unphysical, to be decided in check_the_solver
        if not np.all(
            eigvals >= -1e-3
        ):  # allow for small numerical negative eigenvalues
            strg += f"Density matrix is not positive semidefinite after t = {time}: The lowest eigenvalue is {eigvals.min()}.\n"
            time_cut = time
        if not np.isclose(state.tr(), 1.0):
            strg += f"Density matrix is not trace-preserving after t = {time}: The trace is {state.tr()}.\n "
            time_cut = time
        if strg:
            strg += "Adjust your parameters!"
            print(strg)  # raise ValueError(strg)
            break
    else:
        print(
            "Checks passed. solver appears to be called correctly, and density matrix remains Hermitian and positive."
        )

    return result, time_cut


############## GET 2D DATA


def compute_two_dimensional_polarization(
    T_wait, phi_0, phi_1, times, omega_at=omega_A, plot_example=False
):
    """
    Compute the two-dimensional polarization for a given waiting time (T_wait) and
    the phases of the first and second pulses (phi_0, phi_1).

    Parameters:
        T_wait (float): Waiting time between the second and third pulses.
        phi_0 (float): Phase of the first pulse.
        phi_1 (float): Phase of the second pulse.

    Returns:
        tuple: A tuple containing:
            - tau_coh_vals (numpy.ndarray): Array of coherence times (tau_coh).
            - t_det_vals (numpy.ndarray): Array of detection times (t_det).
            - data (numpy.ndarray): 2D array of polarization values.
    """

    # get the symmetric times, tau_coh, t_det
    tau_coh_vals, t_det_vals = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait=T_wait)

    # initialize the time domain Spectroscopy data tr(Dip_op * rho_final(tau_coh, t_det))
    data = np.zeros(
        (len(tau_coh_vals), len(t_det_vals)), dtype=np.complex64
    )  # might get uncontrollable big!TODO
    # MAYBE USE FILES: data = np.memmap('spectroscopy_data.dat', dtype='complex128', mode='w+', shape=(len(tau_coh_vals), len(t_det_vals)))

    idx_start_0 = 0
    t_start_0 = times[idx_start_0]  # ALWAYS the starting point of the first pulse
    idx_end_0 = np.abs(times - (Delta_ts[0])).argmin()
    # only make the necessary steps (don't calculate too many states that we don't need)
    # latest time at which the next pulse starts:
    idx_start_1_max = np.abs(times - (tau_coh_vals[-1] - Delta_ts[1])).argmin()
    times_0 = times[
        : idx_start_1_max + 1
    ]  # definetly not empty except for when T_wait >= t_max
    if times_0.size == 0:
        times_0 = times[idx_start_0 : idx_end_0 + 1]
    # calculate the evolution of the first pulse in the desired range for tau_coh

    pulse_0 = (t_start_0, phi_0)

    data_0 = compute_pulse_evolution(psi_ini, times_0, curr=pulse_0, omega_at=omega_at)
    # for every tau_coh value -> calculate the next two pulses          # print(f"(T_wait): {T_wait}, (t_max): {t_max}")
    for tau_idx, tau_coh in enumerate(tau_coh_vals):  # print(tau_coh, tau_idx)
        # Calculate the start and end indices for the second pulse
        idx_start_1 = np.abs(times - (tau_coh - Delta_ts[1])).argmin()
        t_start_1 = times[
            idx_start_1
        ]  # Start time of the second pulse        # idx_end_1   = np.abs(times - (t_start_1 + 2 * Delta_ts[1])).argmin() # end of the second pulse
        idx_end_1 = np.abs(
            times - (tau_coh + Delta_ts[1])
        ).argmin()  # index at which the second pulse ends
        # Take the state (after / also during) the first pulse and evolve it with the second (and potentially overlaped first) pulse
        rho_1 = data_0.states[idx_start_1]

        # select range  ->  to reduce computation time
        idx_start_2 = np.abs(times - (tau_coh + T_wait - Delta_ts[2])).argmin()
        idx_end_2 = np.abs(times - (tau_coh + T_wait + Delta_ts[2])).argmin()
        t_start_2 = times[idx_start_2]  # the time at which the third pulse starts
        idx_end_2 = np.abs(
            times - (tau_coh + T_wait + Delta_ts[2])
        ).argmin()  # end of the third pulse
        # idx_start_2_0 = np.abs(times - (T_wait - Delta_ts[2])).argmin() # the first time at which the third pulse starts

        times_1 = times[
            idx_start_1 : idx_start_2 + 1
        ]  # like this: also take the overlap into account;

        if times_1.size == 0:
            times_1 = times[idx_start_1 : idx_start_1 + 1]
        # Handle overlapping pulses: If the second pulse starts before the first pulse ends, combine their contributions
        pulse_1 = (t_start_1, phi_1)
        data_1 = compute_pulse_evolution(
            rho_1, times_1, curr=pulse_1, prev=pulse_0, omega_at=omega_at
        )

        idx_start_2_in_times_1 = np.abs(times_1 - (t_start_2)).argmin()

        rho_2 = data_1.states[
            idx_start_2_in_times_1
        ]  # == state where the third pulse starts

        times_2 = times[
            idx_start_2:
        ]  # the rest of the evolution (third pulse, potentially overlapped with previouses) # can be empty, if tau_coh + T_wait >= t_max
        if times_2.size == 0:
            times_2 = times[idx_start_2 : idx_start_2 + 1]
        # If the second pulse starts before the first pulse ends, combine their contributions
        phi_2 = 0  # FIXED PHASE!
        pulse_f = (t_start_2, phi_2)

        data_f = compute_pulse_evolution(
            rho_2,
            times_2,
            curr=pulse_f,
            prev=pulse_1,
            preprev=pulse_0,
            omega_at=omega_at,
        )

        for t_idx, t_det in enumerate(t_det_vals):

            # store the data for the case
            if t_idx + tau_idx < len(tau_coh_vals):
                actual_det_time = t_start_2 + Delta_ts[2] + t_det
                t_idx_in_times_2 = np.abs(
                    times_2 - (actual_det_time)
                ).argmin()  # detection time index in times_2

                # only if we are still in the physical regime
                if actual_det_time < time_cut:
                    rho_f = data_f.states[t_idx_in_times_2]

                    if RWA_laser:
                        rho_f = apply_RWA_phase_factors(
                            rho_f, times_2[t_idx_in_times_2], omega=omega_laser
                        )
                    value = expect(Dip_op, rho_f)
                    data[tau_idx, t_idx] = np.real(
                        value
                    )  # Rolarization P(t)= ("Real")(<Dip_op>)  # print(f"(t_det): {t_det}, (tau_coh): {tau_coh}, (data): {value}, Time index in times_2: {t_idx_in_times_2}")

                    # make one plot for this case
                    if (
                        t_idx == 0
                        and tau_idx == len(tau_coh_vals) // 3
                        and plot_example
                    ):
                        # Prepare the data for plotting
                        """print("times_0: ", times_0[0], times_0[1],"...",  times_0[-1], "len", len(times_0))
                        print("times_1: ", times_1[0])# , times_1[1],"...",  times_1[-1], "len", len(times_1))
                        print("times_2: ", times_2[0], times_2[1],"...",  times_2[-1], "len", len(times_2))
                        print(
                            "idx_start_0:", idx_start_0, "idx_end_0:", idx_end_0, "\n"
                            "idx_start_1:", idx_start_1, "idx_end_1:", idx_end_1, "idx_start_1_max", idx_start_1_max, "\n"
                            "idx_start_2:", idx_start_2, "idx_end_2:", idx_end_2
                        )"""
                        data_1_expects = get_expect_vals_with_RWA(
                            data_0.states[: idx_start_1 + 1],
                            data_0.times[: idx_start_1 + 1],
                            e_ops=e_ops_list + [Dip_op],
                            omega=omega_laser,
                        )
                        data_2_expects = get_expect_vals_with_RWA(
                            data_1.states[: idx_start_2_in_times_1 + 1],
                            data_1.times[: idx_start_2_in_times_1 + 1],
                            e_ops=e_ops_list + [Dip_op],
                            omega=omega_laser,
                        )
                        data_f_expects = get_expect_vals_with_RWA(
                            data_f.states,
                            data_f.times,
                            e_ops=e_ops_list + [Dip_op],
                            omega=omega_laser,
                        )

                        data_expectations = [
                            np.concatenate(
                                [
                                    data_1_expects[idx],
                                    data_2_expects[idx],
                                    data_f_expects[idx],
                                ]
                            )
                            for idx in range(len(e_ops_list) + 1)
                        ]

                        """print(f"Shape of data_1_expects: {data_1_expects[0].shape}")
                        print(f"Shape of data_2_expects: {data_2_expects[0].shape}")
                        print(f"Shape of data_f_expects: {data_f_expects[0].shape}")
                        print(f"Shape of data_expectations[0]: {data_expectations[0].shape}")
                        print(f"Shape of times_0[:idx_start_1+1]: {times_0[:idx_start_1+1].shape}")
                        print(f"Shape of times_1: {times_1.shape}")
                        print(f"Shape of times_2: {times_2.shape}")"""
                        # Plot the evolution of the electric field and expectation values with overlapping pulses

                        args = get_pulse_args(
                            curr=pulse_f, prev=pulse_1, preprev=pulse_0
                        )
                        Plot_example_evo(
                            times_0[: idx_start_1 + 1],
                            times_1,
                            times_2,
                            data_expectations,
                            args,
                            tau_coh,
                            T_wait,
                            RWA_laser,
                        )

    return t_det_vals, tau_coh_vals, 1j * data  # because E ~ i*P


############## Plot one example of the evolution


def Plot_example_evo(
    times_0, times_1, times_2, datas, args, tau_coh, T_wait, RWA_laser
):
    """
    Plot the evolution of the electric field and expectation values for a given tau_coh and T_wait.

    Parameters:
        times_0, times_1, times_2 (array): Time ranges for the three pulses.
        args (dict): all arguments of the three pulses
        tau_coh (float): Coherence time.
        T_wait (float): Waiting time.
        phi_0 (float): Phase of the first pulse.
        phi_1 (float): Phase of the second pulse.
        data_0, data_1, data_f (qutip.Result): Results of the three pulses.
        idx_tau_coh (int): Indices for the coherence time.

    Returns:
        None
    """
    times_plot = np.concatenate([times_0, times_1, times_2])
    f = Epsilon_pulse
    if RWA_laser:
        f = E_pulse
    E_total = El_field_3_pulses(times_plot, args, f)
    ### Create subplots
    fig, axs = plt.subplots(len(datas) + 1, 1, figsize=(15, 10), sharex=True)

    ### Plot the electric field
    axs[0].plot(times_plot, np.real(E_total), color="C0", linestyle="solid", label="Re")
    axs[0].plot(
        times_plot, np.imag(E_total), color="C1", linestyle="dashed", label="Im"
    )
    axs[0].set_ylabel(r"$E(t) / E_0$")  # normalize to the first pulse amplitude
    axs[0].axvline(
        times_0[0] + Delta_ts[0], color="C1", linestyle="dashed"
    )  # Mark Pulse 1
    axs[0].axvline(
        times_1[0] + Delta_ts[1], color="C2", linestyle="dashdot"
    )  # Mark Pulse 2
    axs[0].axvline(
        times_2[0] + Delta_ts[2], color="C3", linestyle="dotted"
    )  # Mark Pulse 3

    axs[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))  # OLD .legend()

    ### Initialize an empty list to store all the data for plotting
    for idx, data in enumerate(datas):
        label = (
            r"Re$\langle$" + e_ops_labels[idx] + r"$\rangle$"
            if idx < len(e_ops_labels)
            else r"Re$\langle \mu \rangle$"
        )
        axs[idx + 1].plot(times_plot, data, color=f"C{idx % 10}", linestyle="solid")
        axs[idx + 1].set_ylabel(label)
        axs[idx + 1].axvline(
            times_0[0] + Delta_ts[0],
            color="C1",
            linestyle="dashed",
            label=f"Pulse 1 at{times_0[0] + Delta_ts[0]:.2f}",
        )  # Mark Pulse 1
        axs[idx + 1].axvline(
            times_1[0] + Delta_ts[1],
            color="C2",
            linestyle="dashdot",
            label=f"Pulse 2 at{times_1[0] + Delta_ts[1]:.2f}",
        )  # Mark Pulse 2
        axs[idx + 1].axvline(
            times_2[0] + Delta_ts[2],
            color="C3",
            linestyle="dotted",
            label=f"Pulse 3 at{times_2[0] + Delta_ts[2]:.2f}",
        )  # Mark Pulse 3

    axs[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5))  # OLD .legend()
    axs[1].plot(
        [], [], label=str(ODE_Solver), color="C0", linestyle="solid"
    )  # Add a dummy plot for the legend

    fig.suptitle(rf"$\tau = {tau_coh:.2f}, T = {T_wait:.2f}$")
    axs[-1].set_xlabel(r"$t \, [\text{fs}]$")
    plt.show()


############## CONVERT TO SPECTRUM W. Fourier


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
    s2d = np.fft.fftshift(np.fft.fft(np.fft.fft(np.imag(data), axis=1), axis=0))

    return nu_ts, nu_taus, s2d


#######################


def sigma(E, Delta, E0=0):
    """
    Computes the function σ(E - E0) as shown in the image.

    Parameters:
    - E: Energy value or array of values
    - Delta: Full Width at Half Maximum (FWHM)
    - E0: Center energy (default is 0)

    Returns:
    - σ(E - E0): The value of the function at E
    """
    ln2 = np.log(2)  # natural logarithm of 2
    prefactor = np.sqrt(ln2) / (np.sqrt(np.pi) * (Delta / 2))  # normalization factor
    exponent = -((E - E0) ** 2 / ((Delta / 2) ** 2)) * ln2  # Gaussian exponent
    return prefactor * np.exp(exponent)


"""# =============================
# PLOT THE FUNCTION
# =============================

### Generate values for plotting
E_values = np.linspace(-10, 10, 500) # energy range
E0       = omega_A                       # center energy

### Calculate sigma values
sigma_values = sigma(E_values, Delta, E0)

### Plot the function
plt.figure(figsize=(10, 6))
plt.plot(E_values, sigma_values, label=rf"$\Delta = {Delta}, E_0 = {E0}$", color='C0', linestyle='solid')
plt.title(r"$\sigma(E - E_0)$ Function")
plt.xlabel(r"$E$")
plt.ylabel(r"$\sigma(E - E_0)$")
plt.legend()
plt.tight_layout()
plt.show()"""


def sample_from_sigma(N, Delta, E0, E_range=10):
    """
    Samples N values from a normalized σ(E) distribution using rejection sampling.

    Parameters:
    - N: number of samples
    - Delta: FWHM of the sigma distribution
    - E0: center of the distribution
    - E_range: range (in units of Delta) to sample from around E0

    Returns:
    - omega_ats: array of sampled omega values
    """
    # Define the sampling range
    E_min = E0 - E_range * Delta
    E_max = E0 + E_range * Delta
    E_vals = np.linspace(E_min, E_max, 10000)
    sigma_vals = sigma(E_vals, Delta, E0)

    # Normalize sigma to use it as a probability density
    sigma_max = np.max(sigma_vals)

    samples = []
    while len(samples) < N:
        E_try = np.random.uniform(E_min, E_max)
        y_try = np.random.uniform(0, sigma_max)
        if y_try < sigma(E_try, Delta, E0):
            samples.append(E_try)
    return np.array(samples)


def compute_many_polarizations(
    T_wait,
    phi_0,
    phi_1,
    N,
    E0=omega_A,
    Delta=omega_A / 20,
    times=np.array([]),
    **kwargs,
):
    """
    Calls compute_two_dimensional_polarization N times for different omega_ats sampled from σ.

    Parameters:
    - N: number of simulations
    - Delta: FWHM for sigma distribution
    - phi_0, phi_1: laser phases
    - T_wait: delay
    - E0: center of the frequency distribution
    - **kwargs: any additional arguments for compute_two_dimensional_polarization

    Returns:
    - results: list of (t_det_vals, tau_coh_vals, data) tuples
    - omega_ats: list of sampled omega_ats used
    """
    omega_ats = sample_from_sigma(N, Delta, E0)
    results = []

    for omega in omega_ats:
        data = compute_two_dimensional_polarization(
            T_wait=T_wait,
            phi_0=phi_0,
            phi_1=phi_1,
            times=times,
            omega_at=omega,
            **kwargs,
        )
        results.append(data[2])
    data_avg = np.mean(results, axis=0)
    # for the last one extract the times
    t_det_vals = data[0]
    tau_coh_vals = data[1]
    return t_det_vals, tau_coh_vals, data_avg, omega_ats


"""# Test the sampling function
Delta_test = Delta
E0_test = omega_A
samples = sample_from_sigma(100, Delta=Delta_test, E0=E0_test)

# Plot the results
E_vals = np.linspace(E0_test - 5 * Delta_test, E0_test + 5 * Delta_test, 1000)
sigma_vals = sigma(E_vals, Delta_test, E0_test)

plt.figure(figsize=(8, 4))
plt.hist(samples, bins=100, density=True, alpha=0.6, label='Sampled histogram')
plt.plot(E_vals, sigma_vals / np.trapz(sigma_vals, E_vals), 'r-', label=r'Normalized $\sigma$(E)')
plt.xlabel('E')
plt.ylabel('Probability Density')
plt.title(r'Rejection Sampling from $\sigma$(E)')
plt.legend()
plt.show()"""


####################### PLOT 1 in I


def compute_fixed_tau_T(tau_coh, T_wait, phi_0, phi_1, times):
    """
    Compute the data for a fixed tau_coh and T_wait.

    Parameters:
        tau_coh (float): Coherence time.
        T_wait (float): Waiting time.
        phi_0 (float): Phase of the first pulse.
        phi_1 (float): Phase of the second pulse.

    Returns:
        tuple: Time values and corresponding data.
    """

    idx_start_0 = 0
    t_start_0 = times[idx_start_0]  # ALWAYS the starting point of the first pulse
    idx_end_0 = np.abs(times - (Delta_ts[0])).argmin()
    idx_start_1 = np.abs(times - (tau_coh - Delta_ts[1])).argmin()
    t_start_1 = times[idx_start_1]  # Start time of the second pulse

    times_0 = times[
        : idx_start_1 + 1
    ]  # definetly not empty except for when T_wait >= t_max
    if times_0.size == 0:
        times_0 = times[: idx_end_0 + 1]
    # calculate the evolution of the first pulse in the desired range for tau_coh

    pulse_0 = (t_start_0, phi_0)
    data_0 = compute_pulse_evolution(psi_ini, times_0, curr=pulse_0)

    rho_1 = data_0.states[idx_start_1]

    idx_end_1 = np.abs(
        times - (tau_coh + Delta_ts[1])
    ).argmin()  # index at which the second pulse ends
    # Take the state (after / also during) the first pulse and evolve it with the second (and potentially overlaped first) pulse

    # select range  ->  to reduce computation time
    idx_start_2 = np.abs(times - (tau_coh + T_wait - Delta_ts[2])).argmin()
    t_start_2 = times[idx_start_2]  # the time at which the third pulse starts
    idx_end_2 = np.abs(
        times - (tau_coh + T_wait + Delta_ts[2])
    ).argmin()  # end of the third pulse
    # idx_start_2_0 = np.abs(times - (T_wait - Delta_ts[2])).argmin() # the first time at which the third pulse starts

    times_1 = times[
        idx_start_1 : idx_start_2 + 1
    ]  # like this: also take the overlap into account;

    if times_1.size == 0:
        times_1 = times[idx_start_1 : idx_end_1 + 1]

    # Handle overlapping pulses: If the second pulse starts before the first pulse ends, combine their contributions
    pulse_1 = (t_start_1, phi_1)
    data_1 = compute_pulse_evolution(rho_1, times_1, curr=pulse_1, prev=pulse_0)

    idx_start_2_in_times_1 = np.abs(times_1 - (t_start_2)).argmin()

    rho_2 = data_1.states[
        idx_start_2_in_times_1
    ]  # == state where the third pulse starts

    times_2 = times[
        idx_start_2:
    ]  # the rest of the evolution (third pulse, potentially overlapped with previouses) # can be empty, if tau_coh + T_wait >= t_max
    # print(len(times), len(times_0), len(times_1), len(times_2))
    if times_2.size == 0:
        times_2 = [times[idx_start_2]]
    # If the second pulse starts before the first pulse ends, combine their contributions
    phi_2 = 0  # FIXED PHASE!
    pulse_f = (t_start_2, phi_2)

    data_f = compute_pulse_evolution(
        rho_2, times_2, curr=pulse_f, prev=pulse_1, preprev=pulse_0
    )

    t_det_start_idx_in_times_2 = np.abs(
        times_2 - (times_2[0] + Delta_ts[2])
    ).argmin()  # detection time index in times_2
    t_last_puslse_peak = times_2[t_det_start_idx_in_times_2]
    # only if we are still in the physical regime
    states = data_f.states[t_det_start_idx_in_times_2:]
    t_det_vals = data_f.times[t_det_start_idx_in_times_2:]
    data = np.zeros(
        (len(t_det_vals)), dtype=np.complex64
    )  # might get uncontrollable big!TODO

    # print(t_det_vals[0], t_det_vals[1], t_det_vals[-1], len(t_det_vals))

    if RWA_laser:
        states = [
            apply_RWA_phase_factors(state, time, omega=omega_laser)
            for state, time in zip(states, t_det_vals)
        ]

    for t_idx, t_det in enumerate(t_det_vals):
        if t_det < time_cut:
            data[:] = np.real(expect(Dip_op, states[:]))
    return np.array(t_det_vals) - t_det_vals[0], data


# Plot the data for a fixed tau_coh and T_wait
def plot_fixed_tau_T(tau_coh, T_wait, phi_0, phi_1, times):
    """Plot the data for a fixed tau_coh and T."""
    t_det_vals, data = compute_fixed_tau_T(tau_coh, T_wait, phi_0, phi_1, times=times)

    plt.figure(figsize=(10, 6))
    plt.plot(t_det_vals, np.real(data), label=r"$|\langle \mu \rangle|$")
    plt.xlabel(r"$t \, [\text{fs}]$")
    plt.ylabel(r"$|\langle \mu \rangle|$")
    plt.title(
        r"Expectation Value of $|\langle \mu \rangle|$ for fixed $\tau$ and $T$ (T=1000, $\tau=300$)"
    )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


def compute_average_fixed_tau_T(tau_coh, T_wait, times, phases):
    """
    Compute the average data for a fixed tau_coh and T_wait over all phase combinations.

    Parameters:
        tau_coh (float): Coherence time.
        T_wait (float): Waiting time.
        phases (list): List of phase values.

    Returns:
        tuple: Time values and averaged data.
    """
    results = []
    for phi_0 in phases:
        for phi_1 in phases:
            try:
                result = compute_fixed_tau_T(tau_coh, T_wait, phi_0, phi_1, times=times)
                results.append(result)
            except Exception as e:
                print(f"Error in computation for phi_0={phi_0}, phi_1={phi_1}: {e}")
                raise

    # Average the results
    t_det_vals = results[0][0]  # Time values are the same for all computations
    data_sum = np.zeros_like(results[0][1], dtype=complex)
    for _, data in results:
        data_sum += data
    data_avg = data_sum / len(results)

    return t_det_vals, data_avg


def plot_average_fixed_tau_T(tau_coh, T_wait, times, phases):
    """
    Plot the averaged data for a fixed tau_coh and T_wait over all phase combinations.

    Parameters:
        tau_coh (float): Coherence time.
        T_wait (float): Waiting time.
        phases (list): List of phase values.

    Returns:
        None
    """
    t_det_vals, data_avg = compute_average_fixed_tau_T(tau_coh, T_wait, times, phases)

    plt.figure(figsize=(10, 6))
    plt.plot(
        t_det_vals,
        np.abs(data_avg),
        label=r"$|\langle \mu \rangle|$",
        color="C0",
        linestyle="solid",
    )
    plt.xlabel(r"$t \, [\text{fs}]$")
    plt.ylabel(r"$|\langle \mu \rangle|$")
    plt.title(
        rf"Expectation Value of $|\langle \mu \rangle|$ for fixed $\tau={tau_coh}$ and $T={T_wait}$ (averaged over phases)"
    )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


######################## PARALLEL COMPUTATION


def for_one_time_calc_phase_comb(T_wait, phases, times, omega_at=omega_A):
    """
    Compute and average the 2D polarization for all phase combinations for one T_wait.
    Args:
        T_wait (float): Waiting time.
        phases (list): List of phase values.
        times (np.ndarray): Time grid for computation.
    Returns:
        np.ndarray: Averaged 2D polarization data for this T_wait.
    """
    results = []
    for phi1 in phases:
        for phi2 in phases:
            _, _, data = compute_two_dimensional_polarization(
                T_wait, phi1, phi2, times=times, omega_at=omega_at, plot_example=False
            )
            results.append(data)
    averaged_data = np.mean(np.stack(results), axis=0)
    return averaged_data


def parallel_process_all_combinations(phases, times_T, times, omega_at=omega_A):
    """
    Compute the averaged 2D polarization for all T_wait in times_T using all phase combinations.
    Args:
        phases (list): List of phase values.
        times_T (np.ndarray): Array of T_wait values.
        times (np.ndarray): Time grid for computation.
    Returns:
        list: List of averaged 2D polarization arrays for each T_wait.
    """
    results = [None] * len(times_T)

    # =============================
    # Parallelize over T_waits using ThreadPoolExecutor
    # =============================
    with ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(
                for_one_time_calc_phase_comb, T_wait, phases, times, omega_at
            ): idx
            for idx, T_wait in enumerate(times_T)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                averaged_data = future.result()
            except Exception as exc:
                print(f"T_wait index {idx} generated an exception: {exc}")
                averaged_data = None
            results[idx] = averaged_data

    return results


def parallel_process_all_omega_ats(omega_ats, phases, times_T, times):
    """
    Compute the averaged 2D polarization for all omega_ats and all T_wait in times_T using all phase combinations.
    Args:
        omega_ats (list or np.ndarray): List of omega_A values to use.
        phases (list): List of phase values.
        times_T (np.ndarray): Array of T_wait values.
        times (np.ndarray): Time grid for computation.
    Returns:
        list: List of lists, each containing averaged 2D polarization arrays for each T_wait at each omega_at.
    """
    # =============================
    # Compute results for all omega_ats
    # =============================
    all_results = []
    for omega_at in omega_ats:
        results = parallel_process_all_combinations(
            phases, times_T, times, omega_at=omega_at
        )
        all_results.append(results)

    # =============================
    # Average over omega_ats for each T_wait
    # =============================
    # all_results: shape (len(omega_ats), len(times_T))
    averaged_results = []
    for t_idx in range(len(times_T)):
        # Collect the t_idx-th result from each omega_at
        data_for_t = [all_results[o_idx][t_idx] for o_idx in range(len(omega_ats))]
        # Average over omega_ats
        averaged_data = np.mean(np.stack(data_for_t), axis=0)
        averaged_results.append(averaged_data)

    return averaged_results


def run_parallel_for_sampled_omegas(
    N, Delta, E0, phases, times_T, times, sigma=1.0, E_range=10
):
    """
    Sample omega_ats using sample_from_sigma and run parallel_process_all_omega_ats.

    Args:
        N (int): Number of samples.
        Delta (float): Energy splitting parameter.
        E0 (float): Central energy.
        phases (list): List of phase values.
        times_T (np.ndarray): Array of T_wait values.
        times (np.ndarray): Time grid for computation.
        sigma (float, optional): Standard deviation for sampling. Default is 1.0.
        E_range (float, optional): Range for sampling. Default is 10.

    Returns:
        list: Averaged 2D polarization arrays for each T_wait, averaged over sampled omegas.
    """
    omega_ats = sample_from_sigma(N, Delta, E0, sigma=sigma, E_range=E_range)
    averaged_results = parallel_process_all_omega_ats(omega_ats, phases, times_T, times)
    return averaged_results


def extend_and_plot_results(
    averaged_results, times_T, times, extend_for=None, **plot_args_freq
):
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
    plot_positive_color_map(
        (global_ts, global_taus, global_data_time),
        type="imag",
        use_custom_colormap=True,
    )
    plot_positive_color_map(
        (global_nu_ts, global_nu_taus, global_data_freq), **plot_args_freq
    )
