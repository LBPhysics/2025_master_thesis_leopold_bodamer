from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import numpy as np
from src.core.system_parameters import SystemParameters
from src.core.pulse_functions import *
from src.core.pulse_sequences import PulseSequence
import matplotlib.pyplot as plt
import os


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
        t_peak = pulse.pulse_peak_time  # Now interpreted as peak time
        Delta_width = pulse.pulse_FWHM

        # Compute individual pulse envelope using new logic
        individual_envelope = [
            (
                np.cos(np.pi * (t - t_peak) / (2 * Delta_width)) ** 2
                if t_peak - Delta_width <= t <= t_peak + Delta_width
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
            t_peak - Delta_width,
            linestyle="dotted",
            label=rf"$t_{{peak, {idx + 1}}} - \Delta_{{{idx + 1}}}$",
            alpha=0.4,
            color=colors[idx % len(colors)],
        )
        ax.axvline(
            t_peak,
            linestyle="solid",
            label=rf"$t_{{peak, {idx + 1}}}$",
            alpha=0.8,
            color=colors[idx % len(colors)],
            linewidth=2,
        )
        ax.axvline(
            t_peak + Delta_width,
            linestyle="dotted",
            label=rf"$t_{{peak, {idx + 1}}} + \Delta_{{{idx + 1}}}$",
            alpha=0.4,
            color=colors[idx % len(colors)],
        )

    # Final plot labeling
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Envelope Amplitude")
    ax.set_title(r"Pulse Envelopes for Up to Three Pulses")
    ax.legend(loc="upper right", fontsize="small")
    return ax


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
    plt.ylabel(r"$E(t) / E_0$")
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


def plot_positive_color_map(
    datas: tuple,
    T_wait: float = np.inf,
    domain: str = "time",
    type: str = "real",
    output_dir: str = None,  # TODO or maybe not? change to specific output directory
    ODE_Solver: str = None,
    save: bool = False,
    use_custom_colormap: bool = False,
    section: tuple = None,  # (x_min, x_max, y_min, y_max)
    system: SystemParameters = None,
):
    """
    Create a color plot of 2D functional data for positive x and y values.

    Parameters:
        datas (tuple): (x, y, data) where x and y are 1D arrays and data is a 2D array.
        T_wait (float): waiting time to include in plot title and file name.
        domain (str): Either 'time' or 'freq' specifying the domain of the data.
        type (str): Type of data ('real', 'imag', 'abs', or 'phase'). Used only if domain="freq".
        output_dir (str, optional): Directory to save the plot.
        ODE_Solver (str, optional): Solver name for filename.
        positive (bool): Whether to use ONLY positive values of x and y.
        save (bool): If True, saves the plot to a file.
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
    if domain not in ("time", "freq"):
        raise ValueError("Invalid domain. Must be 'time' or 'freq'.")
    if domain == "time":
        colormap = "viridis"
        title = r"$\text{Time domain}$"
        x_title = r"$t_{\text{det}}$ [fs]"
        y_title = r"$\tau_{\text{coh}}$ [fs]"
    else:
        colormap = "plasma"
        title = r"$\text{Freq domain}$"
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
    if save and output_dir and system is not None:
        if not os.path.isdir(output_dir):
            raise ValueError(f"Output directory {output_dir} does not exist.")

        filename_parts = [
            f"{domain}_domain",
        ]
        if domain == "freq":
            filename_parts.append(f"{type}_2D_spectrum")
        else:  # domain == "time":
            filename_parts.append("2D_polarization")

        ### System-specific parameters
        filename_parts.extend(
            [
                f"N={system.N_atoms}",
                f"wA={system.omega_A:.2f}",
                f"muA={system.mu_A:.0f}",
            ]
        )

        ### Add N_atoms=2 specific parameters
        if system.N_atoms == 2:
            filename_parts.extend(
                [
                    f"wb={system.omega_B/system.omega_Ayy:.2f}wA",
                    f"J={system.J:.2f}",
                    f"mub={system.mu_B/system.mu_A:.0f}muA",
                ]
            )

        ### Common parameters for both N_atoms=1 and N_atoms=2
        filename_parts.extend(
            [
                f"wL={system.omega_laser / system.omega_A:.1f}wA",
                f"E0={system.E0:.2e}",
                f"rabigen={system.rabi_gen:.2f}= sqrt({system.rabi_0:.2f}^2+{system.delta_rabi:.2f}^2)",
            ]
        )

        ### Add solver information to filename
        if ODE_Solver is not None:
            filename_parts.append(f"with_{ODE_Solver}")

        file_name_combined = (
            "_".join(filename_parts) + ".png"
        )  # TODO CHANGE TO svg for final result
        save_path_combined = os.path.join(output_dir, file_name_combined)
        plt.savefig(save_path_combined)
    else:
        print("Plot not saved. Ensure 'save' is True and 'output_dir' is specified.")
    plt.show()


def plot_E_pulse(times: np.ndarray, pulse_seq: PulseSequence, ax=None):
    """
    Plot the RWA electric field (envelope only) over time for pulses using PulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (PulseSequence): PulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        ax (matplotlib.axes.Axes): Axes object with the plot.
    """
    # Calculate the RWA electric field over time
    E_field = np.array([E_pulse(t, pulse_seq) for t in times])

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot real and imaginary parts
    ax.plot(
        times,
        np.real(E_field),
        label=r"$\mathrm{Re}[E(t)]$",
        linestyle="solid",
        color="C0",
    )
    ax.plot(
        times,
        np.imag(E_field),
        label=r"$\mathrm{Im}[E(t)]$",
        linestyle="dashed",
        color="C1",
    )
    ax.plot(
        times,
        np.abs(E_field),
        label=r"$|E(t)|$",
        linestyle="dashdot",
        color="C2",
    )

    # Final plot labeling
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Electric Field (RWA)")
    ax.set_title(r"RWA Electric Field $E(t)$ (Envelope Only)")
    ax.legend()
    return ax


def plot_Epsilon_pulse(times: np.ndarray, pulse_seq: PulseSequence, ax=None):
    """
    Plot the full electric field (with carrier) over time for pulses using PulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (PulseSequence): PulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        ax (matplotlib.axes.Axes): Axes object with the plot.
    """
    # Calculate the full electric field over time
    Epsilon_field = np.array([Epsilon_pulse(t, pulse_seq) for t in times])

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot real and imaginary parts
    ax.plot(
        times,
        np.real(Epsilon_field),
        label=r"$\mathrm{Re}[\varepsilon(t)]$",
        linestyle="solid",
        color="C3",
    )
    ax.plot(
        times,
        np.imag(Epsilon_field),
        label=r"$\mathrm{Im}[\varepsilon(t)]$",
        linestyle="dashed",
        color="C4",
    )
    ax.plot(
        times,
        np.abs(Epsilon_field),
        label=r"$|\varepsilon(t)|$",
        linestyle="dashdot",
        color="C5",
    )

    # Final plot labeling
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Electric Field (Full)")
    ax.set_title(r"Full Electric Field $\varepsilon(t)$ (With Carrier)")
    ax.legend()
    return ax


def plot_all_pulse_components(
    times: np.ndarray, pulse_seq: PulseSequence, figsize=(15, 12)
):
    """
    Plot all pulse components: envelope, RWA field, and full field in a comprehensive figure.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (PulseSequence): PulseSequence object containing pulses.
        figsize (tuple): Figure size. Defaults to (15, 12).

    Returns:
        fig (matplotlib.figure.Figure): Figure object with all plots.
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Plot pulse envelope
    plot_pulse_envelope(times, pulse_seq, ax=axes[0])

    # Plot RWA electric field
    plot_E_pulse(times, pulse_seq, ax=axes[1])

    # Plot full electric field
    plot_Epsilon_pulse(times, pulse_seq, ax=axes[2])

    # Add overall title
    fig.suptitle(
        f"Comprehensive Pulse Analysis - {len(pulse_seq.pulses)} Pulse(s)",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()
    return fig
