from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import numpy as np
from qspectro2d.core.system_parameters import SystemParameters
from qspectro2d.core.pulse_functions import *
from qspectro2d.core.pulse_sequences import PulseSequence
import matplotlib.pyplot as plt
import os


def Plot_pulse_envelope(times: np.ndarray, pulse_seq: PulseSequence, ax=None):
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
        Delta_width = pulse.pulse_fwhm

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


def Plot_E_pulse(times: np.ndarray, pulse_seq: PulseSequence, ax=None):
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


def Plot_Epsilon_pulse(times: np.ndarray, pulse_seq: PulseSequence, ax=None):
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


def Plot_all_pulse_components(
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
    Plot_pulse_envelope(times, pulse_seq, ax=axes[0])

    # Plot RWA electric field
    Plot_E_pulse(times, pulse_seq, ax=axes[1])

    # Plot full electric field
    Plot_Epsilon_pulse(times, pulse_seq, ax=axes[2])

    # Add overall title
    fig.suptitle(
        f"Comprehensive Pulse Analysis - {len(pulse_seq.pulses)} Pulse(s)",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()
    return fig


def Plot_fixed_tau_T(t_det_vals: np.ndarray, data: np.ndarray, **kwargs: dict):
    """
    Plot the data for a fixed tau_coh and T.

    Parameters
    ----------
    t_det_vals : array-like
        The time delay values for the x-axis.
    data : array-like
        The data to plot on the y-axis, typically the expectation value of the polarization.
    **kwargs : dict
        Additional keyword arguments:
        - function (str): Name of the function being plotted. Default is "P".
        - show (bool): Whether to display the plot immediately. Default is True.
           Set to False if you want to save the figure elsewhere before displaying.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object, which can be used for further customization or saving.
    """

    f = kwargs.get("function", "P")
    # Set label and title based on f
    y_label = rf"${f}(t)$"
    plot_title = rf"{f} for fixed $\tau$ and $T$"

    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        t_det_vals,
        np.real(data),
        color="C1",
        linestyle="dashed",
        linewidth=0.75,
        label=rf"$\mathrm{{Re}}[{f}(t)]$",
    )
    plt.plot(
        t_det_vals,
        np.imag(data),
        color="C2",
        linestyle="dotted",
        linewidth=0.75,
        label=rf"$\mathrm{{Im}}[{f}(t)]$",
    )
    plt.plot(
        t_det_vals,
        np.abs(data),
        color="C0",
        linestyle="solid",
        label=rf"$|{f}(t)|$",
    )

    plt.xlabel(r"$t \, [\text{fs}]$")
    plt.ylabel(ylabel=y_label)
    plt.title(plot_title)
    if kwargs:
        # Format additional parameters
        text_lines = []
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    text_lines.append(f"{key}: {value:.3g}")
                else:
                    text_lines.append(f"{key}: {value}")
            elif isinstance(value, np.ndarray):
                # Handle numpy arrays safely - show shape instead of content
                text_lines.append(f"{key}: array(shape={value.shape})")
            else:
                # Convert to string and ensure it doesn't have LaTeX special characters
                safe_str = (
                    str(value)
                    .replace("_", "\\_")
                    .replace("^", "\\^")
                    .replace("{", "\\{")
                    .replace("}", "\\}")
                )
                text_lines.append(f"{key}: {safe_str}")

        # Add text box with small font
        info_text = "\n".join(text_lines)
        plt.text(
            0.8,
            0.98,
            info_text,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.01, edgecolor="black"),
        )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # Only show the plot if not being saved elsewhere
    show = kwargs.get("show", True)
    if show:
        plt.show()

    return fig


def Plot_example_evo(
    times_plot: np.ndarray,
    datas: list,
    pulse_seq_f: PulseSequence,
    tau_coh: float,
    T_wait: float,
    system: SystemParameters,
    **kwargs: dict,
):
    """
    Plot the evolution of the electric field and expectation values for a given tau_coh and T_wait.

    Parameters:
        times_plot (np.ndarray): Time axis for the plot. Comprised of three time ranges for the three pulses.
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
    fig, axes = plt.subplots(
        len(datas) + 1, 1, figsize=(14, 2 + 2 * len(datas)), sharex=True
    )

    # Plot electric field
    axes[0].plot(
        times_plot,
        np.real(E_total),
        color="C0",
        linestyle="solid",
        label=r"$\mathrm{Re}[E(t)]$",
    )
    axes[0].plot(
        times_plot,
        np.imag(E_total),
        color="C1",
        linestyle="dashed",
        label=r"$\mathrm{Im}[E(t)]$",
    )
    axes[0].set_ylabel(r"$E(t) / E_0$")
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Plot expectation values
    for idx, data in enumerate(datas):
        ax = axes[idx + 1]
        if hasattr(system, "observable_strs") and idx < len(system.observable_strs):
            label = (
                r"$\mathrm{Re}\langle"
                + " "
                + system.observable_strs[idx]
                + " "
                + r"\rangle$"
            )
        else:
            label = r"$\mathrm{Re}\langle \mu \rangle$"
        ax.plot(times_plot, data, color=f"C{(idx+5)%10}", linestyle="solid")
        ax.axvline(0, color="C2", linestyle="dashed", label="Pulse0")
        ax.axvline(tau_coh, color="C3", linestyle="dashdot", label="Pulse1")
        ax.axvline(tau_coh + T_wait, color="C4", linestyle="dotted", label="Pulse2")
        ax.set_ylabel(label)

    # Set x-label only on the bottom subplot
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axes[-1].set_xlabel(r"$t\,/\,\mathrm{fs}$")

    if kwargs:
        # Format additional parameters
        text_lines = []
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    text_lines.append(f"{key}: {value:.3g}")
                else:
                    text_lines.append(f"{key}: {value}")
            else:
                text_lines.append(f"{key}: {value}")

        # Add text box with small font
        info_text = "\n".join(text_lines)
        plt.text(
            0.8,
            0.98,
            info_text,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.01, edgecolor="black"),
        )

    plt.suptitle(
        rf"$\tau = {tau_coh:.2f}\,\mathrm{{fs}},\quad T = {T_wait:.2f}\,\mathrm{{fs}},\quad \mathrm{{Solver}}$: {system.ODE_Solver}"
    )
    plt.tight_layout()
    plt.show()


def Plot_2d_El_field(
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
) -> None:
    """
    x == t_det, y == tau_coh, data == polarization expectation value
    Create a color plot of 2D functional data for positive x and y values.

    Parameters:
        datas (tuple): (x, y, data) where x and y are 1D arrays and data is a 2D array.
        T_wait (float): waiting time to include in plot title and file name.
        domain (str): Either 'time' or 'freq' specifying the domain of the data.
        type (str): Type of data ('real', 'imag', 'abs', or 'phase'). Used only if domain="freq".
        output_dir (str, optional): Directory to save the plot.
        ODE_Solver (str, optional): Solver name for filename.
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
    # Custom colormap for zero-centered data
    # =============================
    norm = None
    # For real and imag data, use red-white-blue colormap by default
    if type in ("real", "imag", "phase"):
        use_custom_colormap = True

    if use_custom_colormap:
        vmin = np.min(data)
        vmax = np.max(data)
        vcenter = 0

        # Use the built-in 'RdBu_r' colormap - reversed to make red=positive, blue=negative
        colormap = plt.get_cmap("RdBu_r")

        # Center the colormap at zero for diverging data
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
    # Create the pcolormesh plot for the 2D data
    pcolor_plot = plt.pcolormesh(
        x,  # <- ts
        y,  # <- taus
        data,  # <- data[taus, ts]
        shading="auto",
        cmap=colormap,
        norm=norm,
    )

    """
    # Add contour lines with different styles for positive and negative values
    if type in ("real", "imag", "phase"):
        # Determine contour levels based on the data range
        vmax = max(abs(np.min(data)), abs(np.max(data)))
        vmin = -vmax

        # Create evenly spaced levels for both positive and negative regions
        level_count = 8  # Number of contour levels in each region (positive/negative)
        if vmax > 0:
            positive_levels = np.linspace(0.05 * vmax, 0.95 * vmax, level_count)
            negative_levels = np.linspace(0.95 * vmin, 0.05 * vmin, level_count)

            # Plot positive contours (solid lines)
            pos_contour = plt.contour(
                x,
                y,
                data,
                levels=positive_levels,
                colors="black",
                linewidths=0.7,
                alpha=0.8,
            )

            # Plot negative contours (dashed lines)
            neg_contour = plt.contour(
                x,
                y,
                data,
                levels=negative_levels,
                colors="black",
                linewidths=0.7,
                alpha=0.8,
                linestyles="dashed",
            )

            # Optional: Add contour labels to every other contour line
            # plt.clabel(pos_contour, inline=True, fontsize=8, fmt='%.2f', levels=positive_levels[::2])
            # plt.clabel(neg_contour, inline=True, fontsize=8, fmt='%.2f', levels=negative_levels[::2])
    else:
        # For abs and phase, use standard contours
        level_count = 8
        contour_plot = plt.contour(
            x,
            y,
            data,
            levels=level_count,
            colors="black",
            linewidths=0.7,
            alpha=0.8,
        )
        # Optional: Add contour labels
        plt.clabel(
            contour_plot,
            inline=True,
            fontsize=8,
            fmt="%.2f",
            levels=contour_plot.levels[::2],
        )
    """
    # Add colorbar
    cbar = plt.colorbar(label=cbarlabel)

    # Improve overall plot appearance
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    """# Add a border around the plot for better visual definition
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.gca().spines["top"].set_linewidth(1.5)
    plt.gca().spines["right"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)
    plt.gca().spines["left"].set_linewidth(1.5)"""

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
                    f"wb={system.omega_B/system.omega_A:.2f}wA",
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


def Plot_example_Polarization(
    times: np.ndarray,
    P_full: np.ndarray,
    P_only0: np.ndarray,
    P_only1: np.ndarray,
    P_only2: np.ndarray,
    title: str = "Example Polarization Components",
    **kwargs,
):
    """
    Plot the full and individual polarization components as a function of time.

    Parameters:
        times (np.ndarray): Time axis (e.g., detection times).
        P_full (np.ndarray): Full polarization (complex).
        P_only0 (np.ndarray): Polarization from only the first pulse.
        P_only1 (np.ndarray): Polarization from only the second pulse.
        P_only2 (np.ndarray): Polarization from only the third pulse.
        title (str): Plot title.
        **kwargs: Additional keyword arguments for annotation.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        times,
        np.abs(P_full),
        label=r"$|P_{\mathrm{full}}(t)|$",
        color="C0",
        linestyle="solid",
    )
    plt.plot(
        times, np.abs(P_only0), label=r"$|P_0(t)|$", color="C1", linestyle="dashed"
    )
    plt.plot(
        times, np.abs(P_only1), label=r"$|P_1(t)|$", color="C2", linestyle="dashdot"
    )
    plt.plot(
        times, np.abs(P_only2), label=r"$|P_2(t)|$", color="C3", linestyle="dotted"
    )
    plt.plot(
        times,
        np.abs(P_full - P_only0 - P_only1 - P_only2),
        label=r"$|P^{3}(t)|$",
        color="C4",
        linestyle="solid",
    )
    plt.xlabel(r"$t_{\mathrm{det}}$ [fs]")
    plt.ylabel(r"$|P(t)|$")
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Add additional parameters as a text box if provided
    if kwargs:
        text_lines = []
        for key, value in kwargs.items():
            if isinstance(value, float):
                text_lines.append(f"{key}: {value:.3g}")
            else:
                text_lines.append(f"{key}: {value}")
        info_text = "\n".join(text_lines)
        plt.text(
            0.98,
            0.98,
            info_text,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.05, edgecolor="black"),
        )

    plt.tight_layout()
    plt.show()


def Plot_1d_frequency_spectrum(
    nu_vals: np.ndarray,
    spectrum_data: np.ndarray,
    type: str = "abs",
    title: str = "1D Frequency Spectrum",
    output_dir: str = None,
    save: bool = False,
    system: SystemParameters = None,
    **kwargs,
):
    """
    Plot the 1D frequency spectrum from Fourier-transformed polarization data.

    Parameters:
        nu_vals (np.ndarray): Frequency values in wavenumber units (10^4 cm^-1).
        spectrum_data (np.ndarray): Complex spectrum data from FFT.
        type (str): Type of data to plot - 'abs', 'real', 'imag', or 'phase'. Defaults to 'abs'.
        title (str): Plot title. Defaults to '1D Frequency Spectrum'.
        output_dir (str): Directory to save the plot. Defaults to None.
        save (bool): If True, saves the plot to a file. Defaults to False.
        system (SystemParameters): System parameters object for filename. Defaults to None.
        **kwargs: Additional keyword arguments for annotation.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))

    # Plot different representations based on type
    if type == "abs":
        plt.plot(
            nu_vals,
            np.abs(spectrum_data),
            label=r"$|S(\omega)|$",
            color="C0",
            linestyle="solid",
        )
        ylabel = r"$|S(\omega)|$"
    elif type == "real":
        plt.plot(
            nu_vals,
            np.real(spectrum_data),
            label=r"$\mathrm{Re}[S(\omega)]$",
            color="C1",
            linestyle="solid",
        )
        ylabel = r"$\mathrm{Re}[S(\omega)]$"
    elif type == "imag":
        plt.plot(
            nu_vals,
            np.imag(spectrum_data),
            label=r"$\mathrm{Im}[S(\omega)]$",
            color="C2",
            linestyle="solid",
        )
        ylabel = r"$\mathrm{Im}[S(\omega)]$"
    elif type == "phase":
        plt.plot(
            nu_vals,
            np.angle(spectrum_data),
            label=r"$\mathrm{Arg}[S(\omega)]$",
            color="C3",
            linestyle="solid",
        )
        ylabel = r"$\mathrm{Arg}[S(\omega)]$ [rad]"
    else:
        plt.plot(
            nu_vals,
            np.abs(spectrum_data),
            label=r"$|S(\omega)|$",
            color="C0",
            linestyle="solid",
        )
        ylabel = r"$|S(\omega)|$"

    plt.xlabel(r"$\omega$ [$10^4$ cm$^{-1}$]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Add additional parameters as a text box if provided
    if kwargs:
        text_lines = []
        for key, value in kwargs.items():
            if isinstance(value, float):
                text_lines.append(f"{key}: {value:.3g}")
            else:
                text_lines.append(f"{key}: {value}")
        info_text = "\n".join(text_lines)
        plt.text(
            0.98,
            0.98,
            info_text,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.05, edgecolor="black"),
        )

    # =============================
    # Save or show
    # =============================
    if save and output_dir and system is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        filename_parts = [
            f"freq_domain",
            f"{type}_1D_spectrum",
        ]

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
                    f"wb={system.omega_B/system.omega_A:.2f}wA",
                    f"J={system.J:.2f}",
                    f"mub={system.mu_B/system.mu_A:.0f}muA",
                ]
            )

        ### Common parameters
        filename_parts.extend(
            [
                f"wL={system.omega_laser / system.omega_A:.1f}wA",
                f"E0={system.E0:.2e}",
                f"rabigen={system.rabi_gen:.2f}",
            ]
        )

        file_name = "_".join(filename_parts) + ".png"
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path)
    else:
        print("Plot not saved. Ensure 'save' is True and 'output_dir' is specified.")
    plt.show()
