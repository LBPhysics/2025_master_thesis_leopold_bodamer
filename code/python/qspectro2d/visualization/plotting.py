from matplotlib.colors import TwoSlopeNorm
import numpy as np
from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.core.laser_system.laser_fcts import *
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
import matplotlib.pyplot as plt
from typing import Literal, Union
from qspectro2d.config.mpl_tex_settings import COLORS, LINE_STYLES


def plot_pulse_envelope(
    times: np.ndarray, pulse_seq: LaserPulseSequence, ax=None, show_legend=True
):
    """
    Plot the combined pulse envelope over time for up to three pulses using LaserPulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        tuple: (fig, ax) - Figure and axes objects with the plot.
    """
    # Calculate the combined envelope over time
    envelope = [pulse_envelope(t, pulse_seq) for t in times]

    # Create figure and axis if not provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Plot combined envelope
    ax.plot(
        times,
        envelope,
        label=r"$\text{Combined Envelope}$",
        linestyle=LINE_STYLES[0],  # "solid"
        alpha=0.8,
        color=list(COLORS.keys())[0],  # "C0"
    )  # Styles for up to three pulses
    linestyles = LINE_STYLES[1:4]  # ["dashed", "dashdot", "dotted"]
    colors = [list(COLORS.keys())[i] for i in [1, 2, 3]]  # ["C1", "C2", "C3"]

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
        )  # Annotate pulse key points
        ax.axvline(
            t_peak - Delta_width,
            linestyle=LINE_STYLES[3],  # "dotted"
            label=rf"$t_{{peak, {idx + 1}}} - \Delta_{{{idx + 1}}}$",
            alpha=0.4,
            color=colors[idx % len(colors)],
        )
        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[0],  # "solid"
            label=rf"$t_{{peak, {idx + 1}}}$",
            alpha=0.8,
            color=colors[idx % len(colors)],
            linewidth=2,
        )
        ax.axvline(
            t_peak + Delta_width,
            linestyle=LINE_STYLES[3],  # "dotted"
            label=rf"$t_{{peak, {idx + 1}}} + \Delta_{{{idx + 1}}}$",
            alpha=0.4,
            color=colors[idx % len(colors)],
        )

    # Final plot labeling
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Envelope Amplitude")
    ax.set_title(r"Pulse Envelopes for Up to Three Pulses")
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    return fig, ax


def plot_e_pulse(
    times: np.ndarray, pulse_seq: LaserPulseSequence, ax=None, show_legend=True
):
    """
    Plot the RWA electric field (envelope only) over time for pulses using LaserPulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        tuple: (fig, ax) - Figure and axes objects with the plot.
    """
    # Calculate the RWA electric field over time
    E_field = np.array([E_pulse(t, pulse_seq) for t in times])

    # Create figure and axis if not provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Plot real and imaginary parts
    ax.plot(
        times,
        np.real(E_field),
        label=r"$\mathrm{Re}[E(t)]$",
        linestyle=LINE_STYLES[0],  # "solid"
        color=list(COLORS.keys())[0],  # "C0"
    )
    ax.plot(
        times,
        np.imag(E_field),
        label=r"$\mathrm{Im}[E(t)]$",
        linestyle=LINE_STYLES[1],  # "dashed"
        color=list(COLORS.keys())[1],  # "C1"
    )

    # Styles for up to three pulses
    colors = [list(COLORS.keys())[i] for i in [2, 3, 4]]  # ["C2", "C3", "C4"]

    # Plot pulse peak times
    for idx, pulse in enumerate(pulse_seq.pulses[:3]):  # Up to 3 pulses
        t_peak = pulse.pulse_peak_time
        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[3],  # "dotted"
            label=rf"$t_{{peak, {idx + 1}}}$",
            color=colors[idx % len(colors)],
        )

    # Final plot labeling
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Electric Field (RWA)")
    ax.set_title(r"RWA Electric Field Components")
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return fig, ax


def plot_epsilon_pulse(
    times: np.ndarray, pulse_seq: LaserPulseSequence, ax=None, show_legend=True
):
    """
    Plot the full electric field (with carrier) over time for pulses using LaserPulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        tuple: (fig, ax) - Figure and axes objects with the plot.
    """
    # Calculate the full electric field over time
    Epsilon_field = np.array([Epsilon_pulse(t, pulse_seq) for t in times])

    # Create figure and axis if not provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Plot real and imaginary parts
    ax.plot(
        times,
        np.real(Epsilon_field),
        label=r"$\mathrm{Re}[\varepsilon(t)]$",
        linestyle=LINE_STYLES[0],  # "solid"
        color=list(COLORS.keys())[3],  # "C3"
    )
    ax.plot(
        times,
        np.imag(Epsilon_field),
        label=r"$\mathrm{Im}[\varepsilon(t)]$",
        linestyle=LINE_STYLES[1],  # "dashed"
        color=list(COLORS.keys())[4],  # "C4"
    )
    ax.plot(
        times,
        np.abs(Epsilon_field),
        label=r"$|\varepsilon(t)|$",
        linestyle=LINE_STYLES[2],  # "dashdot"
        color=list(COLORS.keys())[5],  # "C5"
    )

    # Styles for up to three pulses
    colors = [list(COLORS.keys())[i] for i in [0, 1, 2]]  # ["C0", "C1", "C2"]

    # Plot pulse peak times
    for idx, pulse in enumerate(pulse_seq.pulses[:3]):  # Up to 3 pulses
        t_peak = pulse.pulse_peak_time
        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[3],  # "dotted"
            label=rf"$t_{{peak, {idx + 1}}}$",
            color=colors[idx % len(colors)],
        )

    # Final plot labeling
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Electric Field (Full)")
    ax.set_title(r"Full Electric Field with Carrier")
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return fig, ax


def plot_all_pulse_components(
    times: np.ndarray, pulse_seq: LaserPulseSequence, figsize=(15, 12)
):
    """
    Plot all pulse components: envelope, RWA field, and full field in a comprehensive figure.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object containing pulses.
        figsize (tuple): Figure size. Defaults to (15, 12).

    Returns:
        fig (matplotlib.figure.Figure): Figure object with all plots.
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Plot pulse envelope
    plot_pulse_envelope(times, pulse_seq, ax=axes[0])

    # Plot RWA electric field
    plot_e_pulse(times, pulse_seq, ax=axes[1])

    # Plot full electric field
    plot_epsilon_pulse(times, pulse_seq, ax=axes[2])

    # Add overall title
    fig.suptitle(
        f"Comprehensive Pulse Analysis - {len(pulse_seq.pulses)} Pulse(s)",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()
    return fig


def plot_example_evo(
    times_plot: np.ndarray,
    datas: list,
    pulse_seq: LaserPulseSequence,
    t_coh: float,
    t_wait: float,
    RWA_SL: bool = False,
    ODE_Solver: str = "Paper_eqs",
    observable_strs: list[str] = [],
    **kwargs: dict,
):
    """
    Plot the evolution of the electric field and expectation values for a given t_coh and t_wait.

    Parameters:
        times_plot (np.ndarray): Time axis for the plot.
        datas (list): List of arrays of expectation values to plot.
        pulse_seq (LaserPulseSequence): Laser pulse sequence object.
        t_coh (float): Coherence time.
        t_wait (float): Waiting time.
        system: System object containing all relevant parameters.
        **kwargs: Additional keyword arguments for annotation.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    # Choose field function based on RWA setting
    if RWA_SL:
        field_func = E_pulse
    else:
        field_func = Epsilon_pulse

    # Calculate total electric field
    E0 = pulse_seq.E0
    E_total = np.array([field_func(t, pulse_seq) / E0 for t in times_plot])

    # Create plot with appropriate size
    fig, axes = plt.subplots(
        len(datas) + 1, 1, figsize=(14, 2 + 2 * len(datas)), sharex=True
    )

    # Plot electric field
    axes[0].plot(
        times_plot,
        np.real(E_total),
        color=list(COLORS.keys())[0],
        linestyle=LINE_STYLES[0],
        label=r"$\mathrm{Re}[E(t)]$",
    )
    axes[0].plot(
        times_plot,
        np.imag(E_total),
        color=list(COLORS.keys())[1],
        linestyle=LINE_STYLES[1],
        label=r"$\mathrm{Im}[E(t)]$",
    )
    axes[0].set_ylabel(r"$E(t) / E_0$")
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Plot expectation values
    for idx, data in enumerate(datas):
        ax = axes[idx + 1]

        # Determine observable label
        if observable_strs and idx < len(observable_strs):
            observable_str = observable_strs[idx]
        else:
            observable_str = r"\mu"

        # Plot real part
        ax.plot(
            times_plot,
            np.real(data),
            color=list(COLORS.keys())[0],
            linestyle=LINE_STYLES[0],
            label=r"$\mathrm{Re}\langle" + " " + observable_str + " " + r"\rangle$",
        )

        # Plot imaginary part
        ax.plot(
            times_plot,
            np.imag(data),
            color=list(COLORS.keys())[1],
            linestyle=LINE_STYLES[1],
            label=r"$\mathrm{Im}\langle" + " " + observable_str + " " + r"\rangle$",
        )

        ax.set_ylabel(r"$\langle" + observable_str + r"\rangle$")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Add parameter information as text box if kwargs provided
    if kwargs:
        text_lines = []
        for key, value in kwargs.items():
            text_lines.append(f"{key}: {str(value)}"[:30])

        plt.text(
            0.8,
            0.98,
            "\n".join(text_lines),
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.01, edgecolor="black"),
        )

    # Set x-label only on the bottom subplot
    axes[-1].set_xlabel(r"$t\,/\,\mathrm{fs}$")

    # Add title and finalize plot
    plt.suptitle(
        rf"$\t_coh = {t_coh:.2f}\,\mathrm{{fs}},\quad t_wait = {t_wait:.2f}\,\mathrm{{fs}},\quad \mathrm{{Solver}}$: {ODE_Solver}"
    )
    plt.tight_layout()

    return fig


def plot_1d_el_field(
    data_x: np.ndarray,
    data_y: np.ndarray,
    domain: Literal["time", "freq"] = "time",
    component: Literal["real", "imag", "abs", "phase"] = "real",
    title: str = None,
    function_symbol: str = "S",
    **kwargs,
) -> plt.Figure:
    """
    Plot 1D electric field data in either time or frequency domain with specified component.

    Parameters:
        data_x (np.ndarray): X-axis values (time in fs or frequency in 10^4 cm^-1)
        data_y (np.ndarray): Complex data array to plot
        domain (Literal["time", "freq"]): Domain of the data - 'time' or 'freq'. Defaults to 'time'.
        component (Literal["real", "imag", "abs", "phase"]): Component to plot - 'real', 'imag', 'abs', or 'phase'.
                                                           Defaults to 'real'.
        title (str, optional): Custom title for the plot. If None, a default title will be generated.
        function_symbol (str): Symbol to use in plot labels. Defaults to "S".
        **kwargs: Additional keyword arguments for annotation.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig = plt.figure()

    # Set domain-specific variables
    if domain == "time":
        x_label = r"$t \, [\text{fs}]$"
        if title is None:
            title = f"{function_symbol} in Time Domain"

        if component == "abs":
            y_data = np.abs(data_y)
            label = rf"$|{function_symbol}(t)|$"
            ylabel = rf"$|{function_symbol}(t)|$"
        elif component == "real":
            y_data = np.real(data_y)
            label = rf"$\mathrm{{Re}}[{function_symbol}(t)]$"
            ylabel = rf"$\mathrm{{Re}}[{function_symbol}(t)]$"
        elif component == "imag":
            y_data = np.imag(data_y)
            label = rf"$\mathrm{{Im}}[{function_symbol}(t)]$"
            ylabel = rf"$\mathrm{{Im}}[{function_symbol}(t)]$"
        elif component == "phase":
            y_data = np.angle(data_y)
            label = rf"$\mathrm{{Arg}}[{function_symbol}(t)]$"
            ylabel = rf"$\mathrm{{Arg}}[{function_symbol}(t)]$ [rad]"
        else:
            # Default to real part if invalid component is provided
            y_data = np.real(data_y)
            label = rf"$\mathrm{{Re}}[{function_symbol}(t)]$"
            ylabel = rf"$\mathrm{{Re}}[{function_symbol}(t)]$"

    elif domain == "freq":
        x_label = r"$\omega$ [$10^4$ cm$^{-1}$]"
        if title is None:
            title = f"{function_symbol} in Frequency Domain"

        if component == "abs":
            y_data = np.abs(data_y)
            label = rf"$|{function_symbol}(\omega)|$"
            ylabel = rf"$|{function_symbol}(\omega)|$"
        elif component == "real":
            y_data = np.real(data_y)
            label = rf"$\mathrm{{Re}}[{function_symbol}(\omega)]$"
            ylabel = rf"$\mathrm{{Re}}[{function_symbol}(\omega)]$"
        elif component == "imag":
            y_data = np.imag(data_y)
            label = rf"$\mathrm{{Im}}[{function_symbol}(\omega)]$"
            ylabel = rf"$\mathrm{{Im}}[{function_symbol}(\omega)]$"
        elif component == "phase":
            y_data = np.angle(data_y)
            label = rf"$\mathrm{{Arg}}[{function_symbol}(\omega)]$"
            ylabel = rf"$\mathrm{{Arg}}[{function_symbol}(\omega)]$ [rad]"
        else:
            # Default to absolute value for frequency domain if invalid component
            y_data = np.abs(data_y)
            label = rf"$|{function_symbol}(\omega)|$"
            ylabel = rf"$|{function_symbol}(\omega)|$"
    else:
        raise ValueError(f"Domain {domain} not recognized. Use 'time' or 'freq'.")

    # Select color and linestyle
    color_idx = {"abs": 0, "real": 1, "imag": 2, "phase": 3}.get(component, 0)
    color = list(COLORS.keys())[color_idx]  # "C0", "C1", etc.
    linestyle = LINE_STYLES[0]  # "solid"

    # Create the plot
    plt.plot(
        data_x,
        y_data,
        label=label,
        color=color,
        linestyle=linestyle,
    )

    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Add additional parameters as a text box if provided
    if kwargs:
        text_lines = []
        for key, value in kwargs.items():
            if key in ["function", "function_symbol", "show"]:
                continue  # Skip internal kwargs

            if isinstance(value, float):
                text_lines.append(f"{key}: {value:.3g}")
            elif isinstance(value, (int, str)):
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

    return fig


def plot_2d_el_field(
    data_x: np.ndarray,
    data_y: np.ndarray,
    data_z: np.ndarray,
    t_wait: float = np.inf,
    domain: Literal["time", "freq"] = "time",
    component: Literal["real", "imag", "abs", "phase"] = "real",
    use_custom_colormap: bool = False,
    section: Union[tuple[float, float, float, float], None] = None,
) -> Union[plt.Figure, None]:
    """
    Create a color plot of 2D electric field data for positive x and y values.

    This function plots 2D spectroscopic data where x represents detection time,
    y represents coherence time, and the data represents polarization expectation values.

    Parameters
    ----------
    data_x : np.ndarray
        1D array representing x grid (time/frequency values).
    data_y : np.ndarray
        1D array representing y grid (time/frequency values).
    data_z : np.ndarray
        2D complex array with shape (len(data_y), len(data_x)).
    t_wait : float, default np.inf
        Waiting time T (fs) to include in plot title and filename. If np.inf,
        no waiting time is displayed.
    domain : {"time", "freq"}, default "time"
        Domain of the data. "time" for time-domain plots (fs), "freq" for
        frequency-domain plots (10^4 cm^-1).
    component : {"real", "imag", "abs", "phase"}, default "real"
        Component of complex data to plot. Used for both title and data processing.
    use_custom_colormap : bool, default False
        If True, uses custom red-white-blue colormap centered at zero.
        Automatically set to True for "real", "imag", and "phase" components.
    section : tuple[float, float, float, float] or None, optional
        Crop section as (x_min, x_max, y_min, y_max) to zoom into specific region.

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure object, or None if an error occurs.

    Raises
    ------
    ValueError
        If array dimensions mismatch, data is all zeros,
        invalid domain/component values, or output_dir doesn't exist when saving.

    Examples
    --------
    >>> x = np.linspace(0, 100, 50)
    >>> y = np.linspace(0, 50, 25)
    >>> data = np.random.complex128((25, 50))
    >>> plot_2d_el_field(x, y, data, domain="time", component="real")
    """
    # =============================
    # VALIDATE INPUT
    # =============================
    x, y, data = data_x, data_y, data_z

    # Check for empty arrays
    if x.size == 0 or y.size == 0 or data.size == 0:
        print(
            f"❌ Warning: Empty arrays detected in plot_2d_el_field. x.shape={x.shape}, y.shape={y.shape}, data.shape={data.shape}"
        )
        return None

    # Convert to real arrays for plotting
    x = np.real(x)
    y = np.real(y)

    data = np.array(data, dtype=np.complex128)  # why did you do this?
    # =============================
    # SECTION CROPPING
    # =============================
    if section is not None:
        x, y, data = crop_2d_data_to_section(x, y, data, section)

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
    # SET PLOT LABELS AND COLORMAP
    # =============================
    if domain not in ("time", "freq"):
        raise ValueError("Invalid domain. Must be 'time' or 'freq'.")
    if domain == "time":
        colormap = "viridis"
        title = r"$\text{Time domain}$"
        x_title = r"$t_{\text{det}}$ [fs]"
        y_title = r"$\t_{\text{coh}}$ [fs]"
    else:
        colormap = "plasma"
        x_title = r"$\omega_{t_{\text{det}}}$ [$10^4$ cm$^{-1}$]"
        y_title = r"$\omega_{\t_{\text{coh}}}$ [$10^4$ cm$^{-1}$]"

    if component not in ("real", "imag", "abs", "phase"):
        raise ValueError(
            "Invalid component. Must be 'real', 'imag', 'abs', or 'phase'."
        )
    if component == "real":
        title = r"$\text{2D Real Spectrum}$"
        data = np.real(data)
    elif component == "imag":
        title = r"$\text{2D Imag Spectrum}$"
        data = np.imag(data)
    elif component == "abs":
        title = r"$\text{2D Abs Spectrum}$"
        data = np.abs(data)
        use_custom_colormap = False
    elif component == "phase":
        title = r"$\text{2D Phase Spectrum}$"
        data = np.angle(data)

    if t_wait != np.inf:
        title += rf"$\ \text{{at }} T = {t_wait:.2f}$ fs"

    # =============================
    # CUSTOM COLORMAP FOR ZERO-CENTERED DATA
    # =============================
    norm = None
    # For real and imag data, use red-white-blue colormap by default
    if component in ("real", "imag", "phase"):
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
    # GENERATE FIGURE
    # =============================
    fig, ax = plt.subplots(figsize=(10, 8))
    # Create the pcolormesh plot for the 2D data
    pcolor_plot = ax.pcolormesh(
        x,  # <- t_dets
        y,  # <- t_cohs
        data,  # <- data[t_cohs, t_dets]
        shading="auto",
        cmap=colormap,
        norm=norm,
    )
    cbar = fig.colorbar(pcolor_plot, ax=ax, label=cbarlabel)

    # Add contour lines with different styles for positive and negative values
    # add_custom_contour_lines(x, y, data, component) # TODO UNCOMMENT IF NEEDED

    # Improve overall plot appearance
    ax.set_title(title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    fig.tight_layout()

    """# Add a border around the plot for better visual definition
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.gca().spines["top"].set_linewidth(1.5)
    plt.gca().spines["right"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)
    plt.gca().spines["left"].set_linewidth(1.5)"""

    return fig


def plot_example_polarization(
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
        matplotlib.figure.Figure: The figure object.
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        times,
        np.abs(P_full),
        label=r"$|P_{\mathrm{full}}(t)|$",
        color=list(COLORS.keys())[0],  # "C0"
        linestyle=LINE_STYLES[0],  # "solid"
    )
    plt.plot(
        times,
        np.abs(P_only0),
        label=r"$|P_0(t)|$",
        color=list(COLORS.keys())[1],
        linestyle=LINE_STYLES[1],  # "C1", "dashed"
    )
    plt.plot(
        times,
        np.abs(P_only1),
        label=r"$|P_1(t)|$",
        color=list(COLORS.keys())[2],
        linestyle=LINE_STYLES[2],  # "C2", "dashdot"
    )
    plt.plot(
        times,
        np.abs(P_only2),
        label=r"$|P_2(t)|$",
        color=list(COLORS.keys())[3],
        linestyle=LINE_STYLES[3],  # "C3", "dotted"
    )
    plt.plot(
        times,
        np.abs(P_full - P_only0 - P_only1 - P_only2),
        label=r"$|P^{3}(t)|$",
        color=list(COLORS.keys())[4],  # "C4"
        linestyle=LINE_STYLES[0],  # "solid"
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
    return fig


# =============================
# HELPER FUNCTIONS
# =============================


def crop_2d_data_to_section(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    section: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop 2D data to a specified rectangular section.

    Parameters:
        x (np.ndarray): X-axis coordinate array
        y (np.ndarray): Y-axis coordinate array
        data (np.ndarray): 2D data array with shape (len(y), len(x))
        section (tuple): Section boundaries as (x_min, x_max, y_min, y_max)

    Returns:
        tuple: Cropped (x, y, data) arrays
    """
    x_min, x_max, y_min, y_max = section

    ### Validate coordinates are within data range
    x_min = max(x_min, np.min(x))
    x_max = min(x_max, np.max(x))
    y_min = max(y_min, np.min(y))
    y_max = min(y_max, np.max(y))

    ### Find indices within the specified section
    x_indices = np.where((x >= x_min) & (x <= x_max))[0]
    y_indices = np.where((y >= y_min) & (y <= y_max))[0]

    ### Ensure indices are within data array bounds
    x_indices = x_indices[x_indices < data.shape[1]]
    y_indices = y_indices[y_indices < data.shape[0]]

    ### Crop data and coordinate arrays
    cropped_data = data[np.ix_(y_indices, x_indices)]
    cropped_x = x[x_indices]
    cropped_y = y[y_indices]

    return cropped_x, cropped_y, cropped_data


def add_custom_contour_lines(
    x: np.ndarray, y: np.ndarray, data: np.ndarray, component: str, level_count: int = 8
) -> None:
    """
    Add custom contour lines to a 2D plot with different styles for positive/negative values.

    Parameters:
        x (np.ndarray): X-axis coordinate array
        y (np.ndarray): Y-axis coordinate array
        data (np.ndarray): 2D data array to contour
        component (str): Data component type ("real", "imag", "phase", "abs")
        level_count (int): Number of contour levels in each region (positive/negative)
    """
    ### Add contour lines with different styles for positive and negative values
    if component in ("real", "imag", "phase"):
        ### Determine contour levels based on the data range
        vmax = max(abs(np.min(data)), abs(np.max(data)))
        vmin = -vmax

        ### Create evenly spaced levels for both positive and negative regions
        if vmax > 0:
            positive_levels = np.linspace(0.05 * vmax, 0.95 * vmax, level_count)
            negative_levels = np.linspace(0.95 * vmin, 0.05 * vmin, level_count)

            ### Plot positive contours (solid lines)
            pos_contour = plt.contour(
                x,
                y,
                data,
                levels=positive_levels,
                colors="black",
                linewidths=0.7,
                alpha=0.8,
            )

            ### Plot negative contours (dashed lines)
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

            ### Optional: Add contour labels to every other contour line
            # plt.clabel(pos_contour, inline=True, fontsize=8, fmt='%.2f', levels=positive_levels[::2])
            # plt.clabel(neg_contour, inline=True, fontsize=8, fmt='%.2f', levels=negative_levels[::2])
    else:
        ### For abs and phase, use standard contours
        contour_plot = plt.contour(
            x,
            y,
            data,
            levels=level_count,
            colors="black",
            linewidths=0.7,
            alpha=0.8,
        )
        ### Optional: Add contour labels
        plt.clabel(
            contour_plot,
            inline=True,
            fontsize=8,
            fmt="%.2f",
            levels=contour_plot.levels[::2],
        )
