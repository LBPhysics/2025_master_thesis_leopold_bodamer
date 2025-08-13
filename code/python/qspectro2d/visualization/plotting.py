from matplotlib.colors import TwoSlopeNorm
import numpy as np
from qspectro2d.core.laser_system.laser_fcts import *
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
import matplotlib.pyplot as plt
from typing import Literal, Union, Tuple
from plotstyle import init_style, COLORS, LINE_STYLES

init_style()


# Provide minimal fallbacks for former helper names
def _maybe(latex_label: str, plain: str | None = None):
    return latex_label if plain is None else plain


def _simplify_figure_text(fig):
    return fig


USE_LATEX = False


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
    envelope = pulse_envelope(times, pulse_seq)

    # Create figure and axis if not provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Plot combined envelope
    ax.plot(
        times,
        envelope,
        label=_maybe(r"$\text{Combined Envelope}$"),
        linestyle=LINE_STYLES[0],
        alpha=0.8,
        color=list(COLORS.keys())[0],
    )  # Styles for up to three pulses
    linestyles = LINE_STYLES[1:4]
    colors = [list(COLORS.keys())[i] for i in [1, 2, 3]]

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
            linestyle=LINE_STYLES[3],
            label=_maybe(
                rf"$t_{{peak, {idx + 1}}} - \Delta_{{{idx + 1}}}$",
                f"t_peak{idx+1}-Δ{idx+1}",
            ),
            alpha=0.4,
            color=colors[idx % len(colors)],
        )
        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[0],
            label=_maybe(rf"$t_{{peak, {idx + 1}}}$", f"t_peak{idx+1}"),
            alpha=0.8,
            color=colors[idx % len(colors)],
            linewidth=2,
        )
        ax.axvline(
            t_peak + Delta_width,
            linestyle=LINE_STYLES[3],
            label=_maybe(
                rf"$t_{{peak, {idx + 1}}} + \Delta_{{{idx + 1}}}$",
                f"t_peak{idx+1}+Δ{idx+1}",
            ),
            alpha=0.4,
            color=colors[idx % len(colors)],
        )

    # Final plot labeling
    ax.set_xlabel(_maybe(r"Time $t$", "Time t"))
    ax.set_ylabel(_maybe(r"Envelope Amplitude"))
    ax.set_title(_maybe(r"Pulse Envelopes for Up to Three Pulses"))
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.close(fig)
    _simplify_figure_text(fig)
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
        label=_maybe(r"$\mathrm{Re}[E(t)]$", "Re E(t)"),
        linestyle=LINE_STYLES[0],  # "solid"
        color=list(COLORS.keys())[0],  # "C0"
    )
    ax.plot(
        times,
        np.imag(E_field),
        label=_maybe(r"$\mathrm{Im}[E(t)]$", "Im E(t)"),
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
            label=_maybe(rf"$t_{{peak, {idx + 1}}}$", f"t_peak{idx+1}"),
            color=colors[idx % len(colors)],
        )

    # Final plot labeling
    ax.set_xlabel(_maybe(r"Time $t$", "Time t"))
    ax.set_ylabel(_maybe(r"Electric Field (RWA)"))
    ax.set_title(_maybe(r"RWA Electric Field Components"))
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.close(fig)
    _simplify_figure_text(fig)
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
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        times,
        np.real(Epsilon_field),
        label=_maybe(r"$\mathrm{Re}[\varepsilon(t)]$", "Re eps(t)"),
        linestyle=LINE_STYLES[0],
        color=list(COLORS.keys())[3],
    )
    ax.plot(
        times,
        np.imag(Epsilon_field),
        label=_maybe(r"$\mathrm{Im}[\varepsilon(t)]$", "Im eps(t)"),
        linestyle=LINE_STYLES[1],
        color=list(COLORS.keys())[4],
    )
    ax.plot(
        times,
        np.abs(Epsilon_field),
        label=_maybe(r"$|\varepsilon(t)|$", "|eps(t)|"),
        linestyle=LINE_STYLES[2],
        color=list(COLORS.keys())[5],
    )
    colors = [list(COLORS.keys())[i] for i in [0, 1, 2]]
    for idx, pulse in enumerate(pulse_seq.pulses[:3]):
        t_peak = pulse.pulse_peak_time
        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[3],
            label=_maybe(rf"$t_{{peak, {idx + 1}}}$", f"t_peak{idx+1}"),
            color=colors[idx % len(colors)],
        )
    ax.set_xlabel(_maybe(r"Time $t$", "Time t"))
    ax.set_ylabel(_maybe(r"Electric Field (Full)", "Electric field (full)"))
    ax.set_title(_maybe(r"Full Electric Field with Carrier", "Full electric field"))
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.close(fig)
    _simplify_figure_text(fig)
    return fig, ax


def plot_1d_el_field(
    axis_det: np.ndarray,
    data: np.ndarray,
    domain: Literal["time", "freq"] = "time",
    component: Literal["real", "imag", "abs", "phase"] = "real",
    title: str | None = None,
    section: Union[tuple[float, float], None] = None,
    function_symbol: str = "S",
    figsize: Tuple[float, float] = (6.5, 4.0),
    normalize: bool = True,
    **kwargs: dict,
) -> plt.Figure:
    """Plot 1D complex data (time or frequency domain).

    # =============================
    # CONTRACT
    # =============================
    Inputs: axis_det (1d), complex data (same length), domain selector, component selector
    Output: matplotlib Figure object
    Normalization: optional (default True) to max absolute amplitude
    Cropping: optional via section=(min,max)
    """
    fig = plt.figure(figsize=figsize)

    # =============================
    # CROP + NORMALIZE
    # =============================
    if section is not None:
        axis_det, data = crop_nd_data_along_axis(
            axis_det, data, section=section, axis=0
        )
    if normalize:
        max_abs = np.abs(data).max()
        if max_abs == 0:
            raise ValueError("Data array is all zeros, cannot normalize.")
        data = data / max_abs

    # =============================
    # COMPONENT HANDLING
    # =============================
    y_data, label, ylabel, x_label, final_title = _resolve_1d_labels_and_component(
        data=data,
        domain=domain,
        component=component,
        function_symbol=function_symbol,
        provided_title=title,
    )

    # =============================
    # STYLE
    # =============================
    color, linestyle = _style_for_component(component)

    # =============================
    # PLOT
    # =============================
    plt.plot(axis_det, y_data, label=_maybe(label), color=color, linestyle=linestyle)
    plt.xlabel(_maybe(x_label))
    plt.ylabel(_maybe(ylabel))
    plt.title(_maybe(final_title))
    plt.legend()
    add_text_box(ax=plt.gca(), kwargs=kwargs)
    plt.tight_layout()
    _simplify_figure_text(fig)
    plt.close(fig)
    return fig


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
    plt.close(fig)
    return fig


def plot_example_evo(
    times_plot: np.ndarray,
    datas: list,
    pulse_seq: LaserPulseSequence,
    rwa_sl: bool = False,
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
    if rwa_sl:
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
        label=_maybe(r"$\mathrm{Re}[E(t)]$", "Re E(t)"),
    )
    axes[0].plot(
        times_plot,
        np.imag(E_total),
        color=list(COLORS.keys())[1],
        linestyle=LINE_STYLES[1],
        label=_maybe(r"$\mathrm{Im}[E(t)]$", "Im E(t)"),
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
            label=_maybe(
                r"$\mathrm{Re}\langle" + " " + observable_str + " " + r"\rangle$",
                f"Re <{observable_str}>",
            ),
        )

        # Plot imaginary part
        ax.plot(
            times_plot,
            np.imag(data),
            color=list(COLORS.keys())[1],
            linestyle=LINE_STYLES[1],
            label=_maybe(
                r"$\mathrm{Im}\langle" + " " + observable_str + " " + r"\rangle$",
                f"Im <{observable_str}>",
            ),
        )

    ax.set_ylabel(
        _maybe(r"$\langle" + observable_str + r"\rangle$", f"<{observable_str}>")
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    add_text_box(ax=axes[0], kwargs=kwargs)

    # Set x-label only on the bottom subplot
    axes[-1].set_xlabel(_maybe(r"$t\,/\,\mathrm{fs}$", "t / fs"))

    plt.tight_layout()
    _simplify_figure_text(fig)
    plt.close(fig)
    return fig


def plot_1d_el_field(
    axis_det: np.ndarray,
    data: np.ndarray,
    domain: Literal["time", "freq"] = "time",
    component: Literal["real", "imag", "abs", "phase"] = "real",
    title: str = None,
    section: Union[tuple[float, float], None] = None,
    function_symbol: str = "S",
    **kwargs: dict,
) -> plt.Figure:
    """
    Plot 1D electric field data in either time or frequency domain with specified component.

    Parameters:
        axis_det (np.ndarray): X-axis values (time in fs or frequency in 10^4 cm^-1)
        data (np.ndarray): Complex data array to plot
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

    if section is not None:
        # Crop the data if a section is specified
        axis_det, data = crop_nd_data_along_axis(
            axis_det, data, section=section, axis=0
        )

    if np.abs(data).max() == 0:
        raise ValueError("Data array is all zeros, cannot normalize.")
    data = data / np.abs(data).max()  # normalize

    # Set domain-specific variables
    if domain == "time":
        x_label = r"$t \, [\text{fs}]$"
        if title is None:
            title = f"{function_symbol} in Time Domain"

        if component == "abs":
            y_data = np.abs(data)
            label = rf"$|{function_symbol}(t)|$"
            ylabel = rf"$|{function_symbol}(t)|$"
        elif component == "real":
            y_data = np.real(data)
            label = rf"$\mathrm{{Re}}[{function_symbol}(t)]$"
            ylabel = rf"$\mathrm{{Re}}[{function_symbol}(t)]$"
        elif component == "imag":
            y_data = np.imag(data)
            label = rf"$\mathrm{{Im}}[{function_symbol}(t)]$"
            ylabel = rf"$\mathrm{{Im}}[{function_symbol}(t)]$"
        elif component == "phase":
            y_data = np.angle(data)
            label = rf"$\mathrm{{Arg}}[{function_symbol}(t)]$"
            ylabel = rf"$\mathrm{{Arg}}[{function_symbol}(t)]$ [rad]"
        else:
            # Default to real part if invalid component is provided
            y_data = np.real(data)
            label = rf"$\mathrm{{Re}}[{function_symbol}(t)]$"
            ylabel = rf"$\mathrm{{Re}}[{function_symbol}(t)]$"

    elif domain == "freq":
        x_label = r"$\omega$ [$10^4$ cm$^{-1}$]"
        if title is None:
            title = f"{function_symbol} in Frequency Domain"

        if component == "abs":
            y_data = np.abs(data)
            label = rf"$|{function_symbol}(\omega)|$"
            ylabel = rf"$|{function_symbol}(\omega)|$"
        elif component == "real":
            y_data = np.real(data)
            label = rf"$\mathrm{{Re}}[{function_symbol}(\omega)]$"
            ylabel = rf"$\mathrm{{Re}}[{function_symbol}(\omega)]$"
        elif component == "imag":
            y_data = np.imag(data)
            label = rf"$\mathrm{{Im}}[{function_symbol}(\omega)]$"
            ylabel = rf"$\mathrm{{Im}}[{function_symbol}(\omega)]$"
        elif component == "phase":
            y_data = np.angle(data)
            label = rf"$\mathrm{{Arg}}[{function_symbol}(\omega)]$"
            ylabel = rf"$\mathrm{{Arg}}[{function_symbol}(\omega)]$ [rad]"
        else:
            # Default to absolute value for frequency domain if invalid component
            y_data = np.abs(data)
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
        axis_det,
        y_data,
        label=_maybe(label),
        color=color,
        linestyle=linestyle,
    )

    plt.xlabel(_maybe(x_label))
    plt.ylabel(_maybe(ylabel))
    plt.title(_maybe(title))
    plt.legend()

    # Add additional parameters as a text box if provided
    add_text_box(ax=plt.gca(), kwargs=kwargs)

    plt.tight_layout()
    _simplify_figure_text(fig)
    plt.close(fig)
    return fig


def plot_2d_el_field(
    axis_det: np.ndarray,  # detection axis
    axis_coh: np.ndarray,  # coherence axis
    data: np.ndarray,  # complex 2D array
    t_wait: float = np.inf,
    domain: Literal["time", "freq"] = "time",
    component: Literal["real", "imag", "abs", "phase"] = "real",
    use_custom_colormap: bool = False,
    section: Union[list[tuple[float, float]], None] = None,
    figsize: Tuple[float, float] = (7.0, 5.6),
    normalize: bool = True,
    **kwargs: dict,
) -> Union[plt.Figure, None]:
    """
    Create a color plot of 2D electric field data for positive x and y values.

    This function plots 2D spectroscopic data where x represents detection time,
    y represents coherence time, and the data represents polarization expectation values.

    Parameters
    ----------
    axis_det : np.ndarray
        1D array representing x grid (time/frequency values).
    axis_coh : np.ndarray
        1D array representing y grid (time/frequency values).
    data : np.ndarray
        2D complex array with shape (len(axis_coh), len(axis_det)).
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
    section : first tuple crops coh axis (coh_min, coh_max),
              second tuple crops det axis (det_min, det_max) to zoom into specific region.

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
    if (
        data.ndim != 2
        or data.shape[0] != len(axis_coh)
        or data.shape[1] != len(axis_det)
    ):
        raise ValueError(
            f"Data shape {data.shape} does not match axis_det ({len(axis_det)}) and axis_coh ({len(axis_coh)}) dimensions."
        )

    # Check for empty arrays
    if axis_det.size == 0 or axis_coh.size == 0 or data.size == 0:
        print(
            f"❌ Warning: Empty arrays detected in plot_2d_el_field. axis_det.shape={axis_det.shape}, axis_coh.shape={axis_coh.shape}, data.shape={data.shape}"
        )
        return None

    # Convert axes to real (robust against minor numerical complex drift)
    axis_det = np.real(axis_det)
    axis_coh = np.real(axis_coh)
    data = np.asarray(data, dtype=np.complex128)
    # =============================
    # SECTION CROPPING
    # =============================
    if section is not None:
        # expect list[(coh_min, coh_max),(det_min, det_max)]
        axis_coh, data = crop_nd_data_along_axis(
            axis_coh, data, section=section[0], axis=0
        )
        axis_det, data = crop_nd_data_along_axis(
            axis_det, data, section=section[1], axis=1
        )
    if normalize:
        max_abs = np.abs(data).max()
        if max_abs == 0:
            raise ValueError("Data array is all zeros, cannot normalize.")
        data = data / max_abs

    # =============================
    # SET PLOT LABELS AND COLORMAP
    # =============================
    data, title_base = _component_2d_data(data=data, component=component)
    colormap, x_title, y_title, domain_suffix = _domain_2d_labels(domain=domain)
    title = title_base + domain_suffix
    if t_wait != np.inf:
        title += rf"$\ (T = {t_wait:.2f}\,\text{{fs}})$"

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
    fig, ax = plt.subplots(figsize=figsize)
    # Create the pcolormesh plot for the 2D data
    im_plot = ax.imshow(
        data,  # data shape: [len(axis_coh), len(axis_det)]
        aspect="auto",
        origin="lower",  # origin at bottom-left
        cmap=colormap,
        norm=norm,
        extent=[
            axis_det[0],
            axis_det[-1],  # X-axis: detection time
            axis_coh[0],
            axis_coh[-1],  # Y-axis: coherence time
        ],
        interpolation="bilinear",  # optional: "none" to avoid smoothing
    )
    cbar = fig.colorbar(im_plot, ax=ax, label=cbarlabel)

    # Add contour lines with different styles for positive and negative values
    # add_custom_contour_lines(axis_coh, axis_det, data, component)

    # Improve overall plot appearance
    ax.set_title(_maybe(title))
    ax.set_xlabel(_maybe(x_title))
    ax.set_ylabel(_maybe(y_title))

    # Add additional parameters as a text box if provided
    add_text_box(ax=ax, kwargs=kwargs)

    fig.tight_layout()

    """# Add a border around the plot for better visual definition plt.gca().spines["top"].set_visible(True); plt.gca().spines["bottom"].set_linewidth(1.5)"""

    # plt.close(fig)  # keep figure open for further user modification if desired
    _simplify_figure_text(fig)
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
    fig = plt.figure(figsize=(10, 6))  # get rid of this fixed figsize
    plt.plot(
        times,
        np.abs(P_full),
        label=_maybe(r"$|P_{\mathrm{full}}(t)|$", "|P_full(t)|"),
        color=list(COLORS.keys())[0],  # "C0"
        linestyle=LINE_STYLES[0],  # "solid"
    )
    plt.plot(
        times,
        np.abs(P_only0),
        label=_maybe(r"$|P_0(t)|$", "|P0(t)|"),
        color=list(COLORS.keys())[1],
        linestyle=LINE_STYLES[1],  # "C1", "dashed"
    )
    plt.plot(
        times,
        np.abs(P_only1),
        label=_maybe(r"$|P_1(t)|$", "|P1(t)|"),
        color=list(COLORS.keys())[2],
        linestyle=LINE_STYLES[2],  # "C2", "dashdot"
    )
    plt.plot(
        times,
        np.abs(P_only2),
        label=_maybe(r"$|P_2(t)|$", "|P2(t)|"),
        color=list(COLORS.keys())[3],
        linestyle=LINE_STYLES[3],  # "C3", "dotted"
    )
    plt.plot(
        times,
        np.abs(P_full - P_only0 - P_only1 - P_only2),
        label=_maybe(r"$|P^{3}(t)|$", "|P3(t)|"),
        color=list(COLORS.keys())[4],  # "C4"
        linestyle=LINE_STYLES[0],  # "solid"
    )
    plt.xlabel(_maybe(r"$t_{\mathrm{det}}$ [fs]", "t_det [fs]"))
    plt.ylabel(_maybe(r"$|P(t)|$", "|P(t)|"))
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Add additional parameters as a text box if provided
    add_text_box(ax=plt.gca(), kwargs=kwargs)

    plt.tight_layout()
    _simplify_figure_text(fig)
    plt.close(fig)
    return fig


# =============================
# HELPER FUNCTIONS
# =============================
def crop_nd_data_along_axis(
    coord_array: np.ndarray,
    nd_data: np.ndarray,
    section: tuple[float, float],
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop n-dimensional data along a specified axis.

    Parameters:
        coord_array (np.ndarray): 1D coordinate array for the specified axis
        nd_data (np.ndarray): N-dimensional data array
        section (tuple[float, float]): Section boundaries as (min_val, max_val)
        axis (int): Axis along which to crop (default: 0)

    Returns:
        tuple: (cropped_coord_array, cropped_nd_data)

    Raises:
        ValueError: If coordinate array length doesn't match data shape along specified axis
    """
    ### Validate input dimensions
    if coord_array.ndim != 1:
        raise ValueError("Coordinate array must be 1-dimensional")

    if nd_data.shape[axis] != len(coord_array):
        raise ValueError(
            f"Data shape along axis {axis} ({nd_data.shape[axis]}) "
            f"does not match coordinate array length ({len(coord_array)})"
        )

    coord_min, coord_max = section

    ### Validate coordinates are within data range
    coord_min = max(coord_min, np.min(coord_array))
    coord_max = min(coord_max, np.max(coord_array))

    ### Find indices within the specified section
    indices = np.where((coord_array >= coord_min) & (coord_array <= coord_max))[0]

    ### Ensure indices are within array bounds
    indices = indices[indices < len(coord_array)]

    ### Crop coordinate array
    cropped_coords = coord_array[indices]

    ### Crop data along specified axis using advanced indexing
    cropped_data = np.take(nd_data, indices, axis=axis)

    return cropped_coords, cropped_data


# =============================
# NEW INTERNAL HELPERS (1D/2D LABEL + COMPONENT LOGIC)
# =============================
def _style_for_component(component: str) -> Tuple[str, str]:
    """Return (color, linestyle) for a given 1D component.

    Strategy: distinct color per component; primary solid line style.
    Fallback: first style/color.
    """
    color_map = {"abs": 0, "real": 1, "imag": 2, "phase": 3}
    idx = color_map.get(component, 0)
    color = list(COLORS.keys())[idx]
    linestyle = LINE_STYLES[0]
    return color, linestyle


def _resolve_1d_labels_and_component(
    data: np.ndarray,
    domain: str,
    component: str,
    function_symbol: str,
    provided_title: str | None,
) -> Tuple[np.ndarray, str, str, str, str]:
    """Process complex 1D data component + build labels.

    Returns: (y_data, legend_label, y_label, x_label, final_title)
    """
    if domain not in ("time", "freq"):
        raise ValueError("Domain not recognized. Use 'time' or 'freq'.")

    in_time = domain == "time"
    var_symbol = "t" if in_time else "\omega"
    x_label = r"$t \, [\text{fs}]$" if in_time else r"$\omega$ [$10^4$ cm$^{-1}$]"
    default_title = f"{function_symbol} in {'Time' if in_time else 'Frequency'} Domain"
    title = provided_title or default_title

    # Compute component
    if component == "abs":
        y_data = np.abs(data)
        base = f"|{function_symbol}({var_symbol})|"
    elif component == "real":
        y_data = np.real(data)
        base = f"\mathrm{{Re}}[{function_symbol}({var_symbol})]"
    elif component == "imag":
        y_data = np.imag(data)
        base = f"\mathrm{{Im}}[{function_symbol}({var_symbol})]"
    elif component == "phase":
        y_data = np.angle(data)
        base = f"\mathrm{{Arg}}[{function_symbol}({var_symbol})]"
    else:
        raise ValueError("Component must be one of 'abs','real','imag','phase'.")

    label = f"${base}$"
    ylabel = label if component != "phase" else f"${base}$ [rad]"
    return y_data, label, ylabel, x_label, title


def _component_2d_data(data: np.ndarray, component: str) -> Tuple[np.ndarray, str]:
    """Return transformed 2D data + base title according to component."""
    if component not in ("real", "imag", "abs", "phase"):
        raise ValueError("Invalid component for 2D plot.")
    if component == "real":
        return np.real(data), r"$\text{2D Real }$"
    if component == "imag":
        return np.imag(data), r"$\text{2D Imag }$"
    if component == "abs":
        return np.abs(data), r"$\text{2D Abs }$"
    return np.angle(data), r"$\text{2D Phase }$"  # phase


def _domain_2d_labels(domain: str) -> Tuple[str, str, str, str]:
    """Return (colormap, x_label, y_label, title_suffix) for domain."""
    if domain not in ("time", "freq"):
        raise ValueError("Invalid domain. Use 'time' or 'freq'.")
    if domain == "time":
        return (
            "viridis",
            _axis_label_time_det(),
            _axis_label_time_coh(),
            r"$\text{Time domain signal}$",
        )
    return (
        "plasma",
        _axis_label_freq_det(),
        _axis_label_freq_coh(),
        r"$\text{Spectrum}$",
    )


# =============================
# AXIS LABEL HELPERS (central definitions)
# =============================
def _axis_label_time() -> str:
    return r"$t$ [fs]"


def _axis_label_freq() -> str:
    return r"$\omega$ [$10^4$ cm$^{-1}$]"


def _axis_label_time_det() -> str:
    return r"$t_{\text{det}}$ [fs]"


def _axis_label_time_coh() -> str:
    return r"$t_{\text{coh}}$ [fs]"


def _axis_label_freq_det() -> str:
    return r"$\omega_{\text{det}}$ [$10^4$ cm$^{-1}$]"


def _axis_label_freq_coh() -> str:
    return r"$\omega_{\text{coh}}$ [$10^4$ cm$^{-1}$]"


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


def add_text_box(ax, kwargs: dict, position: tuple = (0.98, 0.98), fontsize: int = 11):
    """
    Add a text box with additional parameters to a plot.

    Parameters:
        ax (matplotlib.axes.Axes): Axes object to add the text box to.
        kwargs (dict): Dictionary of parameters to display in the text box.
        position (tuple): Position of the text box in axis coordinates (default: top-right).
        fontsize (int): Font size for the text box (default: 11).
    """
    if kwargs:
        text_lines = []
        for key, value in kwargs.items():
            if isinstance(value, float):
                text_lines.append(
                    f"{key}: {value:.3g}"
                )  # Format floats to 3 significant digits
            elif isinstance(value, (int, str)):
                text_lines.append(
                    f"{key}: {value}"
                )  # Add integers and strings directly
            elif isinstance(value, np.ndarray):
                text_lines.append(
                    f"{key}: array(shape={value.shape})"
                )  # Show shape for numpy arrays
            else:
                # Convert other types to string and escape LaTeX special characters
                safe_str = (
                    str(value)
                    .replace("_", "\_")
                    .replace("^", "\^")
                    .replace("{", "\{")
                    .replace("}", "\}")
                )
                text_lines.append(f"{key}: {safe_str}")

        info_text = "\n".join(text_lines)
        ax.text(
            position[0],
            position[1],
            info_text,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.05, edgecolor="black"),
        )
