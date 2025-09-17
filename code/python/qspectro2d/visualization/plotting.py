from matplotlib.colors import TwoSlopeNorm
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Union, Tuple

from plotstyle import init_style, COLORS, LINE_STYLES
from plotstyle.style import simplify_figure_text

from qspectro2d.core.laser_system.laser_fcts import (
    pulse_envelopes,
    _single_pulse_envelope,
    e_pulses,
    epsilon_pulses,
)
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence

init_style()


def plot_pulse_envelopes(
    times: np.ndarray, pulse_seq: LaserPulseSequence, ax=None, show_legend=True
):
    """
    Plot the combined pulse envelope over time for N pulses using LaserPulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        tuple: (fig, ax) - Figure and axes objects with the plot.
    """
    # Calculate the combined envelope over time
    envelope = pulse_envelopes(times, pulse_seq)

    # Create figure and axis if not provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()  # Plot combined envelope
        created_fig = True
    else:
        fig = ax.figure
    ax.plot(
        times,
        envelope,
        label=r"$\text{Combined Envelope}$",
        linestyle=LINE_STYLES[0],
        alpha=0.8,
        color=COLORS[0],
    )  # Styles for individual pulses will cycle
    n_styles = len(LINE_STYLES)
    n_colors = len(COLORS)

    # Plot individual envelopes and annotations (support any number of pulses)
    for idx, pulse in enumerate(pulse_seq.pulses):

        # get the individual envelope from the pulse_envelopes function
        individual_envelope = _single_pulse_envelope(times, pulse)
        t_peak = pulse.pulse_peak_time
        Delta_width = pulse.pulse_fwhm_fs
        ax.plot(
            times,
            individual_envelope,
            label=rf"$\text{{Pulse {idx + 1}}}$",
            linestyle=LINE_STYLES[(idx + 1) % n_styles],  # avoid reusing 0 used above
            alpha=0.6,
            color=COLORS[(idx + 1) % n_colors],
        )  # Annotate pulse key points
        ax.axvline(
            t_peak - Delta_width,
            linestyle=LINE_STYLES[3],
            label=rf"$t_{{peak, {idx + 1}}} - \Delta_{{{idx + 1}}}$",
            alpha=0.4,
            color=COLORS[(idx + 1) % n_colors],
        )
        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[0],
            label=rf"$t_{{peak, {idx + 1}}}$",
            alpha=0.8,
            color=COLORS[(idx + 1) % n_colors],
            linewidth=2,
        )
        ax.axvline(
            t_peak + Delta_width,
            linestyle=LINE_STYLES[3],
            label=rf"$t_{{peak, {idx + 1}}} + \Delta_{{{idx + 1}}}$",
            alpha=0.4,
            color=COLORS[(idx + 1) % n_colors],
        )

    # Final plot labeling
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Envelope Amplitude")
    ax.set_title(r"Pulse Envelopes for Up to Three Pulses")
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if created_fig:
        plt.close(fig)
    simplify_figure_text(fig)
    return fig, ax


def plot_e_pulses(times: np.ndarray, pulse_seq: LaserPulseSequence, ax=None, show_legend=True):
    """
    Plot the RWA electric field (envelope only) over time for N pulses using LaserPulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        tuple: (fig, ax) - Figure and axes objects with the plot.
    """
    # Calculate the RWA electric field over time
    E_field = np.array([e_pulses(t, pulse_seq) for t in times])

    # Create figure and axis if not provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Plot real and imaginary parts
        created_fig = True
    else:
        fig = ax.figure
    ax.plot(
        times,
        np.real(E_field),
        label=r"$\mathrm{Re}[E(t)]$",
        linestyle=LINE_STYLES[0],
        color=COLORS[0],
    )
    ax.plot(
        times,
        np.imag(E_field),
        label=r"$\mathrm{Im}[E(t)]$",
        linestyle=LINE_STYLES[1],
        color=COLORS[1],
    )

    # Styles for any number of pulses (cycle through colors)
    n_colors = len(COLORS)

    # Plot pulse peak times for all pulses
    for idx, pulse in enumerate(pulse_seq.pulses):
        t_peak = pulse.pulse_peak_time
        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[3],  # "dotted"
            label=rf"$t_{{peak, {idx + 1}}}$",
            color=COLORS[(idx + 2) % n_colors],  # offset to avoid field line colors
        )

    # Final plot labeling
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Electric Field (RWA)")
    ax.set_title(r"RWA Electric Field Components")
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if created_fig:
        plt.close(fig)
    simplify_figure_text(fig)
    return fig, ax


def plot_epsilon_pulses(
    times: np.ndarray, pulse_seq: LaserPulseSequence, ax=None, show_legend=True
):
    """
    Plot the full electric field (with carrier) over time for N pulses using LaserPulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        tuple: (fig, ax) - Figure and axes objects with the plot.
    """
    # Calculate the full electric field over time
    Epsilon_field = np.array([epsilon_pulses(t, pulse_seq) for t in times])
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True
    else:
        fig = ax.figure
    ax.plot(
        times,
        np.real(Epsilon_field),
        label=r"$\mathrm{Re}[\varepsilon(t)]$",
        linestyle=LINE_STYLES[0],
        color=COLORS[3],
    )
    ax.plot(
        times,
        np.imag(Epsilon_field),
        label=r"$\mathrm{Im}[\varepsilon(t)]$",
        linestyle=LINE_STYLES[1],
        color=COLORS[4],
    )
    ax.plot(
        times,
        np.abs(Epsilon_field),
        label=r"$|\varepsilon(t)|$",
        linestyle=LINE_STYLES[2],
        color=COLORS[5],
    )
    n_colors = len(COLORS)
    for idx, pulse in enumerate(pulse_seq.pulses):
        t_peak = pulse.pulse_peak_time
        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[3],
            label=rf"$t_{{peak, {idx + 1}}}$",
            color=COLORS[idx % n_colors],
        )
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Electric Field (Full)")
    ax.set_title(r"Full Electric Field with Carrier")
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if created_fig:
        plt.close(fig)
    simplify_figure_text(fig)
    return fig, ax


def plot_all_pulse_components(times: np.ndarray, pulse_seq: LaserPulseSequence, figsize=(15, 12)):
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
    plot_pulse_envelopes(times, pulse_seq, ax=axes[0])

    # Plot RWA electric field
    plot_e_pulses(times, pulse_seq, ax=axes[1])

    # Plot full electric field
    plot_epsilon_pulses(times, pulse_seq, ax=axes[2])

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
    observable_strs: list[str],
    rwa_sl: bool = False,
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
    if rwa_sl:
        field_func = e_pulses
    else:
        field_func = epsilon_pulses

    # Calculate total electric field
    E0 = pulse_seq.E0
    E_total = np.array([field_func(t, pulse_seq) / E0 for t in times_plot])

    # Create plot with appropriate size
    fig, axes = plt.subplots(len(datas) + 1, 1, figsize=(14, 2 + 2 * len(datas)), sharex=True)

    # Plot electric field
    axes[0].plot(
        times_plot,
        np.real(E_total),
        color=COLORS[0],
        linestyle=LINE_STYLES[0],
        label=r"$\mathrm{Re}[E(t)]$",
    )
    axes[0].plot(
        times_plot,
        np.imag(E_total),
        color=COLORS[1],
        linestyle=LINE_STYLES[1],
        label=r"$\mathrm{Im}[E(t)]$",
    )
    axes[0].set_ylabel(r"$E(t) / E_0$")
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Plot expectation values
    for idx, data in enumerate(datas):
        ax = axes[idx + 1]
        if idx == len(observable_strs):
            observable_str = r"\text{Pol}"
            # also Plot imaginary part, because the polarization is complex
            ax.plot(
                times_plot,
                np.imag(data),
                color=COLORS[1],
                linestyle=LINE_STYLES[1],
                label=r"$\mathrm{Im}" + observable_str + r" \rangle$",
            )
        else:
            observable_str = observable_strs[idx]

        # Plot real part
        ax.plot(
            times_plot,
            np.real(data),
            color=COLORS[0],
            linestyle=LINE_STYLES[0],
            label=r"$\mathrm{Re}\langle " + observable_str + r" \rangle$",
        )

        ax.set_ylabel(r"$\langle " + observable_str + r" \rangle$")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    add_text_box(ax=axes[0], kwargs=kwargs)

    # Set x-label only on the bottom subplot
    axes[-1].set_xlabel(r"$t\,/\,\mathrm{fs}$")

    plt.tight_layout()
    simplify_figure_text(fig)
    plt.close(fig)
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
    fig, ax1 = plt.subplots()

    # Plot main components on primary y-axis
    P_3 = P_full - P_only0 - P_only1 - P_only2
    ax1.plot(
        times,
        np.abs(P_full),
        label=r"$full$",
        color=COLORS[0],
        linestyle=LINE_STYLES[0],
    )
    ax1.plot(
        times,
        np.abs(P_only0),
        label=r"$0$",
        color=COLORS[1],
        linestyle=LINE_STYLES[1],
    )
    ax1.plot(
        times,
        np.abs(P_only1),
        label=r"$1$",
        color=COLORS[2],
        linestyle=LINE_STYLES[2],
    )
    ax1.plot(
        times,
        np.abs(P_only2),
        label=r"$2$",
        color=COLORS[3],
        linestyle=LINE_STYLES[3],
    )
    ax1.set_xlabel(r"$t_{\mathrm{det}}$ [fs]")
    ax1.set_ylabel(r"$|P_i(t)|$", color=COLORS[0])
    ax1.tick_params(axis="y", labelcolor=COLORS[0])

    # Create secondary y-axis for P^3
    ax2 = ax1.twinx()
    ax2.plot(
        times,
        np.abs(P_3),
        color=COLORS[4],
        linestyle=LINE_STYLES[4],
    )
    ax2.set_ylabel(r"$|P^{3}(t)|$", color=COLORS[4])
    ax2.tick_params(axis="y", labelcolor=COLORS[4])

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", bbox_to_anchor=(1, 0.5))

    ax1.set_title(title)

    # Add additional parameters as a text box if provided
    add_text_box(ax=ax1, kwargs=kwargs)

    plt.tight_layout()
    simplify_figure_text(fig)
    plt.close(fig)
    return fig


def plot_1d_el_field(
    axis_det: np.ndarray,
    data: np.ndarray,
    domain: Literal["time", "freq"] = "time",
    component: Literal["real", "img", "abs", "phase"] = "real",
    title: str | None = None,
    section: Union[tuple[float, float], None] = None,
    function_symbol: str = "S",
    figsize: Tuple[float, float] = (6.5, 4.0),
    normalize: bool = True,
    **kwargs: dict,
) -> plt.Figure:
    """Plot 1D complex data (time or frequency domain).

    Inputs: axis_det (1d), complex data (same length), domain selector, component selector
    Output: matplotlib Figure object
    Normalization: optional (default True) to max absolute amplitude
    Cropping: optional via section=(min,max)
    """
    fig = plt.figure(figsize=figsize)

    # CROP + NORMALIZE

    if section is not None:
        axis_det, data = crop_nd_data_along_axis(axis_det, data, section=section, axis=0)
    if normalize:
        max_abs = np.abs(data).max()
        if max_abs == 0:
            raise ValueError("Data array is all zeros, cannot normalize.")
        data = data / max_abs

    # COMPONENT HANDLING
    y_data, label, ylabel, x_label, final_title = _resolve_1d_labels_and_component(
        data=data,
        domain=domain,
        component=component,
        function_symbol=function_symbol,
        provided_title=title,
    )

    # STYLE
    color, linestyle = _style_for_component(component)

    # PLOT
    plt.plot(axis_det, y_data, label=label, color=color, linestyle=linestyle)
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.title(final_title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    add_text_box(ax=plt.gca(), kwargs=kwargs)
    plt.tight_layout()
    simplify_figure_text(fig)
    plt.close(fig)
    return fig


def plot_2d_el_field(
    axis_det: np.ndarray,  # detection axis
    axis_coh: np.ndarray,  # coherence axis
    data: np.ndarray,  # complex 2D array
    t_wait: float = np.inf,
    domain: Literal["time", "freq"] = "time",
    component: Literal["real", "img", "abs", "phase"] = "real",
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
    component : {"real", "img", "abs", "phase"}, default "real"
        Component of complex data to plot. Used for both title and data processing.
    use_custom_colormap : bool, default False
        If True, uses custom red-white-blue colormap centered at zero.
        Automatically set to True for "real", "img", and "phase" components.
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

    # VALIDATE INPUT

    if data.ndim != 2 or data.shape[0] != len(axis_coh) or data.shape[1] != len(axis_det):
        raise ValueError(
            f"Data shape {data.shape} does not match axis_det ({len(axis_det)}) and axis_coh ({len(axis_coh)}) dimensions."
        )

    import warnings

    if len(axis_det) == 0 or len(axis_coh) == 0 or data.size == 0:
        warnings.warn(
            f"Empty arrays in plot_2d_el_field: axis_det={axis_det.shape}, axis_coh={axis_coh.shape}, data={data.shape}"
        )

    # Convert axes to real (robust against minor numerical complex drift)
    axis_det = np.real(axis_det)
    axis_coh = np.real(axis_coh)
    data = np.asarray(data, dtype=np.complex128)

    # SECTION CROPPING

    if section is not None:
        # expect list[(coh_min, coh_max),(det_min, det_max)]
        axis_coh, data = crop_nd_data_along_axis(axis_coh, data, section=section[0], axis=0)
        axis_det, data = crop_nd_data_along_axis(axis_det, data, section=section[1], axis=1)
    if normalize:
        max_abs = np.abs(data).max()
        if max_abs == 0:
            raise ValueError("Data array is all zeros, cannot normalize.")
        data = data / max_abs

    # ENSURE MONOTONIC AXES ORIENTATIONS MATCH DATA

    # imshow with an extent assumes the first element of each axis maps to the
    # lower-left corner (when origin='lower'). If axes are descending, flip data
    # along that axis and sort the axis to be ascending so diagonal features
    # remain along y=x and no 90° rotation occurs.
    if axis_det[0] > axis_det[-1]:
        axis_det = axis_det[::-1]
        data = data[:, ::-1]
    if axis_coh[0] > axis_coh[-1]:
        axis_coh = axis_coh[::-1]
        data = data[::-1, :]

    # SET PLOT LABELS AND COLORMAP

    data, title_base = _component_2d_data(data=data, component=component)
    colormap, x_title, y_title, domain_suffix = _domain_2d_labels(domain=domain)
    title = title_base + domain_suffix
    if t_wait != np.inf:
        title += rf"$\ (T = {t_wait:.2f}\,\text{{fs}})$"

    # CUSTOM COLORMAP FOR ZERO-CENTERED DATA

    norm = None
    # For real and imag data, use red-white-blue colormap by default
    if component in ("real", "img", "phase"):
        use_custom_colormap = True
    elif component == "abs":
        use_custom_colormap = False

    if use_custom_colormap:
        vmax = np.max(np.abs(data))
        vmin = -vmax
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

    # GENERATE FIGURE

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
    ax.set_title(title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    # Add additional parameters as a text box if provided
    add_text_box(ax=ax, kwargs=kwargs)

    fig.tight_layout()

    """# Add a border around the plot for better visual definition plt.gca().spines["top"].set_visible(True); plt.gca().spines["bottom"].set_linewidth(1.5)"""

    simplify_figure_text(fig)
    plt.close(fig)  # keep figure open for further user modification if desired
    return fig


# HELPER FUNCTIONS
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


# NEW INTERNAL HELPERS (1D/2D LABEL + COMPONENT LOGIC)
def _style_for_component(component: str) -> Tuple[str, str]:
    """Return (color, linestyle) for a given 1D component.

    Strategy: distinct color per component; primary solid line style.
    Fallback: first style/color.
    """
    color_map = {"abs": 0, "real": 1, "img": 2, "phase": 3}
    idx = color_map.get(component, 0)
    color = COLORS[idx]
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
    elif component == "img":
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
    if component not in ("real", "img", "abs", "phase"):
        raise ValueError("Invalid component for 2D plot.")
    if component == "real":
        return np.real(data), r"$\text{2D Real }$"
    if component == "img":
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
            r"$t_{\text{det}}$ [fs]",
            r"$t_{\text{coh}}$ [fs]",
            r"$\text{Time domain signal}$",
        )
    return (
        "plasma",
        r"$\omega_{\text{det}}$ [$10^4$ cm$^{-1}$]",
        r"$\omega_{\text{coh}}$ [$10^4$ cm$^{-1}$]",
        r"$\text{Spectrum}$",
    )


def add_custom_contour_lines(
    x: np.ndarray, y: np.ndarray, data: np.ndarray, component: str, level_count: int = 8
) -> None:
    """
    Add custom contour lines to a 2D plot with different styles for positive/negative values.

    Parameters:
        x (np.ndarray): X-axis coordinate array
        y (np.ndarray): Y-axis coordinate array
        data (np.ndarray): 2D data array to contour
        component (str): Data component type ("real", "img", "phase", "abs")
        level_count (int): Number of contour levels in each region (positive/negative)
    """
    ### Add contour lines with different styles for positive and negative values
    if component in ("real", "img", "phase"):
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


def add_text_box(
    ax,
    kwargs: dict,
    position: tuple = (0.98, 0.98),
    fontsize: int = 11,
    coords: Literal["axes", "figure"] = "axes",
):
    """
    Add a text box with additional parameters without affecting subplot layout.

    Parameters:
        ax (matplotlib.axes.Axes): Axes object (also used to access the parent Figure).
        kwargs (dict): Dictionary of parameters to display in the text box.
        position (tuple): Position of the text box in the chosen coordinate system.
                          For ``coords='axes'`` this is in axis coordinates (default top-right).
                          For ``coords='figure'`` this is in figure coordinates.
        fontsize (int): Font size for the text box (default: 11).
        coords (Literal["axes","figure"]): Coordinate system to place the text in.
            - "axes": anchored to the given Axes (default; backwards compatible)
            - "figure": anchored to the whole Figure (fully independent of Axes)

    Notes:
        - The created text artist is marked ``in_layout=False`` so that ``tight_layout``
          or ``constrained_layout`` ignore it and do not shrink other subplots.
        - ``clip_on=False`` is used so the text can extend beyond the Axes box if desired.
    """
    if kwargs:
        text_lines = []
        for key, value in kwargs.items():
            if isinstance(value, float):
                # Format floats to 3 significant digits
                text_lines.append(f"{key}: {value:.3g}")
            elif isinstance(value, (int, str)):
                # Add integers and strings directly
                text_lines.append(f"{key}: {value}")
            elif isinstance(value, np.ndarray):
                # Show shape for numpy arrays
                text_lines.append(f"{key}: array(shape={value.shape})")
            else:
                # TODO insert this into my plotstyle package
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

        # Create the text artist either in axes or figure coordinates
        if coords == "figure":
            fig = ax.figure
            artist = fig.text(
                position[0],
                position[1],
                info_text,
                transform=fig.transFigure,
                fontsize=fontsize,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.05, edgecolor="black"),
            )
        else:
            artist = ax.text(
                position[0],
                position[1],
                info_text,
                transform=ax.transAxes,
                fontsize=fontsize,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.05, edgecolor="black"),
                clip_on=False,  # allow text to extend beyond axes region
            )

        # Ensure layout engines ignore this artist so it doesn't shrink subplots
        try:
            artist.set_in_layout(False)
        except Exception:
            # Older Matplotlib versions may not have in_layout on Text; ignore gracefully
            pass
