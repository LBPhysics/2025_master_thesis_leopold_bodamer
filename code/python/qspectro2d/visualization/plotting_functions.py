"""
Data plotting functions for qspectro2d.

This module provides high-level plotting functions for 1D and 2D
spectroscopy data with standardized formatting and output.
"""

import matplotlib.pyplot as plt
import gc
from typing import cast
from qutip import BosonicEnvironment

### Project-specific imports
from qspectro2d.visualization.plotting import (
    plot_1d_el_field,
    plot_2d_el_field,
)
from qspectro2d.spectroscopy.post_processing import (
    extend_time_axes,
    compute_1d_fft_wavenumber,
    compute_2d_fft_wavenumber,
)
from qspectro2d.utils import generate_unique_plot_filename
from qspectro2d.core.bath_system.bath_fcts import extract_bath_parameters

from plotstyle import init_style, save_fig

init_style()


# Helper to collect paths from save_fig (which may return a single path or a list of paths)
def _collect_saved_paths(accumulator, saved):
    if isinstance(saved, (list, tuple, set)):  # save_fig may return a list
        accumulator.extend(saved)
    else:
        accumulator.append(saved)


def _extract_and_prepare_data(loaded_data_and_info: dict):
    """Extract and prepare common data structures for plotting.

    Args:
        loaded_data_and_info: Dictionary containing data, axes, system, and config.

    Returns:
        Tuple of (system, bath_dict, laser_dict, dict_combined, sim_config, signal_types, datas, axes)
    """
    # Explicitly cast for static typing & IDE autocomplete
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
    from qspectro2d.core.simulation import SimulationConfig

    system = cast(AtomicSystem, loaded_data_and_info["system"])
    w0 = system._frequencies_fs[0]
    bath_env = cast(BosonicEnvironment, loaded_data_and_info["bath"])
    bath_dict = extract_bath_parameters(bath_env, w0)
    laser = cast(LaserPulseSequence, loaded_data_and_info["laser"])
    sim_config = cast(SimulationConfig, loaded_data_and_info["sim_config"])
    laser_dict = {k: v for k, v in laser.to_dict().items() if k != "pulses"}  # Exclude "pulses" key

    # Combine dictionaries
    dict_combined = {**system.to_dict(), **bath_dict, **laser_dict}
    dict_combined.update(sim_config.to_dict())

    signal_types = sim_config.signal_types

    # Collect data arrays according to requested signals
    datas = []
    for signal_type in signal_types:
        if signal_type in loaded_data_and_info:
            datas.append(loaded_data_and_info[signal_type])

    axes = loaded_data_and_info["axes"]

    return system, bath_dict, laser_dict, dict_combined, sim_config, signal_types, datas, axes


def _extend_and_fft(datas, signal_types, axes, extend_for, dimension):
    """Extend time axes and compute FFT based on dimension.

    Args:
        datas: List of data arrays.
        signal_types: List of signal types.
        axes: Dictionary of axes.
        extend_for: Tuple for padding.
        dimension: "1d" or "2d".

    Returns:
        Tuple of (extended_axes, data_freq, signal_types_freq)
    """
    datas_freq = []
    signal_types_freq = []
    t_det_vals = axes["t_det"]

    for i, (data, signal_type) in enumerate(zip(datas, signal_types)):
        try:
            if dimension == "1d":
                extended_t_dets, extended_data = extend_time_axes(
                    data=data,
                    t_det=t_det_vals,
                    pad_t_det=extend_for,
                )
                extended_axes = extended_t_dets
            else:  # 2d
                t_coh_vals = axes["t_coh"]
                extended_t_dets, extended_t_cohs, extended_data = extend_time_axes(
                    data=data,
                    t_det=t_det_vals,
                    t_coh=t_coh_vals,
                    pad_t_det=extend_for,
                    pad_t_coh=extend_for,
                )
                extended_axes = (extended_t_dets, extended_t_cohs)

            datas_freq.append(extended_data)
            signal_types_freq.append(signal_type)
            print(f"   ✅ Successfully processed data {i+1}/{len(datas)} ({signal_type})")
        except Exception as e:
            print(f"   ❌ Error processing data {i+1}/{len(datas)} ({signal_type}): {e}")
            continue

    if not datas_freq:
        raise ValueError("No data could be processed for frequency domain plotting")

    # Compute FFT
    if dimension == "1d":
        nu_dets, data_freq = compute_1d_fft_wavenumber(
            extended_axes, datas_freq, signal_types=signal_types_freq
        )
        freq_axes = nu_dets
    else:  # 2d
        nu_dets, nu_cohs, data_freq = compute_2d_fft_wavenumber(
            extended_axes[0], extended_axes[1], datas_freq, signal_types=signal_types_freq
        )
        freq_axes = (nu_dets, nu_cohs)

    return freq_axes, data_freq, signal_types_freq


def _plot_components(
    datas,
    signal_types,
    axes,
    domain,
    components,
    plot_func,
    dict_combined,
    system,
    sim_config,
    dimension,
    **kwargs,
):
    """Generalized plotting function for time or frequency domain.

    Args:
        datas: List of data arrays.
        signal_types: List of signal types.
        axes: Tuple of axes (det, coh if 2d).
        domain: "time" or "freq".
        components: List of components to plot.
        plot_func: Plotting function (plot_1d_el_field or plot_2d_el_field).
        dict_combined: Combined dictionary for plotting.
        system: AtomicSystem object.
        sim_config: SimulationConfig object.
        dimension: "1d" or "2d".
        **kwargs: Additional keyword arguments for plotting.

    Returns:
        List of saved paths.
    """
    list_of_saved_paths = []
    axis_det, axis_coh = axes if dimension == "2d" else (axes, None)

    for component in components:
        try:
            if domain == "time":
                for data, signal_type in zip(datas, signal_types):
                    dict_combined_with_signal = dict_combined.copy()
                    dict_combined_with_signal["signal_type"] = signal_type
                    if dimension == "1d":
                        fig = plot_func(
                            axis_det=axis_det,
                            data=data,
                            domain=domain,
                            component=component,
                            function_symbol=r"$E_{k_s}$",
                            **dict_combined_with_signal,
                        )
                    else:  # 2d
                        fig = plot_func(
                            axis_det=axis_det,
                            axis_coh=axis_coh,
                            data=data,
                            domain=domain,
                            use_custom_colormap=True,
                            component=component,
                            **dict_combined_with_signal,
                        )
            else:  # freq
                if dimension == "1d":
                    fig = plot_func(
                        axis_det=axis_det,
                        data=datas,  # datas_freq is passed as datas here
                        domain=domain,
                        use_custom_colormap=True,
                        component=component,
                        **kwargs,
                    )
                else:  # 2d
                    fig = plot_func(
                        axis_det=axis_det,
                        axis_coh=axis_coh,
                        data=datas,
                        domain=domain,
                        use_custom_colormap=True,
                        component=component,
                        **kwargs,
                    )

            filename = generate_unique_plot_filename(
                system=system,
                sim_config=sim_config,
                domain=domain,
                component=component,
            )

            saved = save_fig(fig, filename=filename)
            _collect_saved_paths(list_of_saved_paths, saved)
        except Exception as e:
            print(f"❌ Error plotting {dimension.upper()} {domain} {component} component: {e}")

    return list_of_saved_paths


def plot_data(loaded_data_and_info: dict, plot_config: dict, dimension: str) -> None:
    """Generalized plotting function for 1D or 2D spectroscopy data.

    Args:
        loaded_data_and_info: Dictionary containing data, axes, system, and config.
        plot_config: Plotting configuration.
        dimension: "1d" or "2d".
    """
    # Extract and prepare data
    system, _, _, dict_combined, sim_config, signal_types, datas, axes = _extract_and_prepare_data(
        loaded_data_and_info
    )

    # Validate axes based on dimension
    t_det_vals = axes.get("t_det")
    if dimension == "1d":
        if axes.get("t_coh") is not None:
            print("❌ 1D data should not have 't_coh' axis. Check your data structure.", flush=True)
            return
        axes_tuple = t_det_vals
    else:  # 2d
        t_coh_vals = axes.get("t_coh")
        if t_coh_vals is None:
            print("❌ 2D data requires 't_coh' axis. Check your data structure.", flush=True)
            return
        axes_tuple = (t_det_vals, t_coh_vals)

    print(f"   Time points: {len(t_det_vals)}")
    print(f"   Time range: {t_det_vals[0]:.1f} to {t_det_vals[-1]:.1f} fs")
    print(f"✅ {dimension.upper()} data loaded successfully! Using signals: {signal_types}")

    # Get configuration
    spectral_components_to_plot = plot_config.get("spectral_components_to_plot", ["abs"])
    extend_for = plot_config.get("extend_for", (1, 1))
    section = plot_config.get("section", None)
    if dimension == "1d":
        section_2d = plot_config.get("section", [(0, 3), (0, 3)])
        section = section_2d[0]  # Only use the first section for 1D data

    time_domain_comps_to_plot = ["real", "abs", "img", "phase"]

    # Plot time domain
    if plot_config.get("plot_time_domain", True):
        print(f"📊 Plotting {dimension.upper()} time domain data...")
        plot_func = plot_1d_el_field if dimension == "1d" else plot_2d_el_field
        saved_paths = _plot_components(
            datas,
            signal_types,
            axes_tuple,
            "time",
            time_domain_comps_to_plot,
            plot_func,
            dict_combined,
            system,
            sim_config,
            dimension,
        )
        print(f"✅ {dimension.upper()} Time domain plots completed!, figs saved under:\n")
        for path in saved_paths:
            print(f" - {str(path)}")

    # Plot frequency domain
    if plot_config.get("plot_frequency_domain", True):
        print(f"📊 Plotting {dimension.upper()} frequency domain data...")
        try:
            freq_axes, data_freq, signal_types_freq = _extend_and_fft(
                datas, signal_types, axes, extend_for, dimension
            )
            plot_func = plot_1d_el_field if dimension == "1d" else plot_2d_el_field
            saved_paths = _plot_components(
                data_freq,
                signal_types_freq,
                freq_axes,
                "freq",
                spectral_components_to_plot,
                plot_func,
                dict_combined,
                system,
                sim_config,
                dimension,
                section=section,
            )
            print(f"✅ {dimension.upper()} Frequency domain plots completed!, figs saved under:\n")
            for path in saved_paths:
                print(f" - {str(path)}")
        except Exception as e:
            print(f"❌ Error in {dimension.upper()} frequency domain processing: {e}")

    # Clean up memory
    plt.close("all")
    gc.collect()
