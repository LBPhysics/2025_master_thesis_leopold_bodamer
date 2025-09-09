"""
Data plotting functions for qspectro2d.

This module provides high-level plotting functions for 1D and 2D
spectroscopy data with standardized formatting and output.
"""

# =============================
# IMPORTS
# =============================
import matplotlib.pyplot as plt
import gc
from typing import cast, TYPE_CHECKING
import numpy as np

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
from plotstyle import init_style, save_fig

init_style()
from qspectro2d.core.bath_system.bath_fcts import extract_bath_parameters


# Helper to collect paths from save_fig (which may return a single path or a list of paths)
def _collect_saved_paths(accumulator, saved):
    if isinstance(saved, (list, tuple, set)):  # save_fig may return a list
        accumulator.extend(saved)
    else:
        accumulator.append(saved)


if TYPE_CHECKING:
    # Imported only for static type checking / IDE autocomplete
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
    from qspectro2d.core.simulation import SimulationConfig
    from qutip import BosonicEnvironment


# =============================
# PLOTTING FUNCTIONS
# =============================
def plot_1d_data(
    loaded_data_and_info: dict,
    plot_config: dict,
) -> None:
    """Plot 1D spectroscopy data using standardized data structure.

    Args:
        loaded_data_and_info: Dictionary containing data, axes, system, and config from load_data_from_abs_path
        plot_config: Plotting configuration
    """
    # Validate and extract data structure
    axes = loaded_data_and_info["axes"]
    t_det_vals = axes.get("t_det")  # detection times
    if axes.get("t_coh") is not None:
        print(
            "❌ 1D data should not have 't_coh' axis. Check your data structure.",
            flush=True,
        )
        return

    # Explicitly cast for static typing & IDE autocomplete
    system = cast("AtomicSystem", loaded_data_and_info["system"])
    w0 = system._frequencies_fs[0]
    bath_env = cast("BosonicEnvironment", loaded_data_and_info["bath"])
    bath_dict = extract_bath_parameters(bath_env, w0)

    laser = cast("LaserPulseSequence", loaded_data_and_info["laser"])
    # Require new key 'sim_config'
    sim_config = cast("SimulationConfig", loaded_data_and_info["sim_config"])
    signal_types = sim_config.signal_types

    # Collect data arrays according to requested signals
    datas: list[np.ndarray] = []
    for signal_type in signal_types:
        if signal_type in loaded_data_and_info:
            datas.append(loaded_data_and_info[signal_type])

    laser_dict = {
        k: v for k, v in laser.to_dict().items() if k != "pulses"
    }  # Exclude "pulses" key

    # Combine dictionaries
    dict_combined = {**system.to_dict(), **bath_dict, **laser_dict}
    dict_combined.update(sim_config.to_dict())

    print(f"   Time points: {len(t_det_vals)}")
    print(f"   Time range: {t_det_vals[0]:.1f} to {t_det_vals[-1]:.1f} fs")

    spectral_components_to_plot = plot_config.get(
        "spectral_components_to_plot", ["imag", "real"]
    )
    extend_for = plot_config.get("extend_for", (1, 1))

    ### Plot time domain data
    if plot_config.get("plot_time_domain", True):
        print("📊 Plotting time domain data...")
        time_domain_comps = ["real", "abs", "imag", "phase"]
        list_of_saved_paths = []
        try:
            for component in time_domain_comps:
                for data, signal_type in zip(datas, signal_types):
                    dict_combined_with_signal = dict_combined.copy()
                    dict_combined_with_signal["signal_type"] = signal_type
                    fig = plot_1d_el_field(
                        axis_det=t_det_vals,
                        data=data,
                        domain="time",
                        component=component,
                        function_symbol=r"$E_{k_s}$",
                        **dict_combined_with_signal,
                    )
                    filename = generate_unique_plot_filename(
                        system=system,
                        sim_config=sim_config,
                        domain="time",
                        component=component,
                    )

                    saved = save_fig(fig, filename=filename)
                    _collect_saved_paths(list_of_saved_paths, saved)  # flatten
            print("✅ 1D Time domain plots completed!, figs saved under:\n")
            for path in list_of_saved_paths:
                print(f" - {str(path)}")
        except Exception as e:
            print(f"❌ Error in 1D time domain plotting: {e}")

    ### Plot frequency domain data
    section_2d = plot_config.get("section", [(0, 3), (0, 3)])
    section = section_2d[0]  # Only use the first section for 1D data
    if plot_config.get("plot_frequency_domain", True):
        print("📊 Plotting 1D frequency domain data...")
        datas_freq: list[np.ndarray] = []
        signal_types_freq: list[str] = []
        for i, (data, signal_type) in enumerate(zip(datas, signal_types)):
            try:
                extended_t_dets, extended_data = extend_time_axes(
                    data=data,
                    t_det=t_det_vals,
                    pad_t_det=extend_for,
                )
                datas_freq.append(extended_data)
                signal_types_freq.append(signal_type)
                print(f"   ✅ Successfully processed data {i+1}/{len(datas)} ({signal_type})")
            except Exception as e:
                print(f"   ❌ Error processing data {i+1}/{len(datas)} ({signal_type}): {e}")
                # Continue with next data array
                continue

        if not datas_freq:
            print("❌ No data could be processed for frequency domain plotting")
            return

        try:
            # Compute FFT with proper signal orientation
            nu_dets, data_freq = compute_1d_fft_wavenumber(
                extended_t_dets, datas_freq, signal_types=signal_types_freq
            )

            # Plot each spectral component separately
            list_of_saved_paths = []
            try:
                for component in spectral_components_to_plot:
                    fig = plot_1d_el_field(
                        axis_det=nu_dets,
                        data=data_freq,
                        domain="freq",
                        use_custom_colormap=True,
                        component=component,
                        section=section,
                    )
                    filename = generate_unique_plot_filename(
                        system=system,
                        sim_config=sim_config,
                        domain="freq",
                        component=component,
                    )

                    saved = save_fig(fig, filename=filename)
                    _collect_saved_paths(list_of_saved_paths, saved)  # flatten
                print("✅ 2D Frequency domain plots completed!, figs saved under:\n")
                for path in list_of_saved_paths:
                    print(f" - {str(path)}")
            except Exception as e:
                print(f"❌ Error plotting 2D {component} component: {e}")
        except Exception as e:
            print(f"❌ Error in 2D frequency domain processing: {e}")

    # Clean up memory
    plt.close("all")
    gc.collect()


def plot_2d_data(
    loaded_data_and_info: dict,
    plot_config: dict,
) -> None:
    """Plot 2D spectroscopy data using standardized data structure.

    Args:
        loaded_data_and_info: Dictionary containing data, axes, system, and config from load_data_from_abs_path
        plot_config: Plotting configuration
    """
    # Validate and extract data structure
    axes = loaded_data_and_info["axes"]
    t_det_vals = axes.get("t_det")  # detection times
    t_coh_vals = axes["t_coh"]  # coherence times
    
    # Explicitly cast objects for static typing & autocomplete (mirror plot_1d_data)
    system = cast("AtomicSystem", loaded_data_and_info["system"])
    w0 = system._frequencies_fs[0]
    bath_env = cast("BosonicEnvironment", loaded_data_and_info["bath"])
    bath_dict = extract_bath_parameters(bath_env, w0)
    laser = cast("LaserPulseSequence", loaded_data_and_info["laser"])
    sim_config = cast("SimulationConfig", loaded_data_and_info["sim_config"])
    laser_dict = {
        k: v for k, v in laser.to_dict().items() if k != "pulses"
    }  # Exclude "pulses" key

    # Combine dictionaries
    dict_combined = {**system.to_dict(), **bath_dict, **laser_dict}
    dict_combined.update(sim_config.to_dict())

    # Get configuration values
    spectral_components_to_plot = plot_config.get(
        "spectral_components_to_plot", ["abs"]
    )
    extend_for = plot_config.get("extend_for", (1, 1))
    section = plot_config.get("section", None)

    signal_types = sim_config.signal_types

    print(f"✅ 2D data loaded successfully! Using signals: {signal_types}")

    # Collect data arrays according to requested signals
    datas: list[np.ndarray] = []
    for signal_type in signal_types:
        if signal_type in loaded_data_and_info:
            datas.append(loaded_data_and_info[signal_type])

    ### Plot time domain data
    if plot_config.get("plot_time_domain", True):
        print("📊 Plotting 2D time domain data...")
        time_domain_comps = ["real", "abs", "imag", "phase"]
        list_of_saved_paths = []
        try:
            for component in time_domain_comps:
                for data, signal_type in zip(datas, signal_types):
                    dict_combined_with_signal = dict_combined.copy()
                    dict_combined_with_signal["signal_type"] = signal_type
                    fig = plot_2d_el_field(
                        axis_det=t_det_vals,
                        axis_coh=t_coh_vals,
                        data=data,
                        domain="time",
                        use_custom_colormap=True,
                        component=component,
                        **dict_combined_with_signal,
                    )

                    filename = generate_unique_plot_filename(
                        system=system,
                        sim_config=sim_config,
                        domain="time",
                        component=component,
                    )

                    saved = save_fig(fig, filename=filename)
                    _collect_saved_paths(list_of_saved_paths, saved)  # flatten
            print("✅ 2D Time domain plots completed!, figs saved under:\n")
            for path in list_of_saved_paths:
                print(f" - {str(path)}")
        except Exception as e:
            print(f"❌ Error in 2D time domain plotting: {e}")

    ### Handle frequency domain processing
    if plot_config.get("plot_frequency_domain", True):
        print("📊 Plotting 2D frequency domain data...")
        datas_freq: list[np.ndarray] = []
        signal_types_freq: list[str] = []
        for i, (data, signal_type) in enumerate(zip(datas, signal_types)):
            try:
                extended_t_dets, extended_t_cohs, extended_data = extend_time_axes(
                    data=data,
                    t_det=t_det_vals,
                    t_coh=t_coh_vals,
                    pad_t_det=extend_for,
                    pad_t_coh=extend_for,
                )
                datas_freq.append(extended_data)
                signal_types_freq.append(signal_type)
                print(f"   ✅ Successfully processed data {i+1}/{len(datas)} ({signal_type})")
            except Exception as e:
                print(f"   ❌ Error processing data {i+1}/{len(datas)} ({signal_type}): {e}")
                # Continue with next data array
                continue

        if not datas_freq:
            print("❌ No data could be processed for frequency domain plotting")
            return

        try:
            nu_dets, nu_cohs, data_freq = compute_2d_fft_wavenumber(
                extended_t_dets, extended_t_cohs, datas_freq, signal_types=signal_types_freq
            )

            # Plot each spectral component separately
            list_of_saved_paths = []
            try:
                for component in spectral_components_to_plot:
                    fig = plot_2d_el_field(
                        axis_det=nu_dets,
                        axis_coh=nu_cohs,
                        data=data_freq,
                        domain="freq",
                        use_custom_colormap=True,
                        component=component,
                        section=section,
                    )
                    filename = generate_unique_plot_filename(
                        system=system,
                        sim_config=sim_config,
                        domain="freq",
                        component=component,
                    )

                    saved = save_fig(fig, filename=filename)
                    _collect_saved_paths(list_of_saved_paths, saved)  # flatten
                print("✅ 2D Frequency domain plots completed!, figs saved under:\n")
                for path in list_of_saved_paths:
                    print(f" - {str(path)}")
            except Exception as e:
                print(f"❌ Error plotting 2D {component} component: {e}")
        except Exception as e:
            print(f"❌ Error in 2D frequency domain processing: {e}")

    # Clean up memory
    plt.close("all")
    gc.collect()
