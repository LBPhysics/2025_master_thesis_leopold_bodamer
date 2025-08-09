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
from qspectro2d.config.mpl_tex_settings import save_fig
from qspectro2d.core.bath_system.bath_fcts import extract_bath_parameters


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
    data = loaded_data_and_info["data"]
    axes = loaded_data_and_info["axes"]
    t_det_vals = axes["axis1"]  # detection times
    if "axis2" in axes:
        print(
            "❌ 1D data should not have 'axis2' in axes. Check your data structure.",
            flush=True,
        )
        return

    system = loaded_data_and_info["system"]
    bath_dict = loaded_data_and_info["bath"]
    # QuTip Environment - extract parameters

    laser = loaded_data_and_info["laser"]
    info_config = loaded_data_and_info["info_config"]

    laser_dict = {
        k: v for k, v in laser.to_dict().items() if k != "pulses"
    }  # Exclude "pulses" key

    # Combine dictionaries
    dict_combined = {**system.to_dict(), **bath_dict, **laser_dict, **info_config}

    print(f"✅ Data loaded with shape: {data.shape}")
    print(f"   Time points: {len(t_det_vals)}")
    print(f"   Time range: {t_det_vals[0]:.1f} to {t_det_vals[-1]:.1f} fs")

    spectral_components_to_plot = plot_config.get(
        "spectral_components_to_plot", ["imag", "real"]
    )
    extend_for = plot_config.get("extend_for", (1, 1))

    ### Plot time domain data
    if plot_config.get("plot_time_domain", True):
        print("📊 Plotting time domain data...")
        try:
            fig = plot_1d_el_field(
                axis_det=t_det_vals,
                data=data,
                domain="time",
                component="abs",
                function_symbol="E_{k_s}",
                **dict_combined,
            )
            filename = generate_unique_plot_filename(
                system,
                info_config=info_config,
                domain="time",
                component="abs",
            )

            save_fig(
                fig, filename=filename, formats=["png"]
            )  # easy to work with, because data is too big
            fig = plot_1d_el_field(
                axis_det=t_det_vals,
                data=data,
                domain="time",
                component="real",
                function_symbol="E_{k_s}",
                **dict_combined,
            )
            filename = generate_unique_plot_filename(
                system,
                info_config=info_config,
                domain="time",
                component="abs",
            )

            save_fig(
                fig, filename=filename, formats=["png"]
            )  # easy to work with, because data is too big
            print("✅ 1D Time domain plots completed!")
        except Exception as e:
            print(f"❌ Error in time domain plotting: {e}")

    ### Plot frequency domain data
    section_2d = plot_config.get("section", [(0, 2), (0, 2)])
    section = section_2d[0]  # Only use the first section for 1D data
    if plot_config.get("plot_frequency_domain", True):
        print("📊 Plotting frequency domain data...")
        # Extend time axes if needed
        if extend_for != (1, 1):
            extended_x, extended_data = extend_time_axes(
                data=data,
                t_det=t_det_vals,
                pad_t_det=extend_for,
            )
        else:
            extended_x, extended_data = t_det_vals, data

        frequencies, data_fft = compute_1d_fft_wavenumber(extended_x, extended_data)
        # Plot each spectral component separately
        for component in spectral_components_to_plot:
            try:
                fig = plot_1d_el_field(
                    axis_det=frequencies,
                    data=data_fft,
                    domain="freq",
                    component=component,
                    section=section,
                )
                filename = generate_unique_plot_filename(
                    system,
                    info_config=info_config,
                    domain="freq",
                    component=component,
                )
                save_fig(fig, filename=filename)
            except Exception as e:
                print(f"❌ Error plotting {component} component: {e}")

        print("✅ 1D Frequency domain plots completed!")

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
    data = loaded_data_and_info["data"]
    axes = loaded_data_and_info["axes"]
    t_det_vals = axes["axis1"]  # detection times
    if "axis2" not in axes:
        print(
            "❌ 2D data should have 'axis2' in axes for coherence times. Check your data structure.",
            flush=True,
        )
        return
    t_coh_vals = axes["axis2"]  # coherence times
    system = loaded_data_and_info["system"]

    bath_dict = loaded_data_and_info["bath"]
    # bath_dict = extract_bath_parameters(bath)
    laser = loaded_data_and_info["laser"]

    info_config = loaded_data_and_info["info_config"]
    t_wait = info_config["t_wait"]
    laser_dict = {
        k: v for k, v in laser.to_dict().items() if k != "pulses"
    }  # Exclude "pulses" key

    # Combine dictionaries
    dict_combined = {**system.to_dict(), **bath_dict, **laser_dict, **info_config}

    # Get configuration values
    spectral_components_to_plot = plot_config.get(
        "spectral_components_to_plot", ["abs"]
    )
    extend_for = plot_config.get("extend_for", (1, 1))
    section = plot_config.get("section", [(0, 2), (0, 2)])

    print(f"✅ 2D data loaded successfully!")

    ### Plot time domain data
    if plot_config.get("plot_time_domain", True):
        print("📊 Plotting 2D time domain data...")
        # Plot each spectral component separately
        time_domain_comps = ["real"]  # , "imag", "abs", "phase"]
        for component in time_domain_comps:
            try:
                fig = plot_2d_el_field(
                    axis_det=t_det_vals,
                    axis_coh=t_coh_vals,
                    data=data,
                    domain="time",
                    use_custom_colormap=True,
                    component=component,
                    **dict_combined,
                )
                filename = generate_unique_plot_filename(
                    system=system,
                    info_config=info_config,
                    domain="time",
                    component=component,
                )

                save_fig(fig, filename=filename, formats=["png"])  # PNG for large data
            except Exception as e:
                print(f"❌ Error in 2D time domain plotting: {e}")
            print("✅ 2D time domain plots completed!")

    ### Handle frequency domain processing
    if plot_config.get("plot_frequency_domain", True):
        print("📊 Plotting 2D frequency domain data...")
        try:
            # Extend time axes if needed
            if extend_for != (1, 1):
                extended_t_dets, extended_t_cohs, extended_data = extend_time_axes(
                    data=data,
                    t_det=t_det_vals,
                    t_coh=t_coh_vals,
                    pad_t_det=extend_for,
                    pad_t_coh=extend_for,
                )
            else:
                extended_t_dets, extended_t_cohs, extended_data = (
                    t_det_vals,
                    t_coh_vals,
                    data,
                )

            # Compute FFT
            nu_dets, nu_cohs, data_freq = compute_2d_fft_wavenumber(
                extended_t_dets, extended_t_cohs, extended_data
            )

            # Plot each spectral component separately
            for component in spectral_components_to_plot:
                try:
                    fig = plot_2d_el_field(
                        axis_det=nu_dets,
                        axis_coh=nu_cohs,
                        data=data_freq,
                        t_wait=t_wait,
                        domain="freq",
                        use_custom_colormap=True,
                        component=component,
                        section=section,
                    )
                    filename = generate_unique_plot_filename(
                        system=system,
                        info_config=info_config,
                        domain="freq",
                        component=component,
                    )

                    save_fig(fig, filename=filename)
                except Exception as e:
                    print(f"❌ Error plotting 2D {component} component: {e}")

            print("✅ 2D frequency domain plots completed!")

        except Exception as e:
            print(f"❌ Error in 2D frequency domain processing: {e}")

    # Clean up memory
    plt.close("all")
    gc.collect()
