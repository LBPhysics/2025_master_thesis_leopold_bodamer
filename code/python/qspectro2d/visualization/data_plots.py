"""
Data plotting functions for qspectro2d.

This module provides high-level plotting functions for 1D and 2D
spectroscopy data with standardized formatting and output.
"""

# =============================
# IMPORTS
# =============================
import numpy as np
import matplotlib.pyplot as plt
import gc
from pathlib import Path
from typing import Dict

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
from qspectro2d.data.files import generate_unique_plot_filename
from qspectro2d.config.mpl_tex_settings import save_fig


# =============================
# HELPER FUNCTIONS FOR PLOTTING
# =============================
def _validate_and_extract_data_structure(
    loaded_data: dict, expected_dim: str = "auto"
) -> dict:
    """
    Validate and extract standardized data structure for plotting functions.

    Args:
        loaded_data: Dictionary from load_data_from_rel_path
        expected_dim: Expected dimensionality ("1d", "2d", or "auto")

    Returns:
        dict: Validated and extracted data components

    Raises:
        ValueError: If data structure is invalid or missing required components
    """
    required_keys = ["data", "axes", "system", "info_config"]
    missing_keys = [key for key in required_keys if key not in loaded_data]

    if missing_keys:
        raise ValueError(f"Missing required keys in loaded_data: {missing_keys}")

    # Extract components
    data = loaded_data["data"]
    axes = loaded_data["axes"]
    system = loaded_data["system"]
    info_config = loaded_data["info_config"]

    # Validate axes structure
    if "axis1" not in axes:
        raise ValueError("Missing 'axis1' in axes data")

    # Determine dimensionality
    has_axis2 = "axis2" in axes
    actual_dim = "2d" if has_axis2 else "1d"

    if expected_dim != "auto" and expected_dim != actual_dim:
        raise ValueError(f"Expected {expected_dim} data but got {actual_dim} data")

    result = {
        "data": data,
        "axis1": axes["axis1"],
        "system": system,
        "info_config": info_config,
        "dimension": actual_dim,
    }

    if has_axis2:
        result["axis2"] = axes["axis2"]

    return result


# =============================
# PLOTTING FUNCTIONS
# =============================
def plot_1d_data(
    loaded_data: dict,
    plot_config: dict,
) -> None:
    """Plot 1D spectroscopy data using standardized data structure.

    Args:
        loaded_data: Dictionary containing data, axes, system, and config from load_data_from_rel_path
        plot_config: Plotting configuration
    """
    # Validate and extract data structure
    extracted = _validate_and_extract_data_structure(loaded_data, expected_dim="1d")
    data = extracted["data"]
    t_det_vals = extracted["axis1"]
    system = extracted["system"]
    info_config = extracted["info_config"]
    t_coh = info_config["t_coh"]
    T_wait = info_config["t_wait"]
    n_freqs = info_config.get("n_freqs", 1)

    print(f"✅ Data loaded with shape: {data.shape}")
    print(f"   Time points: {len(t_det_vals)}")
    print(f"   Time range: {t_det_vals[0]:.1f} to {t_det_vals[-1]:.1f} fs")

    spectral_components_to_plot = plot_config.get(
        "spectral_components_to_plot", ["abs"]
    )
    extend_for = plot_config.get("extend_for", (1, 1))

    ### Plot time domain data
    if plot_config.get("plot_time_domain", True):
        print("📊 Plotting time domain data...")
        try:
            fig = plot_1d_el_field(
                data_x=t_det_vals,
                data_y=data,
                domain="time",
                component="abs",  # Use first component for time domain
                t_coh=t_coh,
                T_wait=T_wait,
                function_symbol="E_{k_s}",
                n_freqs=n_freqs,
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
                    data_x=frequencies,
                    data_y=data_fft,
                    domain="freq",
                    component=component,
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
    loaded_data: dict,
    plot_config: dict,
) -> None:
    """Plot 2D spectroscopy data using standardized data structure.

    Args:
        loaded_data: Dictionary containing data, axes, system, and config from load_data_from_rel_path
        plot_config: Plotting configuration
    """
    # Validate and extract data structure
    extracted = _validate_and_extract_data_structure(loaded_data, expected_dim="2d")
    data = extracted["data"]
    t_coh_vals = extracted["axis1"]  # coherence times
    t_det_vals = extracted["axis2"]  # detection times
    system = extracted["system"]
    info_config = extracted["info_config"]
    T_wait = info_config["t_wait"]

    # Get configuration values
    spectral_components_to_plot = plot_config.get(
        "spectral_components_to_plot", ["abs"]
    )
    extend_for = plot_config.get("extend_for", (1, 1))
    section = plot_config.get("section", (0, 2, 0, 2))

    print(f"✅ 2D data loaded successfully!")

    ### Plot time domain data
    if plot_config.get("plot_time_domain", True):
        print("📊 Plotting 2D time domain data...")
        # Plot each spectral component separately
        time_domain_comps = ["real", "imag", "abs", "phase"]
        for component in time_domain_comps:
            try:
                fig = plot_2d_el_field(
                    data_x=t_det_vals,
                    data_y=t_coh_vals,
                    data_z=data,
                    t_wait=T_wait,
                    domain="time",
                    use_custom_colormap=True,
                    component=component,
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
                extended_ts, extended_taus, extended_data = extend_time_axes(
                    data=data,
                    t_det=t_det_vals,
                    t_coh=t_coh_vals,
                    pad_t_det=extend_for,
                    pad_t_coh=extend_for,
                )
            else:
                extended_ts, extended_taus, extended_data = (
                    t_det_vals,
                    t_coh_vals,
                    data,
                )

            # Compute FFT
            nu_ts, nu_taus, data_freq = compute_2d_fft_wavenumber(
                extended_ts, extended_taus, extended_data
            )

            # Plot each spectral component separately
            for component in spectral_components_to_plot:
                try:
                    fig = plot_2d_el_field(
                        data_x=nu_ts,
                        data_y=nu_taus,
                        data_z=data_freq,
                        t_wait=T_wait,
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
