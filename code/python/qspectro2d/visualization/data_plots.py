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
from config.mpl_tex_settings import save_fig


# =============================
# HELPER FUNCTIONS FOR PLOTTING
# =============================
def _validate_and_extract_data_structure(
    loaded_data: dict, expected_dim: str = "auto"
) -> dict:
    """
    Validate and extract standardized data structure for plotting functions.

    Args:
        loaded_data: Dictionary from load_data_from_paths
        expected_dim: Expected dimensionality ("1d", "2d", or "auto")

    Returns:
        dict: Validated and extracted data components

    Raises:
        ValueError: If data structure is invalid or missing required components
    """
    required_keys = ["data", "axes", "system", "data_config"]
    missing_keys = [key for key in required_keys if key not in loaded_data]

    if missing_keys:
        raise ValueError(f"Missing required keys in loaded_data: {missing_keys}")

    # Extract components
    data = loaded_data["data"]
    axes = loaded_data["axes"]
    system = loaded_data["system"]
    data_config = loaded_data["data_config"]

    # Validate axes structure
    if "axs1" not in axes:
        raise ValueError("Missing 'axs1' in axes data")

    # Determine dimensionality
    has_axs2 = "axs2" in axes
    actual_dim = "2d" if has_axs2 else "1d"

    if expected_dim != "auto" and expected_dim != actual_dim:
        raise ValueError(f"Expected {expected_dim} data but got {actual_dim} data")

    result = {
        "data": data,
        "axs1": axes["axs1"],
        "system": system,
        "data_config": data_config,
        "dimension": actual_dim,
    }

    if has_axs2:
        result["axs2"] = axes["axs2"]

    return result


# =============================
# PLOTTING FUNCTIONS
# =============================
def plot_1d_data(
    loaded_data: dict,
    plot_config: dict,
    output_dir: Path,
) -> None:
    """Plot 1D spectroscopy data using standardized data structure.

    Args:
        loaded_data: Dictionary containing data, axes, system, and config from load_data_from_paths
        plot_config: Plotting configuration
        output_dir: Directory to save plots
    """
    # Validate and extract data structure
    extracted = _validate_and_extract_data_structure(loaded_data, expected_dim="1d")
    data = extracted["data"]
    t_det_vals = extracted["axs1"]
    system = extracted["system"]
    data_config = extracted["data_config"]
    tau_coh = data_config["tau_coh"]
    T_wait = data_config["T_wait"]
    n_freqs = data_config.get("n_freqs", 1)

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
                tau_coh=tau_coh,
                T_wait=T_wait,
                n_freqs=n_freqs,
                function_symbol="E_{k_s}",
            )
            filename = generate_unique_plot_filename(
                system,
                data_config=data_config,
                domain="time",
                component="abs",
            )

            save_fig(fig, filename=filename)
            plt.close(fig)
            print("✅ Time domain plots completed!")
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
                    data_config=data_config,
                    domain="freq",
                    component=component,
                )
                save_fig(fig, filename=filename)
                plt.close(fig)
            except Exception as e:
                print(f"❌ Error plotting {component} component: {e}")

        print("✅ Frequency domain plots completed!")

    # Clean up memory
    plt.close("all")
    gc.collect()


def plot_2d_data(
    loaded_data: dict,
    plot_config: dict,
    output_dir: Path,
) -> None:
    """Plot 2D spectroscopy data using standardized data structure.

    Args:
        loaded_data: Dictionary containing data, axes, system, and config from load_data_from_paths
        plot_config: Plotting configuration
        output_dir: Directory to save plots
    """
    # Validate and extract data structure
    extracted = _validate_and_extract_data_structure(loaded_data, expected_dim="2d")
    data = extracted["data"]
    tau_coh_vals = extracted["axs1"]  # coherence times
    t_det_vals = extracted["axs2"]  # detection times
    system = extracted["system"]
    data_config = extracted["data_config"]
    T_wait = data_config["T_wait"]

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
        try:
            fig = plot_2d_el_field(
                data_x=t_det_vals,
                data_y=tau_coh_vals,
                data_z=data,
                t_wait=T_wait,
                domain="time",
                use_custom_colormap=True,
            )
            filename = generate_unique_plot_filename(
                system=system, data_config=data_config, domain="time"
            )

            save_fig(fig, filename=filename)
            plt.close(fig)
            print("✅ 2D time domain plots completed!")
        except Exception as e:
            print(f"❌ Error in 2D time domain plotting: {e}")

    ### Handle frequency domain processing
    if plot_config.get("plot_frequency_domain", True):
        print("📊 Plotting 2D frequency domain data...")
        try:
            # Extend time axes if needed
            if extend_for != (1, 1):
                extended_ts, extended_taus, extended_data = extend_time_axes(
                    data=data,
                    t_det=t_det_vals,
                    tau_coh=tau_coh_vals,
                    pad_t_det=extend_for,
                    pad_tau_coh=extend_for,
                )
            else:
                extended_ts, extended_taus, extended_data = (
                    t_det_vals,
                    tau_coh_vals,
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
                        data_config=data_config,
                        domain="freq",
                        component=component,
                    )

                    save_fig(fig, filename=filename)
                    plt.close(fig)
                except Exception as e:
                    print(f"❌ Error plotting 2D {component} component: {e}")

            print("✅ 2D frequency domain plots completed!")

        except Exception as e:
            print(f"❌ Error in 2D frequency domain processing: {e}")

    # Clean up memory
    plt.close("all")
    gc.collect()
