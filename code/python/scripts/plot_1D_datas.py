"""
1D Electronic Spectroscopy Data Plotting Script

This script loads and plots 1D electronic spectroscopy data from pickle files
in both time domain and frequency domain for analysis and visualization.
"""

# =============================
# IMPORTS
# =============================
import numpy as np
import os
import sys
import pickle
import gc  # For garbage collection
import psutil  # For memory monitoring
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

### Project-specific imports
from config.paths import DATA_DIR, FIGURES_DIR
from qspectro2d.visualization.plotting import (
    Plot_fixed_tau_T,
    Plot_1d_frequency_spectrum,
)
from qspectro2d.core.system_parameters import SystemParameters
from qspectro2d.spectroscopy.post_processing import compute_1d_fft_wavenumber


# =============================
# UTILITY FUNCTIONS
# =============================
def find_latest_file(data_subdir: str, file_pattern: str = "*.pkl") -> Optional[Path]:
    """Find the most recent file matching pattern in a data subdirectory.

    Args:
        data_subdir: Subdirectory within DATA_DIR (e.g., '1d_spectroscopy/inhomogeneity')
        file_pattern: Glob pattern for file matching

    Returns:
        Path to the latest file or None if not found
    """
    data_dir = DATA_DIR / data_subdir

    if not data_dir.exists():
        print(f"❌ Data directory does not exist: {data_dir}")
        return None

    # Look for files matching the pattern
    files = list(data_dir.glob(file_pattern))

    if not files:
        print(f"❌ No .pkl files found in: {data_dir}")
        return None

    # Sort by modification time and return the latest
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"📁 Found latest file: {latest_file.name}")
    return latest_file


def load_pickle_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load data from a pickle file with error handling.

    Args:
        file_path: Path to the pickle file

    Returns:
        Dictionary containing the loaded data or None if loading fails
    """
    try:
        print(f"📥 Loading data from: {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Data loaded successfully!")
        return data
    except Exception as e:
        print(f"❌ ERROR: Failed to load pickle file: {e}")
        return None


def setup_figure_directory(subdir: str) -> Path:
    """Create output directory for figures.

    Args:
        subdir: Subdirectory under FIGURES_DIR

    Returns:
        Path to the created output directory
    """
    output_dir = FIGURES_DIR / subdir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =============================
# MAIN PLOTTING FUNCTIONS
# =============================
def plot_1d_time_domain(
    t_det_vals: np.ndarray,
    data_avg: np.ndarray,
    config: Dict[str, Any],
    system: SystemParameters,
    output_dir: Path,
    plot_types: List[str] = None,
) -> None:
    """Plot 1D time-domain spectroscopy data.

    Args:
        t_det_vals: Time detection values
        data_avg: Averaged polarization data
        config: Configuration dictionary
        system: System parameters
        output_dir: Output directory for figures
        plot_types: List of plot types to generate
    """
    if plot_types is None:
        plot_types = ["real", "imag", "abs", "phase"]

    print("📊 Plotting 1D time-domain data...")

    ### Create comprehensive plot with all components

    ### Use Plot_fixed_tau_T to create the time domain plot
    Plot_fixed_tau_T(
        t_det_vals=t_det_vals,
        data=data_avg,
        tau_coh=config["tau_coh"],
        T_wait=config["T_wait"],
        n_phases=config["n_phases"],
        n_freqs=config["n_freqs"],
        show=True,  # Don't show immediately, we'll save it
    )

    ### Save the time domain plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"1d_time_domain_combined_"
        f"tau{config['tau_coh']:.0f}_T{config['T_wait']:.0f}_"
        f"ph{config['n_phases']}_freq{config['n_freqs']}_{timestamp}.png"
    )
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"💾 Saved combined time-domain plot: {save_path}")
    plt.show()


def plot_1d_frequency_domain(
    t_det_vals: np.ndarray,
    data_avg: np.ndarray,
    config: Dict[str, Any],
    system: SystemParameters,
    output_dir: Path,
    plot_types: List[str] = None,
) -> None:
    """Plot 1D frequency-domain spectroscopy data.

    Args:
        t_det_vals: Time detection values
        data_avg: Averaged polarization data
        config: Configuration dictionary
        system: System parameters
        output_dir: Output directory for figures
        plot_types: List of plot types to generate
    """
    if plot_types is None:
        plot_types = ["abs", "real", "imag", "phase"]

    print("📊 Computing and plotting 1D frequency-domain data...")

    ### Compute Fourier transform
    nu_vals, spectrum_data = compute_1d_fft_wavenumber(t_det_vals, data_avg)

    ### Plot Spectrum for each type
    for plot_type in plot_types:
        Plot_1d_frequency_spectrum(
            nu_vals=nu_vals,
            spectrum_data=spectrum_data,
            type=plot_type,
            title=f"1D Frequency Spectrum ({plot_type.capitalize()})",
            output_dir=str(output_dir),
            save=True,
            system=system,
            tau_coh=config["tau_coh"],
            T_wait=config["T_wait"],
            n_phases=config["n_phases"],
            n_freqs=config["n_freqs"],
        )

    print(f"📊 Completed frequency-domain plotting for all types: {plot_types}")


# =============================
# MAIN PLOTTING FUNCTION
# =============================
def plot_1d_spectroscopy_data(
    data_subdir: str = "1d_spectroscopy/inhomogeneity",
    file_pattern: str = "*.pkl",
    output_subdir: str = "1d_spectroscopy",
    plot_types: List[str] = None,
    plot_time_domain: bool = True,
    plot_frequency_domain: bool = True,
) -> None:
    """Load and plot 1D spectroscopy data.

    Args:
        data_subdir: Subdirectory containing 1D data files
        file_pattern: Pattern to match pickle files
        output_subdir: Subdirectory for output figures
        plot_types: List of plot types to generate
        plot_time_domain: Whether to plot time domain data
        plot_frequency_domain: Whether to plot frequency domain data
    """
    if plot_types is None:
        plot_types = ["real", "imag", "abs", "phase"]

    print("# =============================")
    print("# LOAD AND PLOT 1D SPECTROSCOPY DATA")
    print("# =============================")

    ### Find and load 1D data
    file_path_1d = find_latest_file(data_subdir, file_pattern)

    if file_path_1d is None:
        print("   Please run the 1D spectroscopy calculation first to generate data.")
        return

    loaded_data = load_pickle_file(file_path_1d)

    if loaded_data is None:
        print("❌ Failed to load 1D spectroscopy data.")
        return

    ### Extract data
    try:
        t_det_vals = loaded_data["t_det_vals"]
        data_avg = loaded_data["data_avg"]
        system = loaded_data["system"]

        config = {
            "tau_coh": loaded_data.get("tau_coh", 0),
            "T_wait": loaded_data.get("T_wait", 0),
            "n_phases": loaded_data.get("n_phases", 1),
            "n_freqs": loaded_data.get("n_freqs", 1),
        }

        print(f"📊 Data summary:")
        print(f"   Time points: {len(t_det_vals)}")
        print(f"   Data shape: {data_avg.shape}")
        print(f"   Time range: {t_det_vals[0]:.1f} to {t_det_vals[-1]:.1f} fs")
        print(f"   Configuration: {config}")

    except KeyError as e:
        print(f"❌ ERROR: Missing key in data file: {e}")
        return

    ### Create output directory
    output_dir = setup_figure_directory(output_subdir)
    print(f"📁 Output directory: {output_dir}")

    ### Plot time domain data
    if plot_time_domain:
        plot_1d_time_domain(
            t_det_vals, data_avg, config, system, output_dir, plot_types
        )

    ### Plot frequency domain data
    if plot_frequency_domain:
        plot_1d_frequency_domain(
            t_det_vals, data_avg, config, system, output_dir, plot_types
        )

    print("\n" + "=" * 60)
    print("1D SPECTROSCOPY PLOTTING COMPLETED")
    print("=" * 60)
    print(f"All plots saved to: {output_dir}")


# =============================
# MAIN EXECUTION
# =============================
def main():
    """Main function for running the 1D plotting script."""

    # Configuration options
    config_options = {
        "data_subdir": "1d_spectroscopy/inhomogeneity",  # Change as needed
        "file_pattern": "*.pkl",
        "output_subdir": "1d_spectroscopy",
        "plot_types": ["real", "imag", "abs", "phase"],
        "plot_time_domain": True,
        "plot_frequency_domain": True,
    }

    print("🚀 Starting 1D Electronic Spectroscopy Plotting...")
    print(f"Configuration: {config_options}")

    ### Run the plotting
    plot_1d_spectroscopy_data(**config_options)


if __name__ == "__main__":
    main()
