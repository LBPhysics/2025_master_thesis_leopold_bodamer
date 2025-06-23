"""
2D Electronic Spectroscopy Data Plotting Script

This script loads and plots 2D electronic spectroscopy data from pickle files
in various formats (real, imaginary, absolute, phase) for analysis and visualization.
"""

# =============================
# IMPORTS
# =============================
import numpy as np
import os
import sys
import gzip
import pickle
import gc  # For garbage collection
import psutil  # For memory monitoring
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional

### Project-specific imports
from config.paths import DATA_DIR, FIGURES_DIR
from qspectro2d.spectroscopy.post_processing import extend_and_plot_results
from qspectro2d.spectroscopy.calculations import get_tau_cohs_and_t_dets_for_T_wait
from qspectro2d.visualization.plotting import Plot_2d_El_field

# from qspectro2d.visualization import mpl_tex_settings # -> TODO I think i want to stick to default latex font, cause my settings currently only work on my pc!


# =============================
# UTILITY FUNCTIONS
# =============================
def find_latest_file(data_subdir: str, file_pattern: str = "*.pkl") -> Optional[Path]:
    """Find the most recent file matching pattern in a data subdirectory.

    Args:
        data_subdir: Subdirectory within DATA_DIR (e.g., '2d_spectroscopy/new_echo_signal/600fs')
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

    # For 2D data, also look for compressed files
    if "2d_spectroscopy" in data_subdir:
        files.extend(list(data_dir.glob("*.pkl.gz")))

    if not files:
        print(f"❌ No files matching '{file_pattern}' found in {data_dir}")
        return None

    # Get the most recent file
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    print(f"✅ Found latest file: {latest_file.name}")
    return latest_file


def load_pickle_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load data from pickle file (supports both .pkl and .pkl.gz).

    Args:
        filepath: Path to the pickle file

    Returns:
        Dictionary containing the loaded data or None if error
    """
    print(f"Loading data from: {filepath.name}")

    try:
        if filepath.suffix == ".gz":
            # Handle compressed pickle files
            with gzip.open(filepath, "rb") as f:
                data = pickle.load(f)
        else:
            # Handle regular pickle files
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        print(f"✅ Data loaded successfully!")
        return data

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def create_output_directory(subdir: str) -> Path:
    """Create output directory for figures.

    Args:
        subdir: Subdirectory name within FIGURES_DIR

    Returns:
        Path to the created output directory
    """
    output_dir = FIGURES_DIR / subdir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =============================
# MAIN PLOTTING FUNCTION
# =============================
def plot_2d_spectroscopy_data(
    data_subdir: str = "2d_spectroscopy",
    file_pattern: str = "*.pkl",
    output_subdir: str = "2d_spectroscopy",
    plot_types: list = None,
    extend_for: tuple = (1, 2.3),
    section: tuple = (1.5, 1.7, 1.5, 1.7),
    plot_time_domain: bool = False,
) -> None:
    """Load and plot 2D spectroscopy data.

    Args:
        data_subdir: Subdirectory containing 2D data files
        file_pattern: Pattern to match pickle files
        output_subdir: Subdirectory for output figures
        plot_types: List of plot types to generate
        extend_for: Frequency extension range
        section: Plot section (x_min, x_max, y_min, y_max)
        plot_time_domain: Whether to plot time domain data
    """
    if plot_types is None:
        plot_types = ["imag", "abs", "real", "phase"]

    print("# =============================")
    print("# LOAD AND PLOT 2D SPECTROSCOPY DATA")
    print("# =============================")

    ### Find and load 2D data
    file_path_2d = find_latest_file(data_subdir, file_pattern)

    if file_path_2d is None:
        print("   Please run the 2D spectroscopy calculation first to generate data.")
        return

    loaded_data = load_pickle_file(file_path_2d)

    if loaded_data is None:
        print("❌ Failed to load 2D spectroscopy data.")
        return

    # Extract 2D data
    two_d_datas = loaded_data["two_d_datas"]
    times_T = loaded_data["times_T"]
    times = loaded_data["times"]
    system_data = loaded_data["system"]

    ### Display data information and memory usage
    print(f"  Loaded {len(two_d_datas)} 2D datasets")
    print(f"  System data: {system_data}")

    # Check memory usage
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
    print(f"  Memory usage after loading: {memory_usage:.2f} GB")

    ### Create output directory
    output_dir = create_output_directory(output_subdir)

    # Get time axes for time domain plotting
    t_det_vals, tau_coh_vals = get_tau_cohs_and_t_dets_for_T_wait(times, times_T[0])

    ### Plot time domain if requested
    if plot_time_domain:
        print("📊 Plotting time domain...")
        try:
            Plot_2d_El_field(
                (t_det_vals, tau_coh_vals, two_d_datas[0]),
                save=True,
                output_dir=output_dir,
                system=system_data,
                use_custom_colormap=True,
            )
            print("✅ Time domain plot completed!")
        except Exception as e:
            print(
                f"❌ Error plotting time domain: {e}"
            )  ### Plot each frequency component type
    print(f"📊 Plotting frequency domain components: {', '.join(plot_types)}")

    for i, plot_type in enumerate(plot_types):
        print(f"📊 Plotting {plot_type} component ({i+1}/{len(plot_types)})...")

        plot_args = {
            "domain": "freq",
            "type": plot_type,
            "save": True,
            "output_dir": output_dir,
            "use_custom_colormap": True,
            "section": section,  # Plot the specified section
            "system": system_data,
        }

        try:
            extend_and_plot_results(
                two_d_datas,
                times_T=times_T,
                times=times,
                extend_for=extend_for,
                **plot_args,
            )
            print(f"✅ {plot_type} plot completed!")

            # Force garbage collection after each plot to free memory
            plt.close("all")  # Close all matplotlib figures
            gc.collect()  # Force garbage collection

            # Check memory usage after each plot
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
            print(f"  Memory usage after {plot_type}: {memory_usage:.2f} GB")

        except Exception as e:
            print(f"❌ Error plotting {plot_type}: {e}")
            plt.close("all")  # Close figures even on error
            gc.collect()

    print(f"🎯 All 2D plots saved to: {output_dir}")


# =============================
# CONFIGURATION FUNCTION
# =============================
def get_plotting_config():
    """Get default plotting configuration.

    Returns:
        Dictionary with default plotting parameters
    """
    return {
        "data_subdir": "2d_spectroscopy",
        "file_pattern": "*.pkl",
        "output_subdir": "2d_spectroscopy",
        "plot_types": ["imag", "abs", "real", "phase"],
        "extend_for": (1, 1),
        "section": (0, 3, 0, 3),  # (x_min, x_max, y_min, y_max)
        "plot_time_domain": True,
    }


# =============================
# MAIN EXECUTION
# =============================
def main():
    """Main function to run the 2D spectroscopy plotting."""
    print("Starting 2D Spectroscopy Data Plotting...")

    # Get configuration
    config = get_plotting_config()

    # You can modify these parameters as needed
    config.update(
        {
            # Example modifications: TODO JUST CHANGE THIS PART
            "data_subdir": "2d_spectroscopy/N_1/paper_eqs/t_max_100fs",
            # "plot_types": ["real", "imag"],  # Plot only real and imaginary parts
            "plot_time_domain": True,  # Enable time domain plotting
            "extend_for": (1, 3),
            "section": (
                1.4,
                1.8,
                1.4,
                1.8,
            ),  # Different section   # (1.5, 1.7, 1.5, 1.7),
        }
    )

    # Run the plotting
    plot_2d_spectroscopy_data(**config)

    print("✅ 2D Spectroscopy plotting completed!")


if __name__ == "__main__":
    main()
