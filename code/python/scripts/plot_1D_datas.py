"""
1D Electronic Spectroscopy Data Plotting Script

This script loads and plots 1D electronic spectroscopy data from pickle files
in various formats (real, imaginary, absolute, phase). All parameters are defined directly in main().
"""

# =============================
# IMPORTS
# =============================
import sys
from pathlib import Path
from common_fcts import plot_1d_spectroscopy_data, plot_1d_from_filepath


# =============================
# MAIN FUNCTION
# =============================
def main():
    """Main function to run the 1D spectroscopy plotting."""

    # Check if filepath was provided as command line argument
    if len(sys.argv) > 1:
        # Mode 1: Plot from specific filepath
        filepath = Path(sys.argv[1])
        plot_from_filepath(filepath)
    else:
        # Mode 2: Plot using search configuration (original behavior)
        plot_with_search_config()


def plot_from_filepath(filepath: Path):
    """Plot 1D data from a specific filepath."""
    print(f"🚀 Starting 1D Electronic Spectroscopy Plotting from file: {filepath.name}")

    ### Plotting configuration
    config = {
        "spectral_components_to_plot": ["real", "imag", "abs", "phase"],
        "plot_time_domain": True,
        "extend_for": (1, 3),
        "section": (1.4, 1.8, 1.4, 1.8),
        "output_subdir": "",
    }

    # Plot from the specific file
    plot_1d_from_filepath(filepath, config)
    print("✅ 1D Spectroscopy plotting completed!")


def plot_with_search_config():
    """Plot 1D data using search configuration (original behavior)."""

    # =============================
    # PLOTTING PARAMETERS - MODIFY HERE
    # =============================

    ### Data source configuration
    data_subdir = "1d_spectroscopy/N_1/paper_eqs/100fs"  # Data subdirectory to search
    file_pattern = "*.pkl"  # File pattern to match

    ### Output configuration
    output_subdir = "1d_spectroscopy"  # Output figures subdirectory

    ### Plot component selection
    spectral_components_to_plot = [
        "imag",
        "abs",
        "real",
        "phase",
    ]  # Which plot types to generate
    plot_time_domain = True  # Plot time domain data

    ### Frequency domain plot settings
    extend_for = (1, 3)  # Frequency extension range
    section = (1.4, 1.8, 1.4, 1.8)  # Plot section (x_min, x_max, y_min, y_max)

    # =============================
    # BUILD CONFIGURATION DICTIONARY
    # =============================
    config = {
        "data_subdir": data_subdir,
        "file_pattern": file_pattern,
        "output_subdir": output_subdir,
        "spectral_components_to_plot": spectral_components_to_plot,
        "plot_time_domain": plot_time_domain,
        "extend_for": extend_for,
        "section": section,
    }

    # =============================
    # PRINT CONFIGURATION SUMMARY
    # =============================
    print(f"🚀 Starting 1D Electronic Spectroscopy Plotting...")
    print(f"  Data source: {data_subdir}")
    print(f"  Plot types: {spectral_components_to_plot}")
    print(f"  Time domain: {plot_time_domain}")
    print(f"  Extend for: {extend_for}")
    print(f"  Section: {section}")
    print(f"  Output: {output_subdir}")
    print("")

    # =============================
    # RUN PLOTTING
    # =============================
    plot_1d_spectroscopy_data(config)

    print("✅ 1D Spectroscopy plotting completed!")


if __name__ == "__main__":
    main()
