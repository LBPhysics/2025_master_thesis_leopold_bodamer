"""
2D Electronic Spectroscopy Data Plotting Script

This script loads and plots 2D electronic spectroscopy data from pickle files
in various formats (real, imaginary, absolute, phase) for analysis and visualization.
"""

import sys
from pathlib import Path
from common_fcts import plot_2d_spectroscopy_data, plot_2d_from_filepath
from config.paths import DATA_DIR


# =============================
# MAIN FUNCTION
# =============================
def main():
    """Main function to run the 2D spectroscopy plotting."""

    # Check if filepath was provided as command line argument
    if len(sys.argv) > 1:
        # Mode 1: Plot from specific filepath
        filepath = Path(sys.argv[1])
        plot_from_filepath(filepath)
    else:
        # Mode 2: Plot using search configuration (original behavior)
        plot_with_search_config()


def plot_from_filepath(filepath: Path):
    """Plot 2D data from a specific filepath or directory."""
    print(f"🚀 Starting 2D Electronic Spectroscopy Plotting from: {filepath}")

    ### Plotting configuration
    config = {
        "spectral_components_to_plot": ["real", "imag", "abs", "phase"],
        "plot_time_domain": True,
        "extend_for": (1, 3),
        "section": (1.4, 1.8, 1.4, 1.8),
        "output_subdir": "",
    }

    # Resolve the full path using DATA_DIR
    full_path = DATA_DIR / filepath

    print(f"Looking for data in: {full_path}")

    # Check if it's a directory or file
    if full_path.is_dir():
        # Find all .pkl files in the directory
        pkl_files = list(full_path.glob("*.pkl"))
        if not pkl_files:
            print(f"❌ No .pkl files found in directory: {full_path}")
            return

        print(f"Found {len(pkl_files)} .pkl files:")
        for pkl_file in pkl_files:
            print(f"  - {pkl_file.name}")

        # Plot all files found
        for pkl_file in pkl_files:
            print(f"\n📊 Plotting: {pkl_file.name}")
            plot_2d_from_filepath(pkl_file, config)

    elif full_path.is_file():
        # Direct file path provided
        plot_2d_from_filepath(full_path, config)

    else:
        print(f"❌ Path does not exist: {full_path}")
        return

    print("✅ 2D Spectroscopy plotting completed!")


def plot_with_search_config():
    """Plot 2D data using search configuration (original behavior)."""

    # =============================
    # PLOTTING PARAMETERS - MODIFY HERE
    # =============================

    ### Data source configuration
    data_subdir = (
        "2d_spectroscopy/N_1/paper_eqs/t_max_100fs"  # Data subdirectory to search
    )
    file_pattern = "*.pkl"  # File pattern to match

    ### Output configuration
    output_subdir = "2d_spectroscopy"  # Output figures subdirectory

    ### Plot types to generate
    spectral_components_to_plot = [
        "real",
        "imag",
        "abs",
        "phase",
    ]  # Available: real, imag, abs, phase

    ### Additional plotting options
    plot_time_domain = True  # Plot time domain data
    extend_for = (1, 3)  # Extend frequency range for plotting
    section = (
        1.4,
        1.8,
        1.4,
        1.8,
    )  # Section to focus on (freq_min, freq_max, freq_min, freq_max)

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
    print(f"🚀 Starting 2D Electronic Spectroscopy Plotting...")
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
    plot_2d_spectroscopy_data(config)

    print("✅ 2D Spectroscopy plotting completed!")


if __name__ == "__main__":
    main()
