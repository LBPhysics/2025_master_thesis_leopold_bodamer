"""
2D Electronic Spectroscopy Data Plotting Script - Simplified Version

This script loads and plots 2D electronic spectroscopy data from pickle files
in various formats (real, imaginary, absolute, phase). All parameters are defined directly in main().
"""

# =============================
# IMPORTS
# =============================
from common_fcts import plot_2d_spectroscopy_data


# =============================
# MAIN FUNCTION
# =============================
def main():
    """Main function to run the 2D spectroscopy plotting."""

    # =============================
    # PLOTTING PARAMETERS - MODIFY HERE
    # =============================

    ### Data source configuration
    data_subdir = "2d_spectroscopy/N_1/paper_eqs/100fs"  # Data subdirectory to search
    file_pattern = "*.pkl"  # File pattern to match

    ### Output configuration
    output_subdir = "2d_spectroscopy"  # Output figures subdirectory

    ### Plot type selection
    plot_types = ["imag", "abs", "real", "phase"]  # Which plot types to generate
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
        "plot_types": plot_types,
        "plot_time_domain": plot_time_domain,
        "extend_for": extend_for,
        "section": section,
    }

    # =============================
    # PRINT CONFIGURATION SUMMARY
    # =============================
    print(f"🚀 Starting 2D Electronic Spectroscopy Plotting...")
    print(f"  Data source: {data_subdir}")
    print(f"  Plot types: {plot_types}")
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
