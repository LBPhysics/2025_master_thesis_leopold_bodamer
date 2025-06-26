"""
1D Electronic Spectroscopy Data Plotting Script

This script loads and plots 1D electronic spectroscopy data from pickle files
in various formats (real, imaginary, absolute, phase) for analysis and visualization.

Usage modes:
1. Direct relative path: python plot_1D_datas.py 1d_spectroscopy/N2_atoms/mesolve/special_dir
2. No arguments: Use search configuration (looking in standard directories)
"""

import sys
from pathlib import Path
from common_fcts import load_latest_data, _plot_1d_data
from config.paths import DATA_DIR, FIGURES_1D_DIR


# =============================
# MAIN FUNCTION
# =============================
def main():
    """Main function to run the 1D spectroscopy plotting."""

    # Check if relative directory path was provided as command line argument
    if len(sys.argv) > 1:
        # Mode 1: Plot from specific relative directory
        relative_dir_str = sys.argv[1]
        relative_dir = Path(relative_dir_str)

        print(
            f"🚀 Starting 1D Electronic Spectroscopy Plotting from relative path: {relative_dir}"
        )

        # Plot with the new workflow
        plot_from_relative_dir(relative_dir)
    else:
        # Mode 2: Plot using search configuration
        plot_with_search_config()


def plot_from_relative_dir(relative_dir: Path):
    """
    Plot 1D data from a specific relative directory using the new workflow.

    Args:
        relative_dir: Relative directory path where the data is stored
    """
    print(
        f"🔍 Looking for latest data in: {DATA_DIR / '1d_spectroscopy' / relative_dir}"
    )

    # Plotting configuration
    config = {
        "spectral_components_to_plot": ["real", "imag", "abs", "phase"],
        "plot_time_domain": True,
        "extend_for": (1, 3),
        "section": (1.4, 1.8, 1.4, 1.8),
    }

    try:
        # Load the latest data file from the relative directory
        data = load_latest_data(relative_dir)

        # Extract system and plotting data
        system = data.get("system")
        if system is None:
            print("❌ Error: Loaded data does not contain system information")
            return

        # Create figure directory if it doesn't exist
        figure_dir = FIGURES_1D_DIR / relative_dir
        figure_dir.mkdir(parents=True, exist_ok=True)

        # Plot the data using the _plot_1d_data function directly
        _plot_1d_data(data, config, figure_dir)

        print(f"✅ Figures saved to: {figure_dir}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def find_latest_data_recursive(base_dir: Path, max_depth: int = 3) -> tuple:
    """
    Recursively search for data files in a directory and its subdirectories.

    Args:
        base_dir: The base directory to start searching
        max_depth: Maximum directory depth to search

    Returns:
        tuple: (found data dictionary, relative directory where found)
              or (None, None) if no data found
    """
    import os

    # First try base directory
    try:
        print(f"🔍 Trying directory: {base_dir}")
        data = load_latest_data("1d_spectroscopy" / base_dir)
        return data, base_dir
    except FileNotFoundError:
        print(f"   No data files found in {base_dir}, searching subdirectories...")

    # If not found, and we haven't reached max depth, search subdirectories
    if max_depth > 0:
        # Get all subdirectories in the data directory
        full_base_dir = DATA_DIR / "1d_spectroscopy" / base_dir

        try:
            # List all subdirectories
            subdirs = [
                d
                for d in os.listdir(full_base_dir)
                if os.path.isdir(os.path.join(full_base_dir, d))
            ]

            # Sort subdirectories by modification time (newest first)
            subdirs.sort(
                key=lambda d: os.path.getmtime(os.path.join(full_base_dir, d)),
                reverse=True,
            )

            # Check each subdirectory
            for subdir in subdirs:
                next_dir = base_dir / Path(subdir)
                try:
                    data, found_dir = find_latest_data_recursive(
                        next_dir, max_depth - 1
                    )
                    if data:
                        return data, found_dir
                except Exception as e:
                    print(f"   Skipping {next_dir}: {e}")
        except FileNotFoundError:
            print(f"   Base directory {full_base_dir} does not exist")
        except Exception as e:
            print(f"   Error searching subdirectories: {e}")

    # If we get here, no data was found
    return None, None


def plot_with_search_config():
    """Plot 1D data using search configuration (looking in standard directories)."""

    # =============================
    # PLOTTING PARAMETERS - MODIFY HERE
    # =============================

    # Base directory to search
    base_dir = Path("")  # Empty string means start at the root of 1d_spectroscopy

    print(f"🔍 Starting search in 1d_spectroscopy directory and subdirectories")

    # Configuration for plotting
    config = {
        "spectral_components_to_plot": ["real", "imag", "abs", "phase"],
        "plot_time_domain": True,
        "extend_for": (1, 3),
        "section": (1.4, 1.8, 1.4, 1.8),
    }

    # Try to find data recursively
    data, found_dir = find_latest_data_recursive(base_dir)

    if data:
        print(f"✅ Found data in directory: {found_dir}")

        # Extract system information
        system = data.get("system")
        if system is None:
            print("❌ Error: Loaded data does not contain system information")
            return

        # Create figure directory
        figure_dir = FIGURES_1D_DIR / found_dir
        figure_dir.mkdir(parents=True, exist_ok=True)

        # Plot the data
        _plot_1d_data(data, config, figure_dir)

        print(f"✅ Figures saved to: {figure_dir}")
    else:
        print(
            "❌ No suitable data files found in 1d_spectroscopy directory or subdirectories"
        )


if __name__ == "__main__":
    main()
