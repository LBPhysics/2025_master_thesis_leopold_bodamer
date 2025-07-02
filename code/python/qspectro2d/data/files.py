"""
File naming and path utilities for qspectro2d.

This module provides functionality for generating standardized filenames
and directory paths for simulation data and plots.
"""

# =============================
# IMPORTS
# =============================
from pathlib import Path
from tkinter import N
from typing import Union

### Project-specific imports
from qspectro2d.core.system_parameters import SystemParameters
from qspectro2d.config.paths import DATA_DIR, FIGURES_PYTHON_DIR  # , FIGURES_DIR,


# =============================
# FILENAME GENERATION FUNCTIONS
# =============================
def _generate_base_filename(system: SystemParameters, info_config: dict) -> str:
    """
    Generate a universal base filename for a calculation based on system and info_config parameters.

    Args:
        system: System parameters object
        info_config: Dictionary containing simulation parameters

    Returns:
        str: Base filename without path
    """
    N_atoms = system.N_atoms
    # simulation_type = info_config.get("simulation_type", "spectroscopy")
    # parts.append(simulation_type)
    # parts.append(f"N{N_atoms}")
    parts = []

    if info_config.get("simulation_type") == "1d":
        # Round tau_coh to 2 decimal places for filename clarity
        tau_val = round(float(info_config["tau_coh"]), 2)
        parts.append(f"tau_{tau_val}")

    parts.append(f"wA{system.omega_A_cm/1e4:.2f}e4")
    parts.append(f"muA{system.mu_A:.2f}")
    if N_atoms == 2:
        parts.append(f"wB{system.omega_B_cm/1e4:.2f}e4")
        parts.append(f"muB{system.mu_B:.2f}")
        if system.J_cm > 0:
            parts.append(f"J{system.J_cm/1e3:.2f}e3")

    n_freqs = info_config.get("n_freqs", 1)

    if n_freqs > 1:
        parts.append(f"Delta{system.Delta_cm/1e4:.2f}e4")
    parts.append("cm-1")
    return "_".join(parts)


def _generate_unique_filename(path: Union[str, Path], base_name: str) -> str:
    """
    Generate a unique base filename in the specified directory, regardless of extension.

    Args:
        path (str or Path): Directory where the file will be saved
        base_name (str): Base name for the file (without extension)

    Returns:
        str: Unique base filename (without extension, but with the whole path)
    """
    path = Path(path)
    candidate = path / base_name
    counter = 1

    while (path / candidate.name).exists() or list(path.glob(f"{candidate.name}.*")):
        candidate = path / f"{base_name}_{counter}"
        counter += 1

    return str(candidate)


def generate_base_sub_dir(info_config: dict, system) -> Path:
    """
    Generate standardized subdirectory path based on system and configuration.
    WILL BE subdir of DATA_DIR OR FIGURES_DIR

    Args:
        info_config: Dictionary containing simulation parameters
        system: System parameters object

    Returns:
        Path: Relative path for data storage
    """
    # Base directory based on number of atoms and solver type
    parts = []

    # Add simulation dimension (1d/2d)
    if "simulation_type" in info_config:
        parts.append(f"{info_config['simulation_type']}_spectroscopy")
    else:
        # Default to the most common case
        parts.append("spectroscopy")

    # Add system details
    N_atoms = system.N_atoms
    parts.append(f"N{N_atoms}")

    # Add solver if available
    parts.append(system.ODE_Solver)

    # Add RWA if available
    parts.append("RWA" if system.RWA_laser else "noRWA")

    # Add time parameters
    parts.append(f"T_det_MAX_{info_config.get('t_det_max', 'not_provided')}")
    parts.append(f"T_wait_{info_config.get('t_wait', 'not_provided')}")
    parts.append(f"dt_{system.dt:.1f}fs")

    if N_atoms == 2:
        # Add coupling strength if applicable
        J_cm = system.J_cm
        if J_cm > 0:
            parts.append(f"Coupled")

    n_freqs = info_config.get("n_freqs", 1)
    if n_freqs > 1:
        parts.append("inhom")
    # Join all parts with path separator
    return Path(*parts)


def generate_unique_data_filename(
    system: SystemParameters,
    info_config: dict,
) -> str:
    """
    Build a standardized filename for data files.

    Args:
        system: System parameters object
        info_config: Dictionary containing simulation parameters

    Returns:
        str: Standardized base filename for the data file (without extension)
    """
    # Start with basic structure
    relative_path = generate_base_sub_dir(info_config, system)
    path = DATA_DIR / relative_path
    path.mkdir(parents=True, exist_ok=True)
    base_name = _generate_base_filename(system, info_config)
    filename = _generate_unique_filename(path, base_name)
    return filename


def generate_unique_plot_filename(
    system: SystemParameters,
    info_config: dict,
    domain: str,
    component: str = None,
) -> str:
    """
    Build a standardized filename for plots.

    Args:
        system: System parameters object
        info_config: Dictionary containing simulation parameters
        domain: Data domain ("time" or "freq")
        component: Optional component name ("real", "imag", "abs", "phase")

    Returns:
        str: Standardized base filename for the plot (without extension)
    """
    # Validate domain
    if domain not in {"time", "freq"}:
        raise ValueError(f"Invalid domain '{domain}'. Expected 'time' or 'freq'.")

    # Validate component if provided
    if component and component not in {"real", "imag", "abs", "phase"}:
        raise ValueError(
            f"Invalid component '{component}'. Expected one of 'real', 'imag', 'abs', 'phase'."
        )

    # Start with basic structure
    relative_path = generate_base_sub_dir(info_config, system)
    path = FIGURES_PYTHON_DIR / relative_path
    path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    base_name = _generate_base_filename(system, info_config)
    base_name += f"_{domain}_domain"
    if component:
        base_name += f"_{component}"
    filename = _generate_unique_filename(path, base_name)
    return filename


def main():
    """
    Test function for filename generation utilities.
    """
    # =============================
    # TEST SETUP
    # =============================

    ### Create mock system parameters
    class MockSystemParameters:
        def __init__(self):
            self.N_atoms = 2
            self.omega_A_cm = 1.5
            self.Delta_cm = 0.0
            self.dt = 0.1
            self.t_max = 100.0
            self.ODE_Solver = "runge_kutta"
            self.RWA_laser = True

    system = MockSystemParameters()

    ### Create mock data configuration
    info_config = {
        "simulation_type": "2d",
        "t_wait": 50,
        "t_det_max": 200,
        "n_phases": 32,
        "n_freqs": 64,
    }

    # =============================
    # TEST UNIQUENESS
    # =============================

    print("Testing filename uniqueness:")
    print("=" * 50)

    ### Test data filename uniqueness
    print("\n### Testing data filename uniqueness:")
    data_filenames = []
    for i in range(5):
        filename = generate_unique_data_filename(system, info_config)
        data_filenames.append(filename)
        print(f"Generated #{i+1}: {filename}")

        # Create the file to test uniqueness
        Path(filename).touch()

    ### Verify all filenames are unique
    unique_data_files = set(data_filenames)
    print(
        f"\nGenerated {len(data_filenames)} filenames, {len(unique_data_files)} unique"
    )
    assert len(data_filenames) == len(
        unique_data_files
    ), "Data filenames are not unique!"

    ### Test plot filename uniqueness
    print("\n### Testing plot filename uniqueness:")
    plot_filenames = []
    for i in range(5):
        filename = generate_unique_plot_filename(system, info_config, "time", "real")
        plot_filenames.append(filename)
        print(f"Generated #{i+1}: {filename}")

        # Create the file to test uniqueness
        Path(filename).touch()

    ### Verify all plot filenames are unique
    unique_plot_files = set(plot_filenames)
    print(
        f"\nGenerated {len(plot_filenames)} filenames, {len(unique_plot_files)} unique"
    )
    assert len(plot_filenames) == len(
        unique_plot_files
    ), "Plot filenames are not unique!"

    ### Test base unique function directly
    print("\n### Testing _generate_unique_filename directly:")
    test_dir = Path("test_unique_dir")
    test_dir.mkdir(exist_ok=True)

    base_name = "test_file"
    unique_filenames = []

    for i in range(3):
        filename = _generate_unique_filename(test_dir, base_name)
        unique_filenames.append(filename)
        print(f"Generated #{i+1}: {filename}")

        # Create the file to test uniqueness
        Path(filename).touch()

    ### Verify uniqueness
    unique_test_files = set(unique_filenames)
    print(
        f"\nGenerated {len(unique_filenames)} filenames, {len(unique_test_files)} unique"
    )
    assert len(unique_filenames) == len(
        unique_test_files
    ), "Base unique function failed!"

    # =============================
    # CLEANUP
    # =============================

    ### Clean up created test files
    print("\n### Cleaning up test files...")
    for filename in data_filenames + plot_filenames + unique_filenames:
        try:
            Path(filename).unlink()
            print(f"Removed: {filename}")
        except FileNotFoundError:
            pass  # File already removed or doesn't exist

    # Remove test directory
    try:
        test_dir.rmdir()
        print(f"Removed directory: {test_dir}")
    except (FileNotFoundError, OSError):
        pass  # Directory not empty or doesn't exist

    print("\n✅ All uniqueness tests passed!")


if __name__ == "__main__":
    main()
