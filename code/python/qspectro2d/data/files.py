"""
File naming and path utilities for qspectro2d.

This module provides functionality for generating standardized filenames
and directory paths for simulation data and plots.
"""

# =============================
# IMPORTS
# =============================
from pathlib import Path
from typing import Union

### Project-specific imports
from qspectro2d.core.system_parameters import SystemParameters
from config.paths import DATA_DIR, FIGURES_DIR, FIGURES_PYTHON_DIR


# =============================
# FILENAME GENERATION FUNCTIONS
# =============================
def _generate_base_filename(system: SystemParameters, data_config: dict) -> str:
    """
    Generate a universal base filename for a calculation based on system and data_config parameters.

    Args:
        system: System parameters object
        data_config: Dictionary containing simulation parameters

    Returns:
        str: Base filename without path
    """
    simulation_type = data_config.get("simulation_type", "spectroscopy")
    N_atoms = system.N_atoms

    parts = [simulation_type]
    parts.append(f"N{N_atoms}")
    parts.append(f"wA{system.omega_A:.2f}")

    if N_atoms == 2:
        parts.append(f"T_wait{data_config.get('t_wait', 0)}fs")

    parts += [
        f"t_det_max_{data_config.get('t_det_max', 0)}",
        f"dt_{system.dt:.1f}",
        f"{data_config.get('n_phases', 0)}ph",
        f"{data_config.get('n_freqs', 1)}freq",
        f"1",
    ]

    return "_".join(parts)


def _generate_unique_filename(path: Union[str, Path], base_name: str) -> str:
    """
    Generate a unique filename in the specified directory.

    Args:
        path (str or Path): Directory where the file will be saved
        base_name (str): Base name for the file (without extension)

    Returns:
        str: Unique filename with full path
    """
    path = Path(path)
    save_path = path / base_name
    counter = 1

    while save_path.exists():
        save_path = path / f"{base_name}_{counter}"
        counter += 1

    return str(save_path)


def generate_base_sub_dir(data_config: dict, system) -> Path:
    """
    Generate standardized subdirectory path based on system and configuration.
    WILL BE subdir of DATA_DIR OR FIGURES_DIR

    Args:
        data_config: Dictionary containing simulation parameters
        system: System parameters object

    Returns:
        Path: Relative path for data storage
    """
    # Base directory based on number of atoms and solver type
    parts = []

    # Add simulation dimension (1d/2d)
    if "simulation_type" in data_config:
        parts.append(f"{data_config['simulation_type']}_spectroscopy")
    else:
        # Default to the most common case
        parts.append("spectroscopy")

    # Add system details
    parts.append(f"N{system.N_atoms}")

    # Add solver if available
    parts.append(system.ODE_Solver)
    parts.append(f"t_max{system.t_max:.1f}fs")

    # Add RWA if available
    parts.append("RWA" if system.RWA_laser else "noRWA")

    # Join all parts with path separator
    return Path(*parts)


def generate_unique_data_filename(
    system: SystemParameters,
    data_config: dict,
) -> str:
    """
    Build a standardized filename for data files.

    Args:
        system: System parameters object
        data_config: Dictionary containing simulation parameters

    Returns:
        str: Standardized base filename for the data file (without extension)
    """
    # Start with basic structure
    relative_path = generate_base_sub_dir(data_config, system)
    path = DATA_DIR / relative_path
    path.mkdir(parents=True, exist_ok=True)
    base_name = _generate_base_filename(system, data_config)
    filename = _generate_unique_filename(path, base_name)
    return filename


def generate_unique_plot_filename(
    system: SystemParameters,
    data_config: dict,
    domain: str,
    component: str = None,
) -> str:
    """
    Build a standardized filename for plots.

    Args:
        system: System parameters object
        data_config: Dictionary containing simulation parameters
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
    relative_path = generate_base_sub_dir(data_config, system)
    path = FIGURES_PYTHON_DIR / relative_path
    path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    base_name = _generate_base_filename(system, data_config)
    base_name += f"_{domain}_domain"
    if component:
        base_name += f"_{component}"

    filename = _generate_unique_filename(path, base_name)
    return filename
