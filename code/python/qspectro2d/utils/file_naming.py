"""
File naming and path utilities for qspectro2d.

This module provides functionality for generating standardized filenames
and directory paths for simulation data and plots.
"""

# =============================
# IMPORTS
# =============================
import fnmatch
from pathlib import Path
from typing import Union

### Project-specific imports
from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.config.paths import DATA_DIR, FIGURES_PYTHON_DIR


# =============================
# FILENAME GENERATION FUNCTIONS
# =============================
def _generate_base_filename(system: AtomicSystem, info_config: dict) -> str:
    """
    Generate a universal base filename for a calculation based on system and info_config parameters.

    Args:
        system: System parameters object
        info_config: Dictionary containing simulation parameters

    Returns:
        str: Base filename without path
    """

    parts = []

    if info_config.get("simulation_type") == "1d":
        # Round t_coh to 2 decimal places for filename clarity
        t_coh_val = round(float(info_config["t_coh"]), 2)
        parts.append(f"t_coh_{t_coh_val}")
    else:
        parts.append(system.freqs_cm.__str__())

    """
    n_atoms = system.n_atoms
    # simulation_type = info_config.get("simulation_type", "spectroscopy")
    # parts.append(simulation_type)
    # parts.append(f"N{n_atoms}")
    parts.append(f"wA{system.omega_A_cm/1e4:.2f}e4")
    parts.append(f"muA{system.mu_A:.2f}")
    if n_atoms == 2:
        parts.append(f"wB{system.omega_B_cm/1e4:.2f}e4")
        parts.append(f"muB{system.mu_B:.2f}")
        J_val = system.J if system.J else info_config.get("J_cm", 0)
        if J_val > 0:
            parts.append(f"J{J_val:.2f}au")  # TODO arbitrary units
    n_freqs = info_config.get("n_freqs", 1)

    if n_freqs > 1:
        parts.append(f"Delta{system.delta_cm/1e4:.2f}e4")
    parts.append("cm-1")
    """
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
    counter = 1

    ### Start with the base name
    candidate_name = base_name

    ### Check if any files with this base name already exist (with any extension)
    while True:
        # Use Path.glob() but we need to handle special characters
        # Let's check all files in directory and filter manually
        try:
            all_files = list(path.iterdir())
            existing_files = [
                f for f in all_files if f.is_file() and f.stem == candidate_name
            ]
        except FileNotFoundError:
            # Directory doesn't exist yet
            existing_files = []

        if not existing_files:
            # No files with this name exist, we can use it
            break

        # Files exist, try next candidate
        print(
            f"🔍 Found {len(existing_files)} existing files with name: {candidate_name}"
        )
        candidate_name = f"{base_name}_{counter}"
        counter += 1

    ### Return the full path with unique base name
    result = str(path / candidate_name)
    print(f"✅ Generated unique filename: {result}")
    return result


def generate_base_sub_dir(info_config: dict, system: AtomicSystem) -> Path:
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
    n_atoms = system.n_atoms
    parts.append(f"N{n_atoms}")

    # Add solver if available
    parts.append(info_config["ode_solver"])

    # Add RWA if available
    parts.append("RWA" if info_config["rwa_sl"] else "noRWA")

    # Add time parameters
    parts.append(
        f"t_dm{info_config.get('t_det_max', 'not_provided')}"
    )  # maximum detection time
    parts.append(f"t_wait_{info_config.get('t_wait', 'not_provided')}")
    parts.append(f"dt_{info_config.get('dt', 'not_provided')}")

    if n_atoms == 2:
        # Add coupling strength if applicable
        J = system.J_cm if system.J_cm is not None else 0
        if J > 0:
            parts.append(f"Coupled")

    n_freqs = info_config.get("n_freqs", 1)
    if n_freqs > 1:
        parts.append("inhom")
    # Join all parts with path separator
    return Path(*parts)


def generate_unique_data_filename(
    system: AtomicSystem,
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
    abs_path = DATA_DIR / relative_path
    print("saving data to:", abs_path)
    abs_path.mkdir(parents=True, exist_ok=True)
    base_name = _generate_base_filename(system, info_config)
    filename = _generate_unique_filename(abs_path, base_name)
    return filename


def generate_unique_plot_filename(
    system: AtomicSystem,
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
    print(f"Generated unique plot filename: {filename}")
    return filename
