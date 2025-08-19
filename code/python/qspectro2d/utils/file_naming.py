"""
File naming and path utilities for qspectro2d.

This module provides functionality for generating standardized filenames
and directory paths for simulation data and plots.
"""

# =============================
# IMPORTS
# =============================
from __future__ import annotations

from pathlib import Path
from typing import Union, Any, Protocol, runtime_checkable, TYPE_CHECKING


@runtime_checkable
class AtomicSystemProto(Protocol):  # minimal structural contract for naming
    n_atoms: int
    coupling_cm: float | int
    frequencies_cm: Any
    delta_cm: Any


@runtime_checkable
class SimulationConfigProto(Protocol):  # minimal structural contract for naming
    simulation_type: str
    ode_solver: str
    rwa_sl: bool
    t_det_max: float
    t_wait: float
    dt: float
    t_coh: float
    n_freqs: int


if TYPE_CHECKING:  # real imports only for type checkers
    from qspectro2d.core.atomic_system.system_class import (
        AtomicSystem as AtomicSystemType,
    )
    from qspectro2d.core.simulation import SimulationConfig as SimulationConfigType

    AtomicSystem = AtomicSystemType  # alias for readability
    SimulationConfig = SimulationConfigType
else:  # runtime uses Protocols (duck typing)
    AtomicSystem = AtomicSystemProto  # type: ignore
    SimulationConfig = SimulationConfigProto  # type: ignore

### Project-specific imports
from project_config.paths import DATA_DIR, FIGURES_PYTHON_DIR, ensure_dirs


# =============================
# FILENAME GENERATION FUNCTIONS
# =============================
def _extract_system_fields(system: AtomicSystem) -> dict:
    """Extract required filename-related fields from an AtomicSystem.

    Assumes attributes exist; keeps minimalist surface for naming only.
    """
    freqs = getattr(system, "frequencies_cm", None)
    return {
        "n_atoms": getattr(system, "n_atoms", None),
        "coupling_cm": getattr(system, "coupling_cm", None),
        "freqs": freqs,
        "delta_cm": getattr(system, "delta_cm", None),
    }


def _extract_sim_fields(sim: SimulationConfig) -> dict:
    return {
        "simulation_type": getattr(sim, "simulation_type", "1d"),
        "ode_solver": getattr(sim, "ode_solver", None),
        "rwa_sl": getattr(sim, "rwa_sl", None),
        "t_det_max": getattr(sim, "t_det_max", None),
        "t_wait": getattr(sim, "t_wait", None),
        "dt": getattr(sim, "dt", None),
        "t_coh": getattr(sim, "t_coh", None),
        "n_freqs": getattr(sim, "n_freqs", 1),
    }


def _generate_base_filename(system: AtomicSystem, sim_config: SimulationConfig) -> str:
    sys_f = _extract_system_fields(system)
    sim_f = _extract_sim_fields(sim_config)
    parts: list[str] = []
    if sim_f["simulation_type"] == "1d":
        t_coh_val = sim_f.get("t_coh")
        if t_coh_val is not None:
            parts.append(f"t_coh_{round(float(t_coh_val), 2)}")
    else:
        freqs = sys_f.get("freqs")
        if freqs is not None:
            parts.append(str(freqs))
    return "_".join(parts) or "run"


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
    return result


def generate_base_sub_dir(sim_config: SimulationConfig, system: AtomicSystem) -> Path:
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
    sim_f = _extract_sim_fields(sim_config)
    sys_f = _extract_system_fields(system)

    parts.append(f"{sim_f['simulation_type']}_spectroscopy")

    # Add system details
    n_atoms = sys_f.get("n_atoms") or "N?"
    parts.append(f"N{n_atoms}")

    # Add solver if available
    parts.append(sim_f.get("ode_solver") or "solver?")

    # Add RWA if available
    parts.append("RWA" if sim_f.get("rwa_sl") else "noRWA")

    # Add time parameters
    parts.append(f"t_dm{sim_f.get('t_det_max', 'na')}")
    parts.append(f"t_wait_{sim_f.get('t_wait', 'na')}")
    parts.append(f"dt_{sim_f.get('dt', 'na')}")

    if n_atoms == 2:
        # Add coupling strength if applicable
        coupling_cm = sys_f.get("coupling_cm") or 0
        if coupling_cm > 0:
            parts.append(f"Coupled")

    n_freqs = sim_f.get("n_freqs", 1)
    if n_freqs > 1:
        parts.append("inhom")
    # Join all parts with path separator
    return Path(*parts)


def generate_unique_data_filename(
    system: AtomicSystem,
    sim_config: SimulationConfig,
    *,
    ensure: bool = True,
) -> str:
    """Return unique base filename (without extension) for a run.

    Accepts either object instances (AtomicSystem / SimulationConfig) or dict-like
    mappings for flexibility.
    """
    relative_path = generate_base_sub_dir(sim_config, system)
    if ensure:
        ensure_dirs()
    abs_path = DATA_DIR / relative_path
    base_name = _generate_base_filename(system, sim_config)
    return _generate_unique_filename(abs_path, base_name)


def generate_unique_plot_filename(
    system: AtomicSystem,
    sim_config: SimulationConfig,
    domain: str,
    component: str | None = None,
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
    relative_path = generate_base_sub_dir(sim_config, system)
    ensure_dirs()
    path = FIGURES_PYTHON_DIR / relative_path  # Caller may create as needed

    base_name = _generate_base_filename(system, sim_config)
    base_name += f"_{domain}_domain"
    if component:
        base_name += f"_{component}"
    filename = _generate_unique_filename(path, base_name)
    return filename
