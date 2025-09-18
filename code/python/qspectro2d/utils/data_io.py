"""
Data I/O operations for qspectro2d.

This module provides functionality for loading and saving simulation data,
including standardized file formats and directory management.
"""

from __future__ import annotations
from curses import meta
import signal
from unittest import result


# IMPORTS

import numpy as np
import pickle
import glob
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING
from qutip import BosonicEnvironment

### Project-specific imports
from project_config.paths import DATA_DIR
from project_config.logging_setup import get_logger

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.simulation import SimulationConfig, SimulationModuleOQS
from qspectro2d.utils.file_naming import generate_unique_data_filename

logger = get_logger(__name__)


# data saving functions
def save_data_file(
    abs_data_path: Path,
    metadata: Optional[dict],
    datas: List[np.ndarray],
    t_det: np.ndarray,
    t_coh: Optional[np.ndarray] = None,
) -> None:
    """Save spectroscopy data(s) with a single np.savez_compressed call.

    Distinctions:
      - Dimensionality (1D vs 2D) inferred from t_coh is None or not.
      - Single vs multi-component data inferred from provided `datas`.

        Stored keys:
            - Axes: 't_det' and optionally 't_coh'
            - One array per signal type stored under its signal name (metadata['signal_types'])
            - all other metadata key-value pairs are stored at the top level
    """
    try:
        abs_data_path.parent.mkdir(parents=True, exist_ok=True)

        # Infer dimensionality
        is_2d = t_coh is not None

        # Base payload
        payload: dict = {
            "t_det": t_det,
        }
        if is_2d:
            payload["t_coh"] = t_coh

        # Optional metadata (e.g., inhom batching info)
        for k, v in metadata.items():
            payload[k] = v

        # Validate and populate component keys
        signal_types = metadata["signal_types"]
        if len(signal_types) != len(datas):
            raise ValueError(
                f"Length of signal_types ({len(signal_types)}) must match number of datas ({len(datas)})"
            )

        for i, data in enumerate(datas):
            sig_key = signal_types[i]
            if is_2d:
                if not isinstance(data, np.ndarray) or data.shape != (len(t_coh), len(t_det)):
                    raise ValueError(
                        f"2D data must have shape (len(t_coh), len(t_det)) = ({len(t_coh)}, {len(t_det)})"
                    )
            else:
                if not isinstance(data, np.ndarray) or data.shape != (len(t_det),):
                    raise ValueError(f"1D data must have shape (len(t_det),) = ({len(t_det)},)")
            payload[sig_key] = data

        # Single write
        np.savez_compressed(abs_data_path, **payload)

    except Exception:
        logger.exception("Failed to save data")
        raise


def save_info_file(
    abs_info_path: Path,
    system: "AtomicSystem",
    bath: BosonicEnvironment,
    laser: "LaserPulseSequence",
    sim_config: "SimulationConfig",
) -> None:
    """
    Save system parameters and data configuration to pickle file.

    Args:
        abs_info_path: Absolute path for the info file (.pkl)
        system: System parameters object
        bath: QuTip Environment instance
        laser: Laser pulse sequence object
        sim_config: SimulationConfig instance used for the run (stored as object, not dict)
    """
    try:
        with open(abs_info_path, "wb") as info_file:
            pickle.dump(
                {
                    "system": system,
                    "bath": bath,
                    "laser": laser,
                    # Store the SimulationConfig instance directly for full fidelity
                    "sim_config": sim_config,
                },
                info_file,
            )
        logger.info("Info saved: %s", abs_info_path)
    except Exception:
        logger.exception("Failed to save info")
        raise


def save_simulation_data(
    sim_module: SimulationModuleOQS,
    metadata: dict,
    datas: List[np.ndarray],
    t_det: np.ndarray,
    t_coh: Optional[np.ndarray] = None,
) -> Path:
    """
    Save spectroscopy simulation data (numpy arrays) along with known axes in one file,
    and system parameters and configuration in another file.

    Parameters:
        datas (List[np.ndarray]): Simulation results (1D/2D or absorptive tuple).
        t_det (np.ndarray): Detection time axis.
        t_coh (Optional[np.ndarray]): Coherence time axis for 2D data.

    Returns:
        Path]: absolute path to DATA_DIR for the saved numpy data file and info file.
    """

    system: "AtomicSystem" = sim_module.system
    sim_config: "SimulationConfig" = sim_module.simulation_config
    bath: "BosonicEnvironment" = sim_module.bath
    laser: "LaserPulseSequence" = sim_module.laser

    # Generate unique filenames
    abs_base_path = generate_unique_data_filename(system, sim_config)
    abs_data_path = Path(f"{abs_base_path}_data.npz")  # still legacy suffix pattern
    abs_info_path = Path(f"{abs_base_path}_info.pkl")

    # Save files
    save_data_file(abs_data_path, metadata, datas, t_det, t_coh=t_coh)
    save_info_file(abs_info_path, system, bath, laser, sim_config)

    # Return absolute path to DATA_DIR
    return abs_data_path


# data loading functions
def load_data_file(abs_data_path: Path) -> dict:
    """
    Load numpy data file (.npz) from absolute path.

    Args:
        abs_data_path: Absolute path to the numpy data file (.npz)

    Returns:
        dict: Dictionary containing loaded numpy data arrays
    """
    try:
        logger.debug("Loading data: %s", abs_data_path)

        with np.load(abs_data_path, allow_pickle=True) as data_file:
            data_dict = {key: data_file[key] for key in data_file.files}
        # Enforce required key
        if "t_det" not in data_dict:
            raise KeyError("Missing 't_det' axis in data file (new format requirement)")
        logger.info("Loaded data: %s", abs_data_path)
        return data_dict
    except Exception:
        logger.exception("Failed to load data: %s", abs_data_path)
        raise


def load_info_file(abs_info_path: Path) -> dict:
    """
    Load pickle info file (.pkl) from absolute path.

    Args:
        abs_info_path: Absolute path to the info file (.pkl)

    Returns:
        dict: Dictionary containing system parameters and data configuration
    """
    try:
        logger.debug("Loading info: %s", abs_info_path)

        # 1. Try to load the file directly if it exists
        if abs_info_path.exists():
            with open(abs_info_path, "rb") as info_file:
                info = pickle.load(info_file)
            logger.info("Loaded info: %s", abs_info_path)
            return info
    except Exception:
        logger.exception("Failed to load info: %s", abs_info_path)
        raise


def load_simulation_data(abs_path: Path) -> dict:
    """Load simulation data (new format only)."""
    # Determine the base path (without file extensions)
    if abs_path.endswith("_data.npz"):
        abs_data_path = Path(abs_path)
        abs_info_path = Path(abs_path[:-9] + "_info.pkl")
    elif abs_path.endswith("_info.pkl"):
        abs_info_path = Path(abs_path)
        abs_data_path = Path(abs_path[:-9] + "_data.npz")
    else:
        raise ValueError("Path must end with '_data.npz' or '_info.pkl'")

    logger.debug("Loading data bundle: %s", abs_path)
    data_dict = load_data_file(abs_data_path)
    info_dict = load_info_file(abs_info_path)

    # Axes
    t_det = data_dict.get("t_det")
    t_coh = data_dict.get("t_coh") if "t_coh" in data_dict else None
    is_2d = t_coh is not None and t_coh.size > 0 if isinstance(t_coh, np.ndarray) else False

    # Base result structure
    sim_config = info_dict["sim_config"]
    result: dict = {
        "system": info_dict.get("system"),
        "bath": info_dict.get("bath"),
        "laser": info_dict.get("laser"),
        "sim_config": sim_config,
        "t_det": t_det,
    }

    # Optional coherence axis
    if is_2d:
        try:
            if hasattr(t_coh, "__len__") and len(t_coh) > 0:
                result["t_coh"] = t_coh
        except Exception:
            pass

    for signal_type in sim_config.signal_types:
        result[signal_type] = data_dict.get(signal_type)

    result["metadata"] = data_dict.get("metadata")

    return result


def list_available_files(abs_base_dir: Path) -> List[str]:
    """
    List all available data files in a directory with their metadata without loading the full data.

    Args:
        abs_base_dir: Base directory path absolute to DATA_DIR

    Returns:
        List[str]: Sorted list of data/info file paths (strings)
    """
    if not abs_base_dir.is_dir():
        logger.warning("Not a directory, using parent: %s", abs_base_dir)
        abs_base_dir = abs_base_dir.parent
    if not abs_base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {abs_base_dir}")

    logger.debug("Listing data files in: %s", abs_base_dir)

    # Find all data files recursively
    logger.debug("Listing data files in: %s", abs_base_dir)

    # Find all data and info files recursively
    data_pattern = str(abs_base_dir / "**" / "*_data.npz")
    info_pattern = str(abs_base_dir / "**" / "*_info.pkl")

    data_files = glob.glob(data_pattern, recursive=True)
    info_files = glob.glob(info_pattern, recursive=True)

    ### Combine and sort all file paths
    all_files = data_files + info_files
    all_files.sort()

    if not all_files:
        logger.warning("No data or info files found: %s", abs_base_dir)
        return []

    # Print summary
    logger.info("Found %d files in %s", len(all_files), abs_base_dir)
    for file_path in all_files:
        logger.debug("file: %s", file_path)

    return all_files
