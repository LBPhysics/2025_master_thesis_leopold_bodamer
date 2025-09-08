"""
Data I/O operations for qspectro2d.

This module provides functionality for loading and saving simulation data,
including standardized file formats and directory management.
"""

from __future__ import annotations

# =============================
# IMPORTS
# =============================
import numpy as np
import pickle
import os
import glob
from pathlib import Path
from typing import Dict, Optional, List, TYPE_CHECKING

### Project-specific imports
from project_config.paths import DATA_DIR

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.simulation import SimulationConfig
from qutip import BosonicEnvironment
from qspectro2d.utils.file_naming import generate_unique_data_filename


# =============================
# DATA SAVING FUNCTIONS
# =============================
def save_data_file(
    abs_data_path: Path,
    data: np.ndarray | tuple[np.ndarray, np.ndarray],
    t_det: np.ndarray,
    t_coh: Optional[np.ndarray] = None,
) -> None:
    """Save spectroscopy data.

    Supported inputs:
        - Single component: np.ndarray (1D or 2D)
        - Tuple (rephasing, non_rephasing): two np.ndarrays of identical shape

    Stored keys:
        Single: data, t_det, (t_coh), axes_description
        Tuple : rephasing, non_rephasing, components, t_det, (t_coh), axes_description

    Axis ordering:
        1D -> ["t_det → axis 0"]
        2D -> ["t_coh → axis 0", "t_det → axis 1"]
    """
    try:
        abs_data_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine dimensionality via provided axes
        is_2d = t_coh is not None
        axes_description = ["t_coh → axis 0", "t_det → axis 1"] if is_2d else ["t_det → axis 0"]

        # =============================
        # SINGLE COMPONENT
        # =============================
        if isinstance(data, np.ndarray):
            if is_2d:
                if data.shape != (len(t_coh), len(t_det)):
                    raise ValueError(
                        f"2D data must have shape (len(t_coh), len(t_det)) = ({len(t_coh)}, {len(t_det)}), got {data.shape}"
                    )
                np.savez_compressed(
                    abs_data_path,
                    data=data,
                    t_det=t_det,
                    t_coh=t_coh,
                    axes_description=np.array(axes_description, dtype=object),
                )
            else:
                if data.shape != (len(t_det),):
                    raise ValueError(
                        f"1D data must have shape (len(t_det),) = ({len(t_det)},), got {data.shape}"
                    )
                np.savez_compressed(
                    abs_data_path,
                    data=data,
                    t_det=t_det,
                    axes_description=np.array(axes_description, dtype=object),
                )
            print(f"✅ Data saved successfully to: {abs_data_path}")
            return

        # =============================
        # TWO-COMPONENT TUPLE (rephasing, non_rephasing)
        # =============================
        if isinstance(data, tuple):
            if len(data) != 2:
                raise ValueError("Tuple data must have exactly 2 elements (rephasing, non_rephasing)")
            rephasing, non_rephasing = data
            if not isinstance(rephasing, np.ndarray) or not isinstance(non_rephasing, np.ndarray):
                raise TypeError("Both tuple elements must be numpy arrays")
            if rephasing.shape != non_rephasing.shape:
                raise ValueError(
                    f"Tuple component shapes differ: {rephasing.shape} vs {non_rephasing.shape}"
                )
            if is_2d:
                if rephasing.shape != (len(t_coh), len(t_det)):
                    raise ValueError(
                        f"2D components must have shape (len(t_coh), len(t_det)) = ({len(t_coh)}, {len(t_det)}), got {rephasing.shape}"
                    )
            else:
                if rephasing.shape != (len(t_det),):
                    raise ValueError(
                        f"1D components must have shape (len(t_det),) = ({len(t_det)},), got {rephasing.shape}"
                    )
            payload = {
                "rephasing": rephasing,
                "non_rephasing": non_rephasing,
                "components": np.array(["rephasing", "non_rephasing"], dtype=object),
                "t_det": t_det,
                "axes_description": np.array(axes_description, dtype=object),
            }
            if is_2d:
                payload["t_coh"] = t_coh
            np.savez_compressed(abs_data_path, **payload)
            print(f"✅ Multi-component data saved to: {abs_data_path}")
            return

        raise TypeError("Unsupported data type for save_data_file")

    except Exception as e:
        print(f"❌ ERROR: Failed to save data: {e}")
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
        print(f"✅ Info saved successfully to: {abs_info_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save info: {e}")
        raise


def save_simulation_data(
    system: "AtomicSystem",
    sim_config: "SimulationConfig",
    bath: BosonicEnvironment,
    laser: "LaserPulseSequence",
    data: np.ndarray,
    t_det: np.ndarray,
    t_coh: Optional[np.ndarray] = None,
) -> Path:
    """
    Save spectroscopy simulation data (numpy arrays) along with known axes in one file,
    and system parameters and configuration in another file.

    Parameters:
        data (np.ndarray): Simulation results (1D or 2D data).
    t_det (np.ndarray): Detection time axis.
    t_coh (Optional[np.ndarray]): Coherence time axis for 2D data.
        system (AtomicSystem): System parameters object.
    sim_config (SimulationConfig): Simulation configuration object.

    Returns:
        Path]: absolute path to DATA_DIR for the saved numpy data file and info file.
    """
    # =============================
    # Generate unique filenames
    # =============================
    abs_base_path = generate_unique_data_filename(system, sim_config)
    abs_data_path = Path(f"{abs_base_path}_data.npz")  # still legacy suffix pattern
    abs_info_path = Path(f"{abs_base_path}_info.pkl")

    # =============================
    # Save files
    # =============================
    save_data_file(abs_data_path, data, t_det, t_coh)
    save_info_file(abs_info_path, system, bath, laser, sim_config)

    # =============================
    # Return absolute path to DATA_DIR
    # =============================
    return abs_data_path


# =============================
# DATA LOADING FUNCTIONS
# =============================
def load_data_file(abs_data_path: Path) -> dict:
    """
    Load numpy data file (.npz) from absolute path.

    Args:
        abs_data_path: Absolute path to the numpy data file (.npz)

    Returns:
        dict: Dictionary containing loaded numpy data arrays
    """
    try:
        with np.load(abs_data_path, allow_pickle=True) as data_file:
            data_dict = {key: data_file[key] for key in data_file.files}
        # Enforce required key
        if "t_det" not in data_dict:
            raise KeyError("Missing 't_det' axis in data file (new format requirement)")
        print(f"✅ Loaded data from: {abs_data_path}")
        return data_dict
    except Exception as e:
        print(f"❌ ERROR: Failed to load data from {abs_data_path}: {e}")
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
        print(f"🔍 Loading info from: {abs_info_path}")

        # 1. Try to load the file directly if it exists
        if abs_info_path.exists():
            with open(abs_info_path, "rb") as info_file:
                info = pickle.load(info_file)
            print(f"✅ Loaded info from: {abs_info_path}")
            return info

        # 2. If not found, search for any .pkl file in the same directory
        print(f"⚠️ File not found. Searching for any .pkl file in the same directory...")
        parent_dir = abs_info_path.parent

        # Find the first .pkl file in the directory
        pkl_files = list(parent_dir.glob("*.pkl"))

        if pkl_files:
            alt_path = pkl_files[0]  # Use the first .pkl file found
            print(f"🔄 Using alternative file: {alt_path}")
            with open(alt_path, "rb") as info_file:
                info = pickle.load(info_file)
            print(f"✅ Loaded info from: {alt_path}")
            return info
        else:
            raise FileNotFoundError(f"No .pkl files found in directory: {parent_dir}")

    except Exception as e:
        print(f"❌ ERROR: Failed to load info from {abs_info_path}: {e}")
        raise


def load_data_from_abs_path(abs_path: str) -> dict:
    """Load simulation data (new format only).

    Expects the saved file to contain a primary 'data' array (or user must
    explicitly handle multi-component files outside this helper).
    No automatic fallback / component selection is performed.
    """
    # Determine the base path (without file extensions)
    if abs_path.endswith("_data.npz"):
        base_path = abs_path[:-9]  # Remove '_data.npz'
    elif abs_path.endswith("_info.pkl"):
        base_path = abs_path[:-9]  # Remove '_info.pkl'
    else:
        base_path = abs_path

    print(f"🔍 Loading data from: {base_path}")
    abs_data_path = base_path + "_data.npz"
    abs_info_path = base_path + "_info.pkl"

    data_dict = load_data_file(Path(abs_data_path))
    info_dict = load_info_file(Path(abs_info_path))

    # Enforce presence of primary data key
    if "data" not in data_dict:
        raise KeyError(
            "'data' key missing in file. Multi-component files must now be handled explicitly by the caller."
        )

    if "sim_config" not in info_dict:
        raise KeyError("Missing 'sim_config' in info file")

    result = {
        "data": data_dict["data"],
        "axes": {"t_det": data_dict.get("t_det")},
        "system": info_dict.get("system"),
        "bath": info_dict.get("bath"),
        "laser": info_dict.get("laser"),
        "sim_config": info_dict.get("sim_config"),
    }
    if "t_coh" in data_dict and data_dict.get("t_coh") is not None:
        result["axes"]["t_coh"] = data_dict["t_coh"]
    return result


def load_latest_data_from_directory(abs_base_dir: str) -> dict:
    """
    Find and load the most recent data file from a base directory and all subdirectories.

    Args:
        abs_base_dir: Base directory path

    Returns:
        dict: The loaded data dictionary from the most recent file
    """
    # =============================
    # validate path

    if not abs_base_dir.is_dir():
        abs_base_dir = abs_base_dir.parent
    if not abs_base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {abs_base_dir}")

    print(f"🔍 Searching for latest data file in: {abs_base_dir}")

    # =============================
    # Find all data files recursively
    # =============================
    data_pattern = str(abs_base_dir / "**" / "*_data.npz")
    data_files = glob.glob(data_pattern, recursive=True)

    if not data_files:
        raise FileNotFoundError(f"No data files found in {abs_base_dir}")

    # =============================
    # Find the most recent file
    # =============================
    latest_file = max(data_files, key=os.path.getmtime)
    latest_path = Path(latest_file)

    # Convert to absolute path string without _data.npz
    abs_path = latest_path
    abs_path_str = str(abs_path)
    if abs_path_str.endswith("_data.npz"):
        abs_path_str = abs_path_str[:-9]  # Remove suffix
    print(f"📅 Loading latest file: {abs_path_str}")

    # =============================
    # Load and return the data
    # =============================
    return load_data_from_abs_path(abs_path_str)


def list_available_data_files(abs_base_dir: Path) -> List[str]:
    """
    List all available data files in a directory with their metadata without loading the full data.

    Args:
        abs_base_dir: Base directory path absolute to DATA_DIR

    Returns:
        List[str]: Sorted list of data/info file paths (strings)
    """
    if not abs_base_dir.is_dir():
        print(f"⚠️  {abs_base_dir} is not a directory, checking parent directory.")
        abs_base_dir = abs_base_dir.parent
    if not abs_base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {abs_base_dir}")

    print(f"📋 Listing data files in: {abs_base_dir}")

    # =============================
    # Find all data files recursively
    # =============================
    print(f"📋 Listing data files in: {abs_base_dir}")

    # =============================
    # Find all data and info files recursively
    # =============================
    data_pattern = str(abs_base_dir / "**" / "*_data.npz")
    info_pattern = str(abs_base_dir / "**" / "*_info.pkl")

    data_files = glob.glob(data_pattern, recursive=True)
    info_files = glob.glob(info_pattern, recursive=True)

    ### Combine and sort all file paths
    all_files = data_files + info_files
    all_files.sort()

    if not all_files:
        print(f"⚠️  No data or info files found in {abs_base_dir}")
        return []

    # =============================
    # Print summary
    # =============================
    print(f"📊 Found {len(all_files)} files:")
    for file_path in all_files:
        print(f"   📄 {file_path}")

    return all_files


def list_data_files_in_directory(abs_base_dir: Path) -> List[str]:
    """
    List all data files in a specific directory (non-recursive) as absolute paths.

    Args:
        abs_base_dir: Base directory path absolute

    Returns:
        List[str]: List of absolute paths (without _data.npz suffix) for files in the same directory
    """
    if not abs_base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {abs_base_dir}")

    print(f"📋 Listing data files in: {abs_base_dir}")

    # =============================
    # Find data files in current directory only (non-recursive)
    # =============================
    data_files = list(abs_base_dir.glob("*_data.npz"))

    if not data_files:
        print(f"⚠️  No data files found in {abs_base_dir}")
        return []

    # =============================
    # remove suffix
    abs_paths = []

    for data_file in data_files:
        abs_path = data_file
        abs_path_str = str(abs_path)

        # Remove '_data.npz' suffix
        if abs_path_str.endswith("_data.npz"):
            abs_path_str = abs_path_str[:-9]

        abs_paths.append(abs_path_str)

    # =============================
    # Sort paths for consistent output
    abs_paths.sort()

    print(f"📊 Found {len(abs_paths)} data files in directory")
    for path in abs_paths:
        print(f"   📄 {path}")

    return abs_paths


# =============================
# TEST CODE (when run directly)
# =============================
if __name__ == "__main__":
    print("🧪 Testing qspectro2d.utils.data_io module...")
    print("=" * 50)

    # Test 1: List available data files
    print("\n📋 Test 1: Listing available data files...")
    try:
        # Try to list files in common directories
        test_dirs = ["1d_spectroscopy", "2d_spectroscopy", "bath_correlator", "tests"]

        for test_dir in test_dirs:
            try:
                print(f"\n🔍 Checking directory: {test_dir}")
                file_info = list_available_data_files(Path(test_dir))
                if file_info:
                    print(f"   Found {len(file_info)} files")
                else:
                    print(f"   No files found in {test_dir}")
            except FileNotFoundError:
                print(f"   Directory {test_dir} does not exist")
            except Exception as e:
                print(f"   Error accessing {test_dir}: {e}")

    except Exception as e:
        print(f"❌ Error in test 1: {e}")

    # Test 2: Try to load latest data from a directory
    print("\n📅 Test 2: Loading latest data...")
    try:
        # Try common directories
        for test_dir in ["1d_spectroscopy", "2d_spectroscopy", "tests"]:
            try:
                print(f"\n🔍 Attempting to load latest from: {test_dir}")
                data = load_latest_data_from_directory(Path(test_dir))
                print(f"   ✅ Successfully loaded data with keys: {list(data.keys())}")

                # Print some basic info about the loaded data
                if "data" in data and data["data"] is not None:
                    print(f"   📊 Data shape: {data['data'].shape}")
                if "axes" in data and data["axes"]:
                    print(f"   📏 Axes: {list(data['axes'].keys())}")
                if "system" in data and data["system"]:
                    print(
                        f"   ⚙️  System: n_atoms={getattr(data['system'], 'n_atoms', 'Unknown')}"
                    )

                # Only test the first successful load to avoid too much output
                break

            except FileNotFoundError:
                print(f"   No data files found in {test_dir}")
            except Exception as e:
                print(f"   Error loading from {test_dir}: {e}")

    except Exception as e:
        print(f"❌ Error in test 2: {e}")

    # Test 4: Show DATA_DIR information
    print(f"\n📁 Test 4: DATA_DIR information...")
    try:
        print(f"   DATA_DIR: {DATA_DIR}")
        print(f"   Exists: {DATA_DIR.exists()}")
        if DATA_DIR.exists():
            subdirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
            print(
                f"   Subdirectories: {[d.name for d in subdirs[:10]]}"
            )  # Show first 10
            if len(subdirs) > 10:
                print(f"   ... and {len(subdirs) - 10} more")
    except Exception as e:
        print(f"❌ Error in test 4: {e}")

    print("\n" + "=" * 50)
    print("🏁 Testing complete!")
