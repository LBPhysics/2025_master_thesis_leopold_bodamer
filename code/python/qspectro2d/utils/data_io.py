"""
Data I/O operations for qspectro2d.

This module provides functionality for loading and saving simulation data,
including standardized file formats and directory management.
"""

# =============================
# IMPORTS
# =============================
import numpy as np
import pickle
import os
import glob
from pathlib import Path
from typing import Dict, Optional, List, TYPE_CHECKING
from datetime import datetime

### Project-specific imports
from qspectro2d.core.bath_system.bath_class import BathSystem
from qspectro2d.config.paths import DATA_DIR

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
from qspectro2d.utils.file_naming import generate_unique_data_filename


# =============================
# DATA SAVING FUNCTIONS
# =============================
def save_data_file(
    abs_data_path: Path,
    data: np.ndarray,
    axis1: np.ndarray,
    axis2: Optional[np.ndarray] = None,
) -> None:
    """
    Save numpy data and axes to compressed .npz file.

    Args:
        abs_data_path: Absolute path for the numpy data file (.npz)
        data: Simulation results (1D or 2D data)
        axis1: First axis (e.g., time or frequency for 1D or 2D data)
        axis2: Second axis (e.g., coherence time for 2D data)
    """
    try:
        if axis2 is not None:
            np.savez_compressed(abs_data_path, data=data, axis1=axis1, axis2=axis2)
        else:
            np.savez_compressed(abs_data_path, data=data, axis1=axis1)
        print(f"✅ Data saved successfully to: {abs_data_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save data: {e}")
        raise


def save_info_file(
    abs_info_path: Path,
    system,  # AtomicSystem
    bath: BathSystem,
    laser: "LaserPulseSequence",
    info_config: dict,
) -> None:
    """
    Save system parameters and data configuration to pickle file.

    Args:
        abs_info_path: Absolute path for the info file (.pkl)
        system: System parameters object
        info_config: Simulation configuration dictionary
    """
    try:
        with open(abs_info_path, "wb") as info_file:
            pickle.dump(
                {
                    "system": system,
                    "bath": bath,
                    "laser": laser,
                    "info_config": info_config,
                },
                info_file,
            )
        print(f"✅ Info saved successfully to: {abs_info_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save info: {e}")
        raise


def save_simulation_data(
    system,  # AtomicSystem
    info_config: dict,
    bath: BathSystem,
    laser: "LaserPulseSequence",
    data: np.ndarray,
    axis1: np.ndarray,
    axis2: Optional[np.ndarray] = None,
) -> Path:
    """
    Save spectroscopy simulation data (numpy arrays) along with known axes in one file,
    and system parameters and configuration in another file.

    Parameters:
        data (np.ndarray): Simulation results (1D or 2D data).
        axis1 (np.ndarray): First axis (e.g., time or frequency for 1D or 2D data).
        axis2 (Optional[np.ndarray]): Second axis (e.g., coherence time for 2D data).
        system (AtomicSystem): System parameters object.
        info_config (dict): Simulation configuration dictionary.

    Returns:
        Path]: absolute path to DATA_DIR for the saved numpy data file and info file.
    """
    # =============================
    # Generate unique filenames
    # =============================
    abs_base_path = generate_unique_data_filename(system, info_config)
    abs_data_path = Path(f"{abs_base_path}_data.npz")
    abs_info_path = Path(f"{abs_base_path}_info.pkl")

    # =============================
    # Save files
    # =============================
    save_data_file(abs_data_path, data, axis1, axis2)
    save_info_file(abs_info_path, system, bath, laser, info_config)

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
    """
    Load simulation data from specific data and info file paths.

    Args:
        abs_path: absolute path for the numpy data file (.npz) or info file (.pkl)

    Returns:
        dict: Dictionary containing loaded data, axes, system, and info_config
    """
    ### Determine the base path (without file extensions)
    if abs_path.endswith("_data.npz"):
        base_path = abs_path[:-9]  # Remove '_data.npz'
    elif abs_path.endswith("_info.pkl"):
        base_path = abs_path[:-9]  # Remove '_info.pkl'
    else:
        base_path = abs_path

    print(f"🔍 Loading data from: {base_path}")
    # =============================
    # Load files
    # =============================
    ### Construct file paths
    abs_data_path = base_path + "_data.npz"
    abs_info_path = base_path + "_info.pkl"
    data_dict = load_data_file(Path(abs_data_path))
    info_dict = load_info_file(Path(abs_info_path))

    # =============================
    # Combine data and info into standardized structure
    # =============================
    result = {
        "data": data_dict.get("data"),
        "axes": {
            "axis1": data_dict.get("axis1"),
        },
        "system": info_dict.get("system"),
        "bath": info_dict.get("bath"),
        "laser": info_dict.get("laser"),
        "info_config": info_dict.get("info_config"),
    }
    # Add second axis if it exists (for 2D data)
    if "axis2" in data_dict:
        result["axes"]["axis2"] = data_dict["axis2"]

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
        abs_path_str = abs_path_str[:-9]  # Remove '_data.npz'
    print(f"📅 Loading latest file: {abs_path_str}")

    # =============================
    # Load and return the data
    # =============================
    return load_data_from_abs_path(abs_path_str)


def list_available_data_files(abs_base_dir: Path) -> Dict[str, dict]:
    """
    List all available data files in a directory with their metadata without loading the full data.

    Args:
        abs_base_dir: Base directory path absolute to DATA_DIR

    Returns:
        Dict[str, dict]: Dictionary with file paths as keys and metadata as values
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
