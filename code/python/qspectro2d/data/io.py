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
from typing import Dict, Optional, List
from datetime import datetime

### Project-specific imports
from qspectro2d.core.system_parameters import SystemParameters
from qspectro2d.config.paths import DATA_DIR  # , FIGURES_DIR

# Handle both relative imports (when imported as module) and absolute imports (when run directly)
try:
    from .files import generate_unique_data_filename
except ImportError:
    from qspectro2d.data.files import generate_unique_data_filename


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
        with open(abs_info_path, "rb") as info_file:
            info = pickle.load(info_file)
        print(f"✅ Loaded info from: {abs_info_path}")
        return info
    except Exception as e:
        print(f"❌ ERROR: Failed to load info from {abs_info_path}: {e}")
        raise


def load_data_from_rel_path(relative_path: str) -> dict:
    """
    Load simulation data from specific data and info file paths.

    Args:
        relative_path: Relative path to DATA_DIR for the numpy data file (.npz) and info file (.pkl)

    Returns:
        dict: Dictionary containing loaded data, axes, system, and info_config
    """
    # =============================
    # Convert relative paths to absolute paths
    # =============================
    abs_data_path = DATA_DIR / (str(relative_path) + "_data.npz")
    abs_info_path = DATA_DIR / (str(relative_path) + "_info.pkl")

    # =============================
    # Load files
    # =============================
    data_dict = load_data_file(abs_data_path)
    info_dict = load_info_file(abs_info_path)

    # =============================
    # Combine data and info into standardized structure
    # =============================
    result = {
        "data": data_dict.get("data"),
        "axes": {
            "axis1": data_dict.get("axis1"),  # Note: saved as 'axis1', loaded as 'axis1'
        },
        "system": info_dict.get("system"),
        "info_config": info_dict.get("info_config"),
    }

    # Add second axis if it exists (for 2D data)
    if "axis2" in data_dict:
        result["axes"]["axis2"] = data_dict["axis2"]

    return result


def load_latest_data_from_directory(base_dir: str) -> dict:
    """
    Find and load the most recent data file from a base directory and all subdirectories.

    Args:
        base_dir: Base directory path relative to DATA_DIR (e.g., "1d_spectroscopy")

    Returns:
        dict: The loaded data dictionary from the most recent file
    """
    # =============================
    # Convert to absolute path and validate
    # =============================
    abs_base_dir = DATA_DIR / base_dir

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

    # Convert to relative path string without _data.npz
    rel_path = latest_path.relative_to(DATA_DIR)
    rel_path_str = str(rel_path)
    if rel_path_str.endswith("_data.npz"):
        rel_path_str = rel_path_str[:-9]  # Remove '_data.npz'
    print(f"📅 Loading latest file: {rel_path_str}")

    # =============================
    # Load and return the data
    # =============================
    return load_data_from_rel_path(rel_path_str)


def list_available_data_files(base_dir: Path) -> Dict[str, dict]:
    """
    List all available data files in a directory with their metadata without loading the full data.

    Args:
        base_dir: Base directory path relative to DATA_DIR

    Returns:
        Dict[str, dict]: Dictionary with file paths as keys and metadata as values
    """
    # =============================
    # Convert to absolute path and validate
    # =============================
    abs_base_dir = DATA_DIR / base_dir

    if not abs_base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {abs_base_dir}")

    print(f"📋 Listing data files in: {abs_base_dir}")

    # =============================
    # Find all data files recursively
    # =============================
    data_pattern = str(abs_base_dir / "**" / "*_data.npz")
    data_files = glob.glob(data_pattern, recursive=True)

    if not data_files:
        print(f"⚠️  No data files found in {abs_base_dir}")
        return {}

    # =============================
    # Collect metadata for each file
    # =============================
    file_info = {}

    for data_file in data_files:
        data_path = Path(data_file)
        rel_data_path = data_path.relative_to(DATA_DIR)

        # Get file statistics
        stat = os.stat(data_path)

        # Check for corresponding info file
        info_file_name = data_path.name.replace("_data.npz", "_info.pkl")
        info_path = data_path.parent / info_file_name
        has_info_file = info_path.exists()

        file_info[str(rel_data_path)] = {
            "size_mb": stat.st_size / (1024 * 1024),  # Size in MB
            "modified_time": datetime.fromtimestamp(stat.st_mtime),
            "has_info_file": has_info_file,
            "info_file_path": (
                str(info_path.relative_to(DATA_DIR)) if has_info_file else None
            ),
        }

    # =============================
    # Print summary
    # =============================
    print(f"📊 Found {len(file_info)} data files:")
    # for file_path, info in file_info.items():
    #     status = "✅" if info["has_info_file"] else "❌"
    #     print(
    #         f"   {status} {file_path} ({info['size_mb']:.2f} MB, {info['modified_time'].strftime('%Y-%m-%d %H:%M')})"
    #     )

    return file_info


def list_data_files_in_directory(base_dir: Path) -> List[str]:
    """
    List all data files in a specific directory (non-recursive) as relative paths.

    Args:
        base_dir: Base directory path relative to DATA_DIR

    Returns:
        List[str]: List of relative paths (without _data.npz suffix) for files in the same directory
    """
    # =============================
    # Convert to absolute path and validate
    # =============================
    abs_base_dir = DATA_DIR / base_dir

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
    # Convert to relative paths and remove suffix
    # =============================
    rel_paths = []

    for data_file in data_files:
        rel_path = data_file.relative_to(DATA_DIR)
        rel_path_str = str(rel_path)

        # Remove '_data.npz' suffix
        if rel_path_str.endswith("_data.npz"):
            rel_path_str = rel_path_str[:-9]

        rel_paths.append(rel_path_str)

    # =============================
    # Sort paths for consistent output
    # =============================
    rel_paths.sort()

    print(f"📊 Found {len(rel_paths)} data files in directory")
    for path in rel_paths:
        print(f"   📄 {path}")

    return rel_paths


# =============================
# DATA SAVING FUNCTIONS
# =============================
def save_data_file(
    data_path: Path,
    data: np.ndarray,
    axis1: np.ndarray,
    axis2: Optional[np.ndarray] = None,
) -> None:
    """
    Save numpy data and axes to compressed .npz file.

    Args:
        data_path: Absolute path for the numpy data file (.npz)
        data: Simulation results (1D or 2D data)
        axis1: First axis (e.g., time or frequency for 1D or 2D data)
        axis2: Second axis (e.g., coherence time for 2D data)
    """
    try:
        if axis2 is not None:
            np.savez_compressed(data_path, data=data, axis1=axis1, axis2=axis2)
        else:
            np.savez_compressed(data_path, data=data, axis1=axis1)
        print(f"✅ Data saved successfully to: {data_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save data: {e}")
        raise


def save_info_file(
    info_path: Path, system: SystemParameters, info_config: dict
) -> None:
    """
    Save system parameters and data configuration to pickle file.

    Args:
        info_path: Absolute path for the info file (.pkl)
        system: System parameters object
        info_config: Simulation configuration dictionary
    """
    try:
        with open(info_path, "wb") as info_file:
            pickle.dump({"system": system, "info_config": info_config}, info_file)
        print(f"✅ Info saved successfully to: {info_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save info: {e}")
        raise


def save_simulation_data(
    system: SystemParameters,
    info_config: dict,
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
        system (SystemParameters): System parameters object.
        info_config (dict): Simulation configuration dictionary.

    Returns:
        Path]: Relative path to DATA_DIR for the saved numpy data file and info file.
    """
    # =============================
    # Generate unique filenames
    # =============================
    base_path = generate_unique_data_filename(system, info_config)
    data_path = Path(f"{base_path}_data.npz")
    info_path = Path(f"{base_path}_info.pkl")

    # =============================
    # Save files
    # =============================
    save_data_file(data_path, data, axis1, axis2)
    save_info_file(info_path, system, info_config)

    # =============================
    # Return relative path to DATA_DIR
    # =============================
    ### Remove both the suffix and the trailing '_data' from the filename for rel_path
    rel_path = data_path.with_suffix("")
    rel_path_str = str(rel_path)
    if rel_path_str.endswith("_data"):
        rel_path_str = rel_path_str[:-5]  # Remove '_data'
    rel_path = Path(rel_path_str).relative_to(DATA_DIR)
    return rel_path


# =============================
# TEST CODE (when run directly)
# =============================
if __name__ == "__main__":
    print("🧪 Testing qspectro2d.data.io module...")
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
                        f"   ⚙️  System: N_atoms={getattr(data['system'], 'N_atoms', 'Unknown')}"
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
