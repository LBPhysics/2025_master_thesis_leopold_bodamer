"""
Generalized Script for Averaging Simulation Data from Pickle Files

This script provides functionality to average data from multiple pickle files
in a memory-efficient way, supporting both 1D and 2D simulation data formats.
"""

# =============================
# IMPORTS
# =============================
import os
import pickle
import numpy as np
import glob
from pathlib import Path
from typing import Tuple, Dict, List, Any, Union, Optional


def get_pkl_files(folder_path: str, pattern: str = "*.pkl") -> List[str]:
    """
    Get all pickle files matching the pattern from the specified folder.

    Parameters
    ----------
    folder_path : str
        Path to folder containing pickle files
    pattern : str
        Glob pattern to match files (default: "*.pkl")

    Returns
    -------
    List[str]
        List of pickle file paths
    """
    pkl_pattern = os.path.join(folder_path, pattern)
    pkl_files = glob.glob(pkl_pattern)
    pkl_files.sort()  # Sort for consistent processing order

    print(f"Found {len(pkl_files)} pickle files to process")
    return pkl_files


def detect_data_type(data: Dict[str, Any]) -> str:
    """
    Detect whether the data is from a 1D or 2D simulation.

    Parameters
    ----------
    data : Dict[str, Any]
        Data dictionary from a pickle file

    Returns
    -------
    str
        Either "1D" or "2D" indicating the data type
    """
    if "data_avg" in data:
        return "1D"
    elif "two_d_datas" in data:
        return "2D"
    else:
        raise ValueError(
            "Unknown data format: neither 1D nor 2D data structure detected"
        )


def determine_average_target(
    data: Dict[str, Any], data_type: str
) -> Tuple[np.ndarray, str]:
    """
    Determine the target array to average based on data type.

    Parameters
    ----------
    data : Dict[str, Any]
        Data dictionary from a pickle file
    data_type : str
        Either "1D" or "2D"

    Returns
    -------
    Tuple[np.ndarray, str]
        Target array to average and the key name in the data dictionary
    """
    if data_type == "1D":
        return data["data_avg"], "data_avg"
    elif data_type == "2D":
        # Return first array from two_d_datas list for shape determination
        return data["two_d_datas"][0], "two_d_datas"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def average_simulation_data(
    folder_path: str,
    output_path: str,
    file_pattern: str = "*.pkl",
    verbose: bool = True,
) -> None:
    """
    Average simulation data from multiple pickle files in a memory-efficient way.
    Supports both 1D and 2D simulation data formats.

    Parameters
    ----------
    folder_path : str
        Path to folder containing pickle files
    output_path : str
        Path where to save the averaged result
    file_pattern : str
        Glob pattern to match files (default: "*.pkl")
    verbose : bool
        Whether to print verbose output (default: True)
    """
    # =============================
    # GET ALL PICKLE FILES
    # =============================
    pkl_files = get_pkl_files(folder_path, file_pattern)

    if len(pkl_files) == 0:
        raise ValueError(
            f"No pickle files found in {folder_path} matching {file_pattern}"
        )

    # =============================
    # LOAD FIRST FILE TO DETERMINE DATA TYPE AND STRUCTURE
    # =============================
    with open(pkl_files[0], "rb") as f:
        first_file_data = pickle.load(f)
        data_type = detect_data_type(first_file_data)

        if verbose:
            print(f"Detected {data_type} simulation data")

        first_array, target_key = determine_average_target(first_file_data, data_type)

    # =============================
    # INITIALIZE STORAGE FOR AVERAGING
    # =============================
    num_files = len(pkl_files)
    metadata = {}  # Will store all metadata except the target arrays

    if data_type == "1D":
        # For 1D: single array to average
        running_sum = np.zeros_like(first_array, dtype=np.float64)
        if verbose:
            print(f"1D array shape: {first_array.shape}, dtype: {first_array.dtype}")

    elif data_type == "2D":
        # For 2D: potentially multiple arrays in two_d_datas list
        n_arrays = len(first_file_data["two_d_datas"])
        running_sums = [
            np.zeros_like(arr, dtype=np.float64)
            for arr in first_file_data["two_d_datas"]
        ]
        if verbose:
            print(f"2D contains {n_arrays} arrays to average")
            for i, arr in enumerate(first_file_data["two_d_datas"]):
                print(f"  Array {i} shape: {arr.shape}, dtype: {arr.dtype}")

    # =============================
    # PROCESS FILES ONE BY ONE
    # =============================
    for i, file_path in enumerate(pkl_files):
        if verbose:
            print(f"Processing file {i+1}/{num_files}: {os.path.basename(file_path)}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

            # Add current file's data to running sum
            if data_type == "1D":
                running_sum += data["data_avg"].astype(np.float64)
            elif data_type == "2D":
                for j, arr in enumerate(data["two_d_datas"]):
                    running_sums[j] += arr.astype(np.float64)

            # Store metadata from last file
            if i == num_files - 1:
                for key, value in data.items():
                    if key != target_key:  # Skip the target arrays we're averaging
                        metadata[key] = value
                if verbose:
                    print(
                        f"Stored metadata from last file: {os.path.basename(file_path)}"
                    )

    # =============================
    # CALCULATE FINAL AVERAGE
    # =============================
    if data_type == "1D":
        averaged_data = running_sum / num_files
        if verbose:
            print(f"Calculated average over {num_files} files")
    elif data_type == "2D":
        averaged_data = [running_sum / num_files for running_sum in running_sums]
        if verbose:
            print(
                f"Calculated average over {num_files} files for {len(averaged_data)} arrays"
            )

    # =============================
    # PREPARE RESULT DICTIONARY
    # =============================
    result = metadata.copy()

    if data_type == "1D":
        result["data_avg"] = averaged_data
    elif data_type == "2D":
        result["two_d_datas"] = averaged_data

    # =============================
    # SAVE AVERAGED RESULT
    # =============================
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    if verbose:
        print(f"Saved averaged data to: {output_path}")
        if data_type == "1D":
            print(f"Final array shape: {averaged_data.shape}")
            print(f"Final array dtype: {averaged_data.dtype}")
        elif data_type == "2D":
            for i, arr in enumerate(averaged_data):
                print(f"Final array {i} shape: {arr.shape}")
                print(f"Final array {i} dtype: {arr.dtype}")


if __name__ == "__main__":
    # =============================
    # CONFIGURATION
    # =============================
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example for 1D data averaging
    data_path = os.path.join(script_dir, "../data/1d_spectroscopy/average_BR_RWA_avg")
    output_path = os.path.join(
        script_dir, "../data/1d_spectroscopy/", "1d_data_averaged_501.pkl"
    )
    file_pattern = "1d_data_*.pkl"

    # Example for 2D data averaging
    # data_path = os.path.join(script_dir, "../data/2d_spectroscopy/multiple_runs")
    # output_path = os.path.join(data_path, "2d_data_averaged.pkl")
    # file_pattern = "2d_data_*.pkl"

    # =============================
    # RUN AVERAGING
    # =============================
    average_simulation_data(data_path, output_path, file_pattern)
