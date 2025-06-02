"""
Script to average two_d_datas from multiple pickle files in a memory-efficient way. AT THE moment
"""

import os
import pickle
import numpy as np
import glob
from typing import Tuple, Dict, Any


def get_pkl_files(folder_path: str) -> list:
    """
    Get all pickle files from the specified folder.

    Parameters
    ----------
    folder_path : str
        Path to folder containing pickle files

    Returns
    -------
    list
        List of pickle file paths
    """
    pkl_pattern = os.path.join(folder_path, "data_tmax_2000_dt_0.2*.pkl")
    pkl_files = glob.glob(pkl_pattern)
    pkl_files.sort()  # Sort for consistent processing order

    print(f"Found {len(pkl_files)} pickle files to process")
    return pkl_files


def get_array_shape_from_first_file(file_path: str) -> Tuple[int, int]:
    """
    Load first file to determine the shape of two_d_datas array.

    Parameters
    ----------
    file_path : str
        Path to first pickle file

    Returns
    -------
    Tuple[int, int]
        Shape of the two_d_datas array
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        # two_d_datas is a list containing numpy arrays
        shape = data["two_d_datas"][0].shape
        print(f"Array shape determined from first file: {shape}")
        print(f"Number of arrays in two_d_datas list: {len(data['two_d_datas'])}")
        return shape


def average_pkl_files_memory_efficient(folder_path: str, output_path: str) -> None:
    """
    Average two_d_datas from multiple pickle files in a memory-efficient way.

    Parameters
    ----------
    folder_path : str
        Path to folder containing pickle files
    output_path : str
        Path where to save the averaged result
    """
    # =============================
    # GET ALL PICKLE FILES
    # =============================
    pkl_files = get_pkl_files(folder_path)

    if len(pkl_files) == 0:
        raise ValueError("No pickle files found in the specified folder")

    # =============================
    # DETERMINE ARRAY SHAPE
    # =============================
    array_shape = get_array_shape_from_first_file(pkl_files[0])

    # =============================
    # INITIALIZE RUNNING AVERAGE
    # =============================
    running_sum = np.zeros(array_shape, dtype=np.float64)
    num_files = len(pkl_files)
    system_data = None
    times = None
    times_T = None

    # =============================
    # PROCESS FILES ONE BY ONE
    # =============================
    for i, file_path in enumerate(pkl_files):
        print(f"Processing file {i+1}/{num_files}: {os.path.basename(file_path)}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

            ### Add current file's data to running sum
            # two_d_datas is a list, we want the first (and likely only) element
            current_array = data["two_d_datas"][0].astype(np.float64)
            running_sum += current_array

            ### Store metadata from last file (could be any file)
            if i == num_files - 1:  # Last file
                system_data = data["system"]
                times = data["times"]
                times_T = data["times_T"]
                print(f"Stored metadata from last file: {os.path.basename(file_path)}")

    # =============================
    # CALCULATE FINAL AVERAGE
    # =============================
    averaged_two_d_datas = running_sum / num_files
    print(f"Calculated average over {num_files} files")

    # =============================
    # SAVE AVERAGED RESULT
    # =============================
    averaged_data = {
        "system": system_data,
        "times": times,
        "times_T": times_T,
        "two_d_datas": [
            averaged_two_d_datas
        ],  # Keep as list to maintain original structure
    }

    with open(output_path, "wb") as f:
        pickle.dump(averaged_data, f)

    print(f"Saved averaged data to: {output_path}")
    print(f"Final array shape: {averaged_two_d_datas.shape}")
    print(f"Final array dtype: {averaged_two_d_datas.dtype}")


if __name__ == "__main__":
    # =============================
    # CONFIGURATION
    # =============================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        script_dir, "papers_with_proteus_output/average_over_freqs"
    )
    output_path = os.path.join(data_path, "averaged_data.pkl")

    # =============================
    # RUN AVERAGING
    # =============================
    average_pkl_files_memory_efficient(data_path, output_path)
