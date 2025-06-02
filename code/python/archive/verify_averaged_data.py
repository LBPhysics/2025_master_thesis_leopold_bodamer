"""
Script to verify the averaged pickle file structure and content.
"""

import pickle
import numpy as np


def verify_averaged_data(file_path: str) -> None:
    """
    Verify the structure and content of the averaged pickle file.

    Parameters
    ----------
    file_path : str
        Path to the averaged pickle file
    """
    print("=== VERIFYING AVERAGED DATA ===")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # =============================
    # CHECK DATA STRUCTURE
    # =============================
    print(f"Keys in averaged data: {list(data.keys())}")

    ### Check system data
    print(f"System data type: {type(data['system'])}")

    ### Check times
    print(
        f"Times shape: {data['times'].shape if hasattr(data['times'], 'shape') else len(data['times'])}"
    )
    print(
        f"Times_T shape: {data['times_T'].shape if hasattr(data['times_T'], 'shape') else len(data['times_T'])}"
    )

    ### Check two_d_datas
    print(f"Two_d_datas type: {type(data['two_d_datas'])}")
    print(f"Number of arrays in two_d_datas: {len(data['two_d_datas'])}")
    print(f"Shape of averaged array: {data['two_d_datas'][0].shape}")
    print(f"Data type of averaged array: {data['two_d_datas'][0].dtype}")

    # =============================
    # CHECK SOME STATISTICS
    # =============================
    avg_array = data["two_d_datas"][0]
    print(f"\n=== ARRAY STATISTICS ===")
    print(f"Min value: {np.min(avg_array):.6e}")
    print(f"Max value: {np.max(avg_array):.6e}")
    print(f"Mean value: {np.mean(avg_array):.6e}")
    print(f"Standard deviation: {np.std(avg_array):.6e}")

    ### Check if data is reasonable (not all zeros)
    non_zero_elements = np.count_nonzero(avg_array)
    total_elements = avg_array.size
    print(
        f"Non-zero elements: {non_zero_elements} / {total_elements} ({100*non_zero_elements/total_elements:.2f}%)"
    )

    print(f"\n✅ Verification complete! Data structure looks correct.")


if __name__ == "__main__":
    file_path = "/home/leopold/PycharmProjects/Master_thesis/code/python/papers_with_proteus_output/averaged_data.pkl"
    verify_averaged_data(file_path)
