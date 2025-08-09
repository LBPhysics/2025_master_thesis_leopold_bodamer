from platform import system
from qspectro2d.utils import (
    save_simulation_data,
    list_available_data_files,
    load_info_file,
)
from pathlib import Path
import numpy as np
import sys


def main():
    # =============================
    # Set base directory as a parameter
    # =============================
    import argparse

    parser = argparse.ArgumentParser(description="Stack 1D data into 2D along t_coh.")
    parser.add_argument(
        "--abs_path",
        type=str,
        default="1d_spectroscopy",
        help="Base directory containing 1D data files (absolute path)",
    )
    args = parser.parse_args()
    abs_path = args.abs_path

    print("\n🔍 Scanning available files:")
    print(f"   Base directory: {abs_path}")
    ### Filter to only include _data.npz files
    abs_paths = list_available_data_files(Path(abs_path))
    data_files = [path for path in abs_paths if path.endswith("_data.npz")]
    abs_data_paths = list(set(data_files))

    if not abs_data_paths:
        print("❌ No valid data files found.")
        sys.exit(1)

    print(f"\n📥 Loading {len(abs_data_paths)} data files (memory-efficient)...\n")

    results = []
    shapes = []

    for path in abs_data_paths:
        try:
            abs_data_path = Path(path)
            data_npz = np.load(abs_data_path, mmap_mode="r")
            data = data_npz["data"]

            ### Extract t_coh value from filename (remove _data.npz first)
            path_without_suffix = str(path).replace("_data.npz", "")
            t_coh_str = path_without_suffix.split("t_coh_")[1]
            t_coh_val = t_coh_str.split("_")[0]
            t_coh = float(t_coh_val)

            results.append((t_coh, data))
            shapes.append(data.shape)

            print(f"   ✅ Loaded: t_coh = {t_coh}")
        except Exception as e:
            print(f"   ❌ Failed to load {path}: {e}")

    if not results:
        print("❌ No valid data loaded — cannot stack. Aborting.")
        sys.exit(1)

    if len(set(shapes)) > 1:
        print("❌ Inconsistent data shapes — cannot safely stack.")
        for s in set(shapes):
            print(f"   Detected shape: {s}")
        sys.exit(1)

    # Sort by t_coh
    results.sort(key=lambda r: r[0])

    shape_single = results[0][1].shape
    dtype = results[0][1].dtype
    num_t_coh = len(results)

    stacked_data = np.empty((num_t_coh, *shape_single), dtype=dtype)
    t_coh_vals = np.empty(num_t_coh)

    for i, (t_coh, data) in enumerate(results):
        stacked_data[i] = data
        t_coh_vals[i] = t_coh

    # Load metadata once from the first file
    abs_info_path = [Path(path) for path in abs_paths if path.endswith("_info.pkl")]
    loaded_info_data = load_info_file(abs_info_path[0])
    system = loaded_info_data["system"]
    bath = loaded_info_data["bath_params"]
    laser = loaded_info_data["laser"]
    info_config = loaded_info_data["info_config"]

    # Get time axis (assumes same for all)
    t_det_vals = data_npz["axis1"]
    if t_det_vals[0] != t_coh_vals[0] or t_det_vals[-1] != t_coh_vals[-1]:
        print(
            "❌ Inconsistent time axes between t_coh and t_det. "
            "Ensure all files have the same time range."
        )
        sys.exit(1)
    # Update config
    info_config["simulation_type"] = "2d"
    info_config["t_coh"] = ""  # now spans many values

    abs_path = save_simulation_data(
        system,
        info_config,
        bath,
        laser,
        stacked_data,
        axis1=t_coh_vals,
        axis2=t_det_vals,
    )

    print(f"\n✅ Final 2D data saved to: {abs_path}")
    print(f"\n🎯 To plot this data, run:")
    print(f'python plot_datas.py --abs_path "{abs_path}"')


if __name__ == "__main__":
    main()
