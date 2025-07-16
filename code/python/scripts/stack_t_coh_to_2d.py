from platform import system
from qspectro2d.utils import (
    save_simulation_data,
    load_data_from_rel_path,
    list_available_data_files,
    load_info_file,
)
from qspectro2d.config import DATA_DIR
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
        "--rel_path",
        type=str,
        default="1d_spectroscopy",
        help="Base directory containing 1D data files (relative to data root)",
    )
    args = parser.parse_args()
    rel_path = args.rel_path

    print("\n🔍 Scanning available files:")
    print(f"   Base directory: {rel_path}")
    print(f"   Full path: {DATA_DIR / rel_path}")
    files_info = list_available_data_files(Path(DATA_DIR / rel_path))

    # Collect unique rel_paths
    rel_paths = list(
        {
            str(Path(p).with_suffix("").with_name(Path(p).stem[:-5]))
            for p in files_info.keys()
        }
    )

    if not rel_paths:
        print("❌ No valid data files found.")
        sys.exit(1)

    print(f"\n📥 Loading {len(rel_paths)} files (memory-efficient)...\n")

    results = []
    shapes = []

    for path in rel_paths:
        try:
            abs_data_path = DATA_DIR / (str(path) + "_data.npz")
            data_npz = np.load(abs_data_path, mmap_mode="r")
            data = data_npz["data"]

            # Extract t_coh value from filename
            t_coh_str = str(path).split("t_coh_")[1]
            t_coh_val = t_coh_str.split("_")[0]
            t_coh = float(t_coh_val)

            results.append((t_coh, data, path))  # also keep path
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

    for i, (t_coh, data, _) in enumerate(results):
        stacked_data[i] = data
        t_coh_vals[i] = t_coh

    # Load metadata once from the first file
    abs_info_path = DATA_DIR / (str(results[0][2]) + "_info.pkl")
    info_dict = load_info_file(abs_info_path)
    system = info_dict["system"]
    # workaround
    from qspectro2d.core.bath_system import BathSystem
    from qspectro2d.core.laser_system import LaserPulseSequence

    bath = BathSystem()  # info_dict["bath"]
    laser = LaserPulseSequence()  # info_dict["laser"]
    info_config = info_dict["info_config"]

    # Get time axis (assumes same for all)
    t_det = np.load(DATA_DIR / (str(results[0][2]) + "_data.npz"), mmap_mode="r")[
        "axis1"
    ]

    # Update config
    info_config["simulation_type"] = "2d"
    info_config["t_coh"] = ""  # now spans many values

    rel_path = save_simulation_data(
        system, info_config, bath, laser, stacked_data, axis1=t_coh_vals, axis2=t_det
    )

    print(f"\n✅ Final 2D data saved to: {rel_path}")
    print(f"\n🎯 To plot this data, run:")
    print(f'python plot_datas.py --rel_path "{rel_path}"')


if __name__ == "__main__":
    main()
