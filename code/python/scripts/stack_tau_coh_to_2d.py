from platform import system
from qspectro2d.data import (
    save_simulation_data,
    load_data_from_rel_path,
    list_available_data_files,
    load_info_file,
    load_data_file
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
    parser = argparse.ArgumentParser(description="Stack 1D data into 2D along tau_coh.")
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

    # Collect rel_paths from info keys (strip _data.npz for rel_path compatibility)
    rel_paths = list({ # set to avoid duplicates
        str(Path(p).with_suffix("").with_name(Path(p).stem[:-5]))
        for p in files_info.keys()
    })

    if not rel_paths:
        print("❌ No valid data files found.")
        sys.exit(1)

    results = []

    print(f"\n📥 Loading {len(rel_paths)} files...\n")
    for path in rel_paths:
        try:
            # only on
            abs_data_path = DATA_DIR / (str(path) + "_data.npz")
            data_dict = load_data_file(abs_data_path)
            # Extract tau_coh value from filename (expects ...tau_<val>..._data.npz)
            tau_str = str(path).split("tau_")[1]
            tau_val = tau_str.split("_")[0]
            tau     = float(tau_val)

            results.append((tau, data_dict))
            print(f"   ✅ Loaded: (tau_coh={tau})")
        except Exception as e:
            print(f"   ❌ Failed to load {path}: {e}")

    if not results:
        print("❌ No valid data loaded — cannot stack. Aborting.")
        sys.exit(1)

    # Sort by tau_coh
    results.sort(key=lambda r: r[0])

    # Extract data
    all_data = [r[1]["data"] for r in results]
    all_tau = [r[0] for r in results]
    print(results[0][1])
    t_det = results[0][1]["axis1"]

    stacked_data = np.stack(all_data, axis=0)
    tau_vals = np.array(all_tau)

    abs_info_path = DATA_DIR / (str(path) + "_info.pkl") # 
    info_dict = load_info_file(abs_info_path)
    system = info_dict["system"]
    info_config = info_dict["info_config"]

    # change the type to 2d
    print(info_config)
    info_config["simulation_type"] = "2d"
    info_config["tau_coh"] = ""
    
    rel_path = save_simulation_data(
        system, info_config, stacked_data, axis1=tau_vals, axis2=t_det
    )
    print(f"\n✅ Final 2D data saved to: {rel_path}")
    print(f"\n🎯 To plot this data, run:")
    print(f'python plot_datas.py --rel_path "{rel_path}"')


if __name__ == "__main__":
    main()
