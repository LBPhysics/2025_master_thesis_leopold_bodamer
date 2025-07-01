from qspectro2d.data import (
    save_simulation_data,
    load_data_from_rel_path,
    list_available_data_files,
)
from qspectro2d.config import DATA_DIR
from pathlib import Path
import numpy as np
import sys


def main():
    base_dir = "1d_spectroscopy"

    print("\n🔍 Scanning available files:")
    files_info = list_available_data_files(Path(base_dir))

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
            result = load_data_from_rel_path(path)
            tau = result["data_config"]["tau_coh"]
            results.append((tau, result))
            print(f"   ✅ Loaded: {path} (tau_coh={tau})")
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
    t_det = results[0][1]["axes"]["axs1"]
    system = results[0][1]["system"]
    data_config = results[0][1]["data_config"]

    # change the type to 2d
    data_config["simulation_type"] = "2d"
    data_config["tau_coh"] = ""

    stacked_data = np.stack(all_data, axis=0)
    tau_vals = np.array(all_tau)

    rel_path = save_simulation_data(
        system, data_config, stacked_data, axs1=tau_vals, axs2=t_det
    )
    print(f"\n✅ Final 2D data saved to: {rel_path}")


if __name__ == "__main__":
    main()
