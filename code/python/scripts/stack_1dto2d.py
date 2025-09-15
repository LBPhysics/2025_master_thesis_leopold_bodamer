from typing import cast, TYPE_CHECKING
from qspectro2d.utils import (
    save_simulation_data,
    list_available_data_files,
    load_info_file,
)
from pathlib import Path
import argparse
import numpy as np
import sys

if TYPE_CHECKING:
    # Imported only for static type checking / IDE autocomplete
    from qspectro2d.core.simulation import SimulationConfig


def main():
    parser = argparse.ArgumentParser(description="Stack 1D data into 2D along t_coh.")
    parser.add_argument(
        "--abs_path",
        type=str,
        default="1d_spectroscopy",
        help="Base directory containing 1D data files (absolute path)",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip stacking if a 2D file already exists.",
    )
    args = parser.parse_args()
    abs_path = args.abs_path

    print("\n🔍 Scanning available files:")
    print(f"   Base directory: {abs_path}")
    base_dir = Path(abs_path)

    # Optional early-exit: detect already stacked 2D file in corresponding 2d_spectroscopy tree
    if args.skip_if_exists:
        # Map 1d -> 2d directory and check there first
        parts = list(base_dir.parts)
        if "1d_spectroscopy" in parts:
            idx = parts.index("1d_spectroscopy")
            parts[idx] = "2d_spectroscopy"
            mapped_2d_dir = Path(*parts)
            if mapped_2d_dir.exists():
                candidate_files = sorted(mapped_2d_dir.glob("*_data.npz"))
                for f in candidate_files:
                    try:
                        with np.load(f, mmap_mode="r") as npz:
                            if "t_coh" in npz.files:
                                print(
                                    f"✅ Detected existing stacked 2D file in mapped dir: {f}\nSkipping stacking."
                                )
                                print(
                                    f"🎯 To plot run: python plot_datas.py --abs_path '{str(f)[:-9]}'"
                                )
                                return
                    except Exception:
                        continue
        # Local directory check (backward compatible)
        for f in sorted(base_dir.glob("*_data.npz")):
            try:
                with np.load(f, mmap_mode="r") as npz:
                    if "t_coh" in npz.files:
                        print(f"✅ Detected existing stacked 2D file: {f}\nSkipping stacking.")
                        print(f"🎯 To plot run: python plot_datas.py --abs_path '{str(f)[:-9]}'")
                        return
            except Exception:
                continue

    # Filter to only include _data.npz files
    abs_paths = list_available_data_files(base_dir)
    data_files = [path for path in abs_paths if path.endswith("_data.npz")]
    abs_data_paths = list(set(data_files))

    if not abs_data_paths:
        print("❌ No valid data files found.")
        sys.exit(1)

    print(f"\n📥 Loading {len(abs_data_paths)} data files (memory-efficient)...\n")

    results = []
    shapes = []
    signal_types = None

    for path in abs_data_paths:
        try:
            abs_data_path = Path(path)
            data_npz = np.load(abs_data_path, mmap_mode="r")
            # If any file already has t_coh axis, treat as already stacked and abort stacking logic
            if args.skip_if_exists and "t_coh" in data_npz.files:
                print(f"✅ Found already stacked 2D file: {abs_data_path}. Aborting stacking.")
                print(
                    f'🎯 To plot this data, run:\npython plot_datas.py --abs_path "{str(abs_data_path)[:-9]}"'
                )
                return
            signal_types = data_npz["signal_types"]
            datas: list[np.ndarray] = []
            for sig_type in signal_types:
                if sig_type in data_npz.files:
                    datas.append(data_npz[sig_type])
            # Extract t_coh value from filename (remove _data.npz first)
            path_without_suffix = str(path).replace("_data.npz", "")
            t_coh_str = path_without_suffix.split("t_coh_")[1]
            t_coh_val = t_coh_str.split("_")[0]
            t_coh = float(t_coh_val)

            results.append((t_coh, datas))
            shapes.append(datas[0].shape)

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

    shape_single = results[0][1][0].shape  # Shape of first data array from first signal type
    dtype = results[0][1][0].dtype  # Dtype of first data array
    num_t_coh = len(results)
    num_signal_types = len(signal_types)

    # Create a list of stacked arrays, one for each signal type
    stacked_data = [
        np.empty((num_t_coh, *shape_single), dtype=dtype) for _ in range(num_signal_types)
    ]
    t_coh_vals = np.empty(num_t_coh)

    for i, (t_coh, datas) in enumerate(results):
        # Stack each signal type separately
        for j, data_array in enumerate(datas):
            stacked_data[j][i] = data_array
        t_coh_vals[i] = t_coh

    # Load metadata once from the first file
    abs_info_path = [Path(path) for path in abs_paths if path.endswith("_info.pkl")]
    loaded_info_data = load_info_file(abs_info_path[0])
    system = loaded_info_data["system"]
    bath = loaded_info_data.get("bath") or loaded_info_data.get("bath_params")
    laser = loaded_info_data["laser"]
    sim_config: SimulationConfig = loaded_info_data.get("sim_config")

    # Get time axis (assumes same for all)
    t_det_vals = data_npz["t_det"]  # new required key
    sim_config.simulation_type = "2d"
    sim_config.t_coh = 0.0  # indicates varied

    # stacked_data shape: (num_t_coh, len(t_det_vals)) => axes: t_coh (axis0), t_det (axis1)
    abs_path = save_simulation_data(
        system=system,
        sim_config=sim_config,
        bath=bath,
        laser=laser,
        datas=stacked_data,  # Pass as list
        t_det=t_det_vals,
        t_coh=t_coh_vals,
    )
    print("✅ Stacking completed.")
    print(f"\n🎯 To plot this data, run:")
    print(f'python plot_datas.py --abs_path "{abs_path}"')


if __name__ == "__main__":
    main()
