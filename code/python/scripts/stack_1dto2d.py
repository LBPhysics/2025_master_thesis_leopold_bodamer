from typing import cast, TYPE_CHECKING
from qspectro2d.utils import (
    list_available_data_files,
    load_info_file,
    save_data_file,
    save_info_file,
    generate_unique_data_filename,
)
from pathlib import Path
import argparse
import numpy as np
import sys

if TYPE_CHECKING:
    # Imported only for static type checking / IDE autocomplete
    from qspectro2d.core.simulation import SimulationConfig


def map_1d_dir_to_2d_dir(data_dir: Path) -> Path:
    """Map a 1D data directory path to its corresponding 2D directory.

    Example:
      /.../data/1d_spectroscopy/N2/.../t_dm100.0_t_wait_0.0_dt_0.1
      -> /.../data/2d_spectroscopy/N2/.../t_dm100.0_t_wait_0.0_dt_0.1
    If pattern not found, returns the original path.
    """
    parts = list(data_dir.parts)
    try:
        idx = parts.index("1d_spectroscopy")
        parts[idx] = "2d_spectroscopy"
        return Path(*parts)
    except ValueError:
        return data_dir


def detect_existing_2d(data_dir: Path) -> str | None:
    """Return the path to a detected 2D file in the mapped 2D directory, if any.

    A 2D file is identified by the presence of a 't_coh' axis inside *_data.npz.
    Only searches inside the 2D mirror directory of the provided 1D directory.
    Returns the full path including the '_data.npz' suffix if found, else None.
    """
    target_dir = map_1d_dir_to_2d_dir(data_dir)
    if not target_dir.exists():
        return None
    for f in sorted(target_dir.glob("*_data.npz")):
        try:
            with np.load(f, mmap_mode="r") as npz:  # type: ignore
                if "t_coh" in npz.files:
                    return str(f)
        except Exception:
            continue
    return None


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
    if not base_dir.is_dir():
        base_dir = base_dir.parent

    # Optional early-exit: detect already stacked 2D file
    if args.skip_if_exists:
        existing = detect_existing_2d(base_dir)
        if existing:
            print(f"✅ Existing 2D stacked file detected: {existing} (skipping stacking)")
            print(f"🎯 To plot run: python plot_datas.py --abs_path '{existing}'")
            return
        # Fallback: check the provided directory itself for any 2D file
        for f in sorted(base_dir.glob("*_data.npz")):
            try:
                with np.load(f, mmap_mode="r") as npz:
                    if "t_coh" in npz.files:
                        print(f"✅ Existing 2D stacked file detected: {f} (skipping stacking)")
                        print(f"🎯 To plot run: python plot_datas.py --abs_path '{str(f)}'")
                        return
            except Exception:
                continue

    # Filter to only include _data.npz files and sort deterministically
    abs_paths = list_available_data_files(base_dir)
    data_files = sorted({path for path in abs_paths if path.endswith("_data.npz")})
    abs_data_paths = list(data_files)

    if not abs_data_paths:
        print("❌ No valid data files found.")
        sys.exit(1)

    print(f"\n📥 Loading {len(abs_data_paths)} data files (memory-efficient)...\n")

    results = []
    shapes = []
    signal_types = None
    t_det_vals = None

    for path in abs_data_paths:
        try:
            abs_data_path = Path(path)
            with np.load(abs_data_path, mmap_mode="r") as data_npz:
                # If any file already has t_coh axis, treat as already stacked and abort stacking logic
                if args.skip_if_exists and "t_coh" in data_npz.files:
                    print(f"✅ Found already stacked 2D file: {abs_data_path}. Aborting stacking.")
                    print(
                        f'🎯 To plot this data, run:\npython plot_datas.py --abs_path "{str(abs_data_path)[:-9]}"'
                    )
                    return
                if signal_types is None:
                    signal_types = data_npz["signal_types"]
                else:
                    other_signal_types = data_npz["signal_types"]
                    if not np.array_equal(other_signal_types, signal_types):
                        raise ValueError("Inconsistent signal_types across files; cannot stack.")
                datas: list[np.ndarray] = []
                for sig_type in signal_types:
                    if sig_type in data_npz.files:
                        datas.append(data_npz[sig_type])
                if t_det_vals is None:
                    t_det_vals = data_npz["t_det"]
                # Extract t_coh value from metadata in the datafile
                if "metadata" in data_npz.files:
                    metadata = (
                        data_npz["metadata"].item()
                        if hasattr(data_npz["metadata"], "item")
                        else data_npz["metadata"]
                    )
                    t_coh = float(metadata["t_coh_value"])
                else:
                    raise KeyError(f"Missing 'metadata' in data file: {abs_data_path}")

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

    # Load metadata once from the first data file's paired info
    first_data_path = Path(abs_data_paths[0])
    first_info_path = Path(str(first_data_path).replace("_data.npz", "_info.pkl"))
    loaded_info_data = load_info_file(first_info_path)
    system = loaded_info_data["system"]
    bath = loaded_info_data.get("bath") or loaded_info_data.get("bath_params")
    laser = loaded_info_data["laser"]
    sim_config: SimulationConfig = loaded_info_data.get("sim_config")

    # Get time axis (assumes same for all), captured from loop above
    if t_det_vals is None:
        raise RuntimeError("t_det axis not found in any input file.")
    sim_config.simulation_type = "2d"
    sim_config.t_coh = 0.0  # indicates varied

    # Save data and info separately with metadata
    # stacked_data shape: (num_t_coh, len(t_det_vals)) => axes: t_coh (axis0), t_det (axis1)
    abs_base_path = generate_unique_data_filename(system, sim_config)
    abs_data_path = Path(f"{abs_base_path}_data.npz")

    metadata = {
        "stacked": True,
        "n_inputs": int(num_t_coh),
        "source_base_dir": str(base_dir),
        "t_coh_min": float(np.min(t_coh_vals)),
        "t_coh_max": float(np.max(t_coh_vals)),
    }

    # Write compressed data file
    save_data_file(
        abs_data_path=abs_data_path,
        datas=stacked_data,
        t_det=t_det_vals,
        t_coh=t_coh_vals,
        signal_types=list(signal_types),
        metadata=metadata,
    )

    # Write paired metadata/info file
    abs_info_path = Path(f"{abs_base_path}_info.pkl")
    save_info_file(
        abs_info_path=abs_info_path,
        system=system,
        bath=bath,
        laser=laser,
        sim_config=sim_config,
    )
    print("✅ Stacking completed.")
    print(f"\n🎯 To plot this data, run:")
    print(f'python plot_datas.py --abs_path "{abs_data_path}"')


if __name__ == "__main__":
    main()
