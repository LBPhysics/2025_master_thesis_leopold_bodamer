"""Stack multiple 1D simulation results (varying t_coh) into a single 2D file.

Simplified workflow (no inhomogeneity, no partial merging):
        - Each input ``*_data.npz`` corresponds to one coherence time value and
            contains ``t_coh_value``, ``signal_types``, and the time detection axis ``t_det``.
        - Files reside in a directory under ``.../data/1d_spectroscopy/...``.
        - A mirror directory under ``.../data/2d_spectroscopy/...`` will host the
            stacked 2D dataset.

The produced 2D file stores axes ``t_coh`` and ``t_det`` and a list of stacked
signal arrays with shape (n_t_coh, ... original 1D shape ...).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from qspectro2d.utils import (
    list_available_files,
    load_info_file,
    save_simulation_data,
)
from qspectro2d.core.simulation import SimulationModuleOQS
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack multiple 1D spectra into a 2D dataset.")
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

    # Filter to only include _data.npz files and sort deterministically
    abs_paths = list_available_files(base_dir)
    data_files = sorted({p for p in abs_paths if p.endswith("_data.npz")})
    abs_data_paths = list(data_files)

    if not abs_data_paths:
        print("❌ No valid data files found.")
        sys.exit(1)

    print(f"\n📥 Loading {len(abs_data_paths)} data files (memory-efficient)...\n")

    # Group by t_coh_value (assume at most one file per value)
    # ----------------------------------------------------------------------------------
    groups: dict[float, dict] = {}
    signal_types: list[str] | None = None
    t_det_axis: np.ndarray | None = None
    shape_reference: tuple[int, ...] | None = None

    for fp in abs_data_paths:
        try:
            with np.load(fp, mmap_mode="r", allow_pickle=True) as npz:  # type: ignore[arg-type]
                sig_types = list(npz.get("signal_types"))  # type: ignore[index]
                if signal_types is None:
                    signal_types = sig_types
                elif signal_types != sig_types:
                    raise ValueError("Inconsistent signal_types across files")

                # Coherence time value
                t_coh_raw = npz.get("t_coh_value")  # type: ignore[index]
                if isinstance(t_coh_raw, (list, tuple)):
                    t_coh_raw = t_coh_raw[0]
                t_coh_val = float(t_coh_raw)

                # Load signal components
                sigs = []
                for s in signal_types:
                    if s not in npz:
                        raise ValueError(f"Signal '{s}' missing in {fp}")
                    arr = np.array(npz[s])
                    sigs.append(arr)
                    if shape_reference is None:
                        shape_reference = arr.shape
                    elif shape_reference != arr.shape:
                        raise ValueError(
                            f"Shape mismatch for t_coh={t_coh_val}: {arr.shape} vs {shape_reference}"
                        )
                if t_det_axis is None:
                    t_det_axis = np.array(npz["t_det"])  # type: ignore[index]

                if t_coh_val in groups:
                    # If duplicates exist, keep the last one encountered (simple rule)
                    print(
                        f"   🔁 Duplicate t_coh={t_coh_val:.4g} found; replacing previous entry with {fp}"
                    )
                groups[t_coh_val] = {"signals": sigs, "path": fp}
                print(f"   ✅ Loaded t_coh={t_coh_val:.4g} from {fp}")
        except Exception as e:
            print(f"   ❌ Skipping {fp}: {e}")

    if not groups or signal_types is None or t_det_axis is None:
        print("❌ No valid inputs to stack.")
        sys.exit(1)

    # Prepare stacked arrays in ascending t_coh order (no averaging/merging)
    sorted_items = sorted(groups.items(), key=lambda kv: kv[0])
    n_t_coh = len(sorted_items)
    dtype = sorted_items[0][1]["signals"][0].dtype
    n_signals = len(signal_types)
    stacked_data = [np.empty((n_t_coh, *shape_reference), dtype=dtype) for _ in range(n_signals)]
    t_coh_vals = np.empty(n_t_coh)
    sources: list[str] = []
    for i, (t_coh_val, entry) in enumerate(sorted_items):  # i is the index along the t_coh axis
        t_coh_vals[i] = t_coh_val
        arrays = entry["signals"]
        for j, arr in enumerate(arrays):
            stacked_data[j][i] = arr  # holds signal type j at coherence index i.
        sources.append(entry["path"])

    # Load simulation structural info from one accompanying _info.pkl file
    first_info = Path(str(abs_data_paths[0]).replace("_data.npz", "_info.pkl"))
    info_payload = load_info_file(first_info)
    system = info_payload["system"]
    bath = info_payload.get("bath") or info_payload.get("bath_params")
    laser = info_payload["laser"]
    sim_config: SimulationConfig = info_payload["sim_config"]
    sim_config.simulation_type = "2d"
    sim_config.t_coh = None

    metadata = {
        "signal_types": signal_types,
        "n_inputs": int(n_t_coh),
        "source_base_dir": str(base_dir),
        "t_coh_min": float(np.min(t_coh_vals)),
        "t_coh_max": float(np.max(t_coh_vals)),
        "merged_partials": False,
        "provenance": {
            "sources": sources,
            "note": "Simplified stacking (no averaging). One file per t_coh.",
        },
    }

    sim_oqs = SimulationModuleOQS(
        simulation_config=sim_config,
        system=system,
        bath=bath,
        laser=laser,
    )
    out_path = save_simulation_data(
        sim_oqs,
        metadata,
        stacked_data,
        t_det=t_det_axis,
        t_coh=t_coh_vals,
    )
    print("✅ Stacking complete.")
    print("\n🎯 To plot run:")
    print(f"python plot_datas.py --abs_path '{out_path}'")


if __name__ == "__main__":
    main()
