"""Stack multiple 1D simulation results (varying t_coh) into a single 2D file.

Workflow assumptions:
    - Each input ``*_data.npz`` corresponds to one coherence time value and
        contains metadata including ``t_coh_value`` and ``signal_types`` plus the
        time detection axis ``t_det``.
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
from typing import TYPE_CHECKING

import numpy as np

from qspectro2d.utils import (
    list_available_files,
    load_info_file,
    save_simulation_data,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qspectro2d.core.simulation import SimulationConfig, SimulationModuleOQS


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

    # Weighted merge of partial files with same t_coh_value
    # ----------------------------------------------------------------------------------
    groups: dict[float, list[dict]] = {}
    signal_types: list[str] | None = None
    t_det_axis: np.ndarray | None = None
    shape_reference: tuple[int, ...] | None = None

    for fp in abs_data_paths:
        try:
            with np.load(fp, mmap_mode="r") as npz:  # type: ignore[arg-type]
                metadata = npz.get("metadata", None)
                if metadata is None:
                    raise ValueError("metadata missing")
                sig_types = list(metadata["signal_types"])  # type: ignore[index]
                if signal_types is None:
                    signal_types = sig_types
                elif signal_types != sig_types:
                    raise ValueError("Inconsistent signal_types across files")
                t_coh_val = float(metadata["t_coh_value"])  # type: ignore[index]
                n_inhom_batch = int(metadata.get("n_inhomogen_in_batch", 0))
                arrays = []
                for s in signal_types:
                    if s not in npz:
                        raise ValueError(f"Signal '{s}' missing in {fp}")
                    arr = np.array(npz[s])
                    arrays.append(arr)
                    if shape_reference is None:
                        shape_reference = arr.shape
                    elif shape_reference != arr.shape:
                        raise ValueError(
                            f"Shape mismatch for t_coh={t_coh_val}: {arr.shape} vs {shape_reference}"
                        )
                if t_det_axis is None:
                    t_det_axis = np.array(npz["t_det"])  # type: ignore[index]
                groups.setdefault(t_coh_val, []).append(
                    {"arrays": arrays, "weight": float(max(1, n_inhom_batch)), "path": fp}
                )
                print(f"   ✅ Loaded t_coh={t_coh_val:.4g} (weight {n_inhom_batch}) from {fp}")
        except Exception as e:
            print(f"   ❌ Skipping {fp}: {e}")

    if not groups or signal_types is None or t_det_axis is None:
        print("❌ No valid inputs to stack.")
        sys.exit(1)

    # Merge: for each t_coh take weighted average over partial contributions
    merged: list[tuple[float, list[np.ndarray], float]] = []
    per_t_provenance: list[dict] = []
    for t_coh_val, entries in sorted(groups.items(), key=lambda kv: kv[0]):
        total_w = sum(e["weight"] for e in entries)
        n_signals = len(signal_types)
        acc = [
            np.zeros(shape_reference, dtype=entries[0]["arrays"][i].dtype) for i in range(n_signals)
        ]
        for e in entries:
            w = e["weight"]
            for i, arr in enumerate(e["arrays"]):
                acc[i] += w * arr
        merged_arrays = [a / total_w for a in acc]
        merged.append((t_coh_val, merged_arrays, total_w))
        per_t_provenance.append(
            {
                "t_coh_value": t_coh_val,
                "total_weight": total_w,
                "n_partials": len(entries),
                "sources": [e["path"] for e in entries],
            }
        )
        if len(entries) > 1:
            print(
                f"   🔗 Merged {len(entries)} partial files for t_coh={t_coh_val:.4g} (total weight {total_w})"
            )

    n_t_coh = len(merged)
    dtype = merged[0][1][0].dtype
    n_signals = len(signal_types)
    stacked_data = [np.empty((n_t_coh, *shape_reference), dtype=dtype) for _ in range(n_signals)]
    t_coh_vals = np.empty(n_t_coh)
    total_weights = []
    for i, (t_coh_val, arrays, total_w) in enumerate(merged):
        t_coh_vals[i] = t_coh_val
        for j, arr in enumerate(arrays):
            stacked_data[j][i] = arr
        total_weights.append(total_w)

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
        "stacked": True,
        "n_inputs": int(n_t_coh),
        "source_base_dir": str(base_dir),
        "signal_types": signal_types,
        "t_coh_min": float(np.min(t_coh_vals)),
        "t_coh_max": float(np.max(t_coh_vals)),
        "merged_partials": True,
        "t_coh_total_weights": total_weights,
        "provenance": per_t_provenance,
    }

    sim_oqs = SimulationModuleOQS(
        system=system,
        bath=bath,
        laser=laser,
        sim_config=sim_config,
    )
    out_path = save_simulation_data(
        sim_oqs,
        stacked_data,
        t_det=t_det_axis,
        t_coh=t_coh_vals,
        metadata=metadata,
    )
    print("✅ Stacking complete.")
    print("\n🎯 To plot run:")
    print(f"python plot_datas.py --abs_path '{out_path}'")


if __name__ == "__main__":
    main()
