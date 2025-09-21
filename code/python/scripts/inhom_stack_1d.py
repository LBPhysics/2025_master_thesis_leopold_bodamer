"""Stack/average inhomogeneous 1D configurations produced by calc_datas.py.

Usage:
    python inhom_stack_1d.py --abs_path "<path to one _data.npz>" [--skip_if_exists]

Given one file path from an inhomogeneous batch (same t_coh, same group id),
this script finds all sibling files in the same directory tree with matching
`inhom_group_id`, loads the individual 1D arrays per signal type, averages
them over the inhomogeneous configurations, and writes a new file containing
the averaged result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import numpy as np

from qspectro2d.utils import load_simulation_data, save_data_file


def _collect_group_files(anchor: Path) -> List[Path]:
    """Find all files with the same inhom_group_id as `anchor` in its directory.

    The anchor must be a path to a single `_data.npz` file from an inhomogeneous 1D run.
    """
    base = load_simulation_data(anchor)
    if not base.get("inhom_enabled", False):
        raise ValueError("Provided file is not marked as inhomogeneous (inhom_enabled=False).")
    group_id = base.get("inhom_group_id")
    if group_id is None:
        raise ValueError("Missing inhom_group_id in anchor file metadata.")

    dir_path = Path(anchor).parent
    all_npz = list(dir_path.glob("*_data.npz"))
    matches: List[Path] = []
    for p in all_npz:
        try:
            d = load_simulation_data(p)
        except Exception:
            continue
        if d.get("inhom_enabled", False) and d.get("inhom_group_id") == group_id:
            matches.append(p)
    if not matches:
        raise FileNotFoundError("No matching inhomogeneous files found for group.")
    return sorted(matches)


def average_inhom_1d(abs_path: Path, *, skip_if_exists: bool = False) -> Path:
    """Average all 1D arrays across inhomogeneous configs for current group.

    Returns the path to the newly written averaged file.
    """
    files = _collect_group_files(Path(abs_path))

    # Load first to get axes and metadata
    first = load_simulation_data(files[0])
    t_det = np.asarray(first["t_det"], dtype=float)
    signal_types: List[str] = list(map(str, first["signal_types"]))

    # Collect arrays per type
    stacks: dict[str, List[np.ndarray]] = {k: [] for k in signal_types}
    for f in files:
        d = load_simulation_data(f)
        # sanity: axes must match shape
        if not np.allclose(d["t_det"], t_det):
            raise ValueError(f"Mismatched t_det axis in {f}")
        for k in signal_types:
            arr = np.asarray(d[k])
            if arr.shape != (t_det.size,):
                raise ValueError(f"Unexpected array shape for {k} in {f}: {arr.shape}")
            stacks[k].append(arr)

    # Average
    averaged: List[np.ndarray] = []
    for k in signal_types:
        data = np.stack(stacks[k], axis=0)  # (n_files, t_det)
        averaged.append(np.mean(data, axis=0))

    # Compose metadata for output
    metadata = {
        "signal_types": signal_types,
        "t_coh_value": float(first.get("t_coh_value", 0.0)),
        "time_cut": float(first.get("time_cut", 0.0)),
        "inhom_enabled": True,
        "inhom_averaged": True,
        "inhom_group_id": first.get("inhom_group_id"),
        "inhom_total": int(first.get("inhom_total", len(files))),
        "inhom_n_files": len(files),
    }

    # Write averaged to a sibling file with suffix `_inhom_avg`
    out_path = Path(str(files[0]).replace("_data.npz", "_inhom_avg_data.npz"))
    if skip_if_exists and out_path.exists():
        return out_path

    save_data_file(out_path, metadata, averaged, t_det)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Average inhomogeneous 1D configs into one file.")
    p.add_argument("--abs_path", type=str, required=True, help="Path to one *_data.npz file")
    p.add_argument(
        "--skip_if_exists", action="store_true", help="Do not overwrite existing averaged file"
    )
    args = p.parse_args()

    out = average_inhom_1d(Path(args.abs_path), skip_if_exists=args.skip_if_exists)
    print(f"✅ Wrote averaged file: {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
