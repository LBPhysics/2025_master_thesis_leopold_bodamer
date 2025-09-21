"""Stack many 1D per-t_coh results into a single 2D dataset.

Usage:
  python stack_1dto2d.py --abs_path \
    "/home/<user>/Master_thesis/data/1d_spectroscopy/.../t_dm..._t_wait..._dt.../"

Behavior:
- Discovers all "*_data.npz" files in the given folder.
- Loads each file, reads its "t_coh_value", "t_det", and arrays named by "signal_types".
- Sorts by t_coh_value, stacks arrays into 2D: shape (n_tcoh, n_tdet).
- Writes output into the corresponding 2D directory by replacing
  "data/1d_spectroscopy" with "data/2d_spectroscopy" and saving "2d_data.npz".

Keep it simple and readable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import copy
from typing import List, Dict, Any, Tuple

import numpy as np

from qspectro2d.utils.data_io import (
    load_data_file,
    load_info_file,
    save_simulation_data,
)


def _discover_1d_files(folder: Path) -> List[Path]:
    """Return sorted list of all *_data.npz files in the folder.

    If any averaged files ("*_inhom_avg_data.npz") exist, prefer only those.
    This avoids stacking raw per-config files with duplicate t_coh values.
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    candidates = sorted(folder.glob("*_data.npz"))
    avgs = [p for p in candidates if str(p).endswith("_inhom_avg_data.npz")]
    return avgs if avgs else candidates


def _derive_2d_folder(from_1d_folder: Path) -> Path:
    """Map 1D folder .../data/1d_spectroscopy/... -> .../data/2d_spectroscopy/..."""
    parts = list(from_1d_folder.parts)
    try:
        idx = parts.index("1d_spectroscopy")
    except ValueError as exc:
        raise ValueError("The provided path must include '1d_spectroscopy'") from exc
    parts[idx] = "2d_spectroscopy"
    return Path(*parts)


def _load_entries(
    files: List[Path],
) -> Tuple[List[float], np.ndarray, List[str], Dict[str, List[np.ndarray]]]:
    """Load all files and organize data by signal type.

    Returns:
        tcoh_vals: list of t_coh_value (floats)
        t_det: detection time axis (from first file; validated against others)
        signal_types: list of signal keys
        per_sig_data: mapping signal_name -> list of 1D arrays (ordered like files)
    """
    if not files:
        raise FileNotFoundError("No *_data.npz files found in the given folder")

    tcoh_vals: List[float] = []
    t_det: np.ndarray | None = None
    signal_types: List[str] | None = None
    per_sig_data: Dict[str, List[np.ndarray]] = {}

    for fp in files:
        d = load_data_file(fp)
        if "t_coh_value" not in d:
            raise KeyError(f"Missing 't_coh_value' in {fp}")
        if "t_det" not in d:
            raise KeyError(f"Missing 't_det' in {fp}")
        if "signal_types" not in d:
            raise KeyError(f"Missing 'signal_types' in {fp}")

        tcoh_vals.append(float(d["t_coh_value"]))
        if t_det is None:
            t_det = d["t_det"]
        else:
            if d["t_det"].shape != t_det.shape or not np.allclose(d["t_det"], t_det):
                raise ValueError(f"Inconsistent t_det across files; first={files[0]}, bad={fp}")

        stypes = list(map(str, d["signal_types"]))
        if signal_types is None:
            signal_types = stypes
            for s in signal_types:
                per_sig_data[s] = []
        else:
            if stypes != signal_types:
                raise ValueError(
                    f"Inconsistent signal_types across files; first={files[0]}, bad={fp}"
                )

        for s in signal_types:
            if s not in d:
                raise KeyError(f"Missing data for signal '{s}' in {fp}")
            arr = d[s]
            if arr.ndim != 1:
                raise ValueError(
                    f"Expected 1D array for signal '{s}' in {fp}, got shape {arr.shape}"
                )
            per_sig_data[s].append(arr)

    assert t_det is not None and signal_types is not None

    # Guard against duplicate coherence values when averaging was not used
    unique_count = len(set(map(lambda x: float(x), tcoh_vals)))
    if unique_count != len(tcoh_vals):
        raise ValueError(
            "Found duplicate t_coh_value entries. If this directory contains raw inhomogeneous "
            "per-config files, run inhom_stack_1d.py first or point this script to the folder "
            "containing only averaged files (_inhom_avg_data.npz)."
        )

    return tcoh_vals, t_det, signal_types, per_sig_data


def _stack_to_2d(
    tcoh_vals: List[float],
    t_det: np.ndarray,
    signal_types: List[str],
    per_sig_data: Dict[str, List[np.ndarray]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Sort by t_coh and stack into 2D arrays per signal.

    Returns:
        t_coh: sorted array of t_coh values
        stacked: mapping signal -> 2D array with shape (n_tcoh, n_tdet)
    """
    order = np.argsort(np.asarray(tcoh_vals))
    t_coh = np.asarray(tcoh_vals, dtype=float)[order]
    stacked: Dict[str, np.ndarray] = {}
    for s in signal_types:
        mat = np.vstack([per_sig_data[s][i] for i in order])
        stacked[s] = mat
    return t_coh, stacked


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack 1D per-t_coh outputs into a 2D dataset.")
    parser.add_argument(
        "--abs_path", type=str, required=True, help="Absolute path to the 1D results directory"
    )
    args = parser.parse_args()

    sanitized = args.abs_path.strip().strip('"').strip("'").replace("\r", "").replace("\n", "")
    in_dir = Path(sanitized).expanduser().resolve()
    print("=" * 80)
    print("STACK 1D -> 2D")
    print(f"Input directory: {in_dir}")

    files = _discover_1d_files(in_dir)
    print(f"Found {len(files)} files to stack")
    if not files:
        print("No files found; aborting.")
        sys.exit(1)

    tcoh_vals, t_det, signal_types, per_sig_data = _load_entries(files)
    t_coh, stacked = _stack_to_2d(tcoh_vals, t_det, signal_types, per_sig_data)

    # --- Contribution analysis & reporting ---
    # Build per-row non-zero masks from stacked data:
    # mask_any: row has any non-zero across any signal
    # mask_all: row has non-zero in every signal (i.e., no all-zero signal at that t_coh)
    # mask_all_zero: row is all-zero across all signals
    n_rows = len(t_coh)
    if n_rows > 0:
        per_signal_row_any = []
        for s in signal_types:
            arr2d = stacked[s]
            # Ensure correct shape
            if arr2d.ndim != 2 or arr2d.shape[0] != n_rows:
                raise ValueError(
                    f"Stacked array for signal '{s}' has unexpected shape {arr2d.shape}"
                )
            per_signal_row_any.append(np.any(arr2d != 0, axis=1))

        # Combine across signals
        per_signal_row_any = np.stack(per_signal_row_any, axis=0)  # (n_signals, n_rows)
        mask_any = np.any(per_signal_row_any, axis=0)
        mask_all = np.all(per_signal_row_any, axis=0)
        mask_all_zero = ~mask_any

        # Pretty-print helpers
        def fmt_vals(vals: np.ndarray) -> str:
            return ", ".join(f"{v:.3f}" for v in vals)

        idx_all = np.arange(n_rows)
        print("-" * 80)
        print(f"Processed coherence values (n={n_rows}):")
        print(f"  indices: [{', '.join(map(str, idx_all))}]")
        print(f"  t_coh(fs): [{fmt_vals(t_coh)}]")

        # Rows with any non-zero contribution
        any_idxs = idx_all[mask_any]
        any_vals = t_coh[mask_any]
        print(f"Rows with any non-zero contribution (n={any_idxs.size}):")
        print(f"  indices: [{', '.join(map(str, any_idxs))}]")
        print(f"  t_coh(fs): [{fmt_vals(any_vals)}]")

        # Rows entirely zero across all signals
        zero_idxs = idx_all[mask_all_zero]
        zero_vals = t_coh[mask_all_zero]
        print(f"Rows all-zero across signals (n={zero_idxs.size}):")
        if zero_idxs.size:
            print(f"  indices: [{', '.join(map(str, zero_idxs))}]")
            print(f"  t_coh(fs): [{fmt_vals(zero_vals)}]")
        else:
            print("  (none)")

        # Rows with only non-zero contributions in every signal
        all_idxs = idx_all[mask_all]
        all_vals = t_coh[mask_all]
        print(f"Rows with only non-zero contributions in every signal (n={all_idxs.size}):")
        if all_idxs.size:
            print(f"  indices: [{', '.join(map(str, all_idxs))}]")
            print(f"  t_coh(fs): [{fmt_vals(all_vals)}]")
        else:
            print("  (none)")

    # Load the 1D bundle info once and re-use it for saving via save_simulation_data
    first_data = files[0]
    if str(first_data).endswith("_data.npz"):
        first_info = Path(str(first_data)[:-9] + "_info.pkl")
    else:
        first_info = first_data.with_suffix(".pkl")

    info = load_info_file(first_info)
    if not info:
        print(f"❌ Could not load info from {first_info}; cannot save 2D bundle.")
        sys.exit(1)

    # Prepare a minimal sim module stub with adjusted simulation_type for 2D naming
    system = info["system"]
    bath = info["bath"]
    laser = info["laser"]
    original_cfg = info["sim_config"]
    sim_cfg_2d = copy.deepcopy(original_cfg)
    # Ensure the directory naming routes to 2D location
    if hasattr(sim_cfg_2d, "simulation_type"):
        sim_cfg_2d.simulation_type = "2d"

    from qspectro2d.core import SimulationModuleOQS  # avoid circular import

    sim_2d = SimulationModuleOQS(sim_cfg_2d, system, laser, bath)
    metadata: Dict[str, Any] = {
        "t_coh_averaged": True,
        "inhom_averaged": False,
        "signal_types": list(signal_types),
    }
    datas: List[np.ndarray] = [stacked[s] for s in signal_types]

    out_path = save_simulation_data(
        sim_module=sim_2d,
        metadata=metadata,
        datas=datas,
        t_det=t_det,
        t_coh=t_coh,
    )

    print(f"Saved 2D dataset: {out_path}")
    print(f"To plot the 2D data, run:")
    print(f"python plot_datas.py --abs_path {out_path}")
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
