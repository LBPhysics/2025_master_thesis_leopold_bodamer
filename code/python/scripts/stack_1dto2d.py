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
from typing import List, Dict, Any, Tuple

import numpy as np

from qspectro2d.utils.data_io import load_data_file


def _discover_1d_files(folder: Path) -> List[Path]:
    """Return sorted list of all *_data.npz files in the folder."""
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    return sorted(folder.glob("*_data.npz"))


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


def _save_2d(
    out_folder: Path,
    t_det: np.ndarray,
    t_coh: np.ndarray,
    signal_types: List[str],
    stacked: Dict[str, np.ndarray],
) -> Path:
    """Write a single compressed file with axes and components named by signal_types."""
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / "2d_data.npz"
    payload: Dict[str, Any] = {
        "t_det": t_det,
        "t_coh": t_coh,
        "signal_types": np.array(signal_types, dtype=object),
    }
    for s, arr in stacked.items():
        payload[s] = arr
    np.savez_compressed(out_path, **payload)
    return out_path


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

    out_dir = _derive_2d_folder(in_dir)
    out_path = _save_2d(out_dir, t_det, t_coh, signal_types, stacked)

    print(f"Saved 2D dataset: {out_path}")
    print(f"To plot the 2D data, run:")
    print(f"python plot_datas.py --abs_path {out_path}")
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
