"""Flexible 1D / pseudo-2D (batched multiple t_coh values) electronic spectroscopy runner.

Two execution modes sharing the same underlying simulation object:
    1d mode:
        Processes a single coherence time (``t_coh`` from the config) and averages
        over an (optional) inhomogeneous frequency ensemble. Work can be split
        across several batches by partitioning the inhomogeneous samples.

    2d mode:
        Treats every detection time value ``times_det`` as a coherence time point
        and (optionally) splits the Cartesian product (t_coh, inhom sample) over
        batches. Each processed t_coh produces an individual 1D file which can
        later be stacked into a true 2D dataset using ``stack_1dto2d.py``.

The resulting files are stored via ``save_simulation_data`` and contain
metadata keys required by downstream stacking & plotting scripts:
    - signal_types
    - t_coh_value
    - stacked (False here; True after stacking)
    - n_batches / batch_idx / n_inhomogen_in_batch (provenance)

Examples:
    python calc_datas.py --simulation_type 1d
    python calc_datas.py --simulation_type 1d --n_batches 4 --batch_idx 2
    python calc_datas.py --simulation_type 2d --n_batches 10 --batch_idx 0
"""

from __future__ import annotations

import argparse
import time
import warnings
import numpy as np

from project_config.paths import SCRIPTS_DIR
from qspectro2d.spectroscopy.one_d_field import parallel_compute_1d_e_comps
from qspectro2d.spectroscopy.inhomogenity import sample_from_gaussian
from qspectro2d.utils import save_simulation_data
from qspectro2d.config.create_sim_obj import create_base_sim_oqs
from qspectro2d.core.simulation import SimulationModuleOQS

# Silence noisy but harmless warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*c_ops, e_ops, args and options will be keyword only from qutip 5\.3.*",
    module=r"qutip\.solver\.brmesolve",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _generate_freq_samples(sim_oqs: SimulationModuleOQS) -> np.ndarray:
    """Return Gaussian-distributed (or deterministic) frequency samples.

    Shape: (n_inhomogen, n_atoms). If ``n_inhomogen == 0`` returns empty array.
    """
    n_samples = int(sim_oqs.simulation_config.n_inhomogen)
    base_freqs = np.asarray(sim_oqs.system.frequencies_cm, dtype=float)
    fwhm_cm = float(sim_oqs.system.delta_inhomogen_cm)
    if n_samples <= 0:
        return np.empty((0, base_freqs.size), dtype=float)
    if np.isclose(fwhm_cm, 0.0):
        return np.tile(base_freqs, (n_samples, 1))
    samples = sample_from_gaussian(n_samples, fwhm_cm, base_freqs)
    return np.atleast_2d(samples)


def _avg_E_over_freqs_for_tcoh(
    sim_oqs: SimulationModuleOQS,
    t_coh_val: float,
    freq_samples: np.ndarray,
    freq_indices: np.ndarray,
    *,
    time_cut: float,
) -> tuple[list[np.ndarray], int]:
    """Average polarization / field components over selected inhom samples.

    Parameters
    ----------
    sim_oqs : SimulationModuleOQS
        Simulation object (mutated in-place for each frequency sample).
    t_coh_val : float
        Coherence time value to set.
    freq_samples : np.ndarray
        All available frequency samples.
    freq_indices : np.ndarray
        Indices (into freq_samples) assigned to this batch / t_coh.
    time_cut : float
        Time cut forwarded to field computation.

    Returns
    -------
    (avg_list, count)
        avg_list is a list of arrays (one per signal type) averaged over count samples.
        If no samples -> ([], 0).
    """
    sim_cfg = sim_oqs.simulation_config
    sim_cfg.t_coh = float(t_coh_val)
    t_wait = sim_cfg.t_wait
    sim_oqs.reset_times_local()  # recompute local time axes
    sim_oqs.laser.update_delays([t_coh_val, t_wait])

    if freq_indices.size == 0:
        return [], 0

    sum_components: list[np.ndarray] | None = None
    contribs = 0
    for idx in freq_indices:
        freqs_cm = freq_samples[int(idx)].tolist()
        sim_oqs.system.update_frequencies_cm(freqs_cm)
        E_list = parallel_compute_1d_e_comps(sim_oqs=sim_oqs, time_cut=time_cut)
        if sum_components is None:
            sum_components = [np.array(x, copy=True) for x in E_list]
        else:
            for acc, new_val in zip(sum_components, E_list):
                acc += new_val
        contribs += 1

    assert sum_components is not None  # contribs>0 ensures
    avg_components = [S / float(contribs) for S in sum_components]
    return avg_components, contribs


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------
def run_1d_mode(args) -> None:
    config_path = SCRIPTS_DIR / "config.yaml"
    sim_oqs, time_cut = create_base_sim_oqs(config_path=config_path)

    t_coh_val = float(sim_oqs.simulation_config.t_coh)
    print(f"🎯 Running 1D mode with t_coh = {t_coh_val:.2f} fs (from config)")

    freq_samples = _generate_freq_samples(sim_oqs)
    n_inhom = freq_samples.shape[0]
    n_batches = max(1, int(args.n_batches))
    batch_idx = int(args.batch_idx)
    if n_inhom <= 0:
        raise ValueError("No inhomogeneous samples available (n_inhomogen == 0)")
    if batch_idx < 0 or batch_idx >= n_batches:
        raise ValueError(f"batch_idx {batch_idx} out of range for n_batches {n_batches}")

    freq_chunks = np.array_split(np.arange(n_inhom), n_batches)
    freq_idx_subset = freq_chunks[batch_idx]
    avg_E, contribs = _avg_E_over_freqs_for_tcoh(
        sim_oqs, t_coh_val, freq_samples, freq_idx_subset, time_cut=time_cut
    )

    if contribs == 0:
        print("ℹ️  No inhomogeneous samples assigned to this batch; nothing to save.")
        return

    # Persist averaged dataset
    metadata = {
        "signal_types": sim_oqs.simulation_config.signal_types,
        "stacked": False,
        "n_batches": n_batches,
        "batch_idx": batch_idx,
        "n_inhomogen_in_batch": int(contribs),
        "t_idx": 0,
        "t_coh_value": float(t_coh_val),
    }
    abs_data_path = save_simulation_data(sim_oqs, metadata, avg_E, t_det=sim_oqs.times_det)

    print(
        f"✅ Saved 1D result for t_coh={t_coh_val:.2f} fs with {contribs}/{n_inhom} inhom samples."
    )
    print(f'\n🎯 To plot run: \npython plot_datas.py --abs_path "{abs_data_path}"')


def run_2d_mode(args) -> None:
    config_path = SCRIPTS_DIR / "config.yaml"
    n_batches = max(1, int(args.n_batches))
    batch_idx = int(args.batch_idx)

    sim_oqs, time_cut = create_base_sim_oqs(config_path=config_path)
    sim_cfg = sim_oqs.simulation_config

    print(f"🎯 Running 2D mode batch {batch_idx + 1}/{n_batches}")

    t_coh_vals = sim_oqs.times_det  # reuse detection times as coherence-axis grid
    freq_samples = _generate_freq_samples(sim_oqs)
    n_t = len(t_coh_vals)
    n_inhom = freq_samples.shape[0]
    if n_inhom <= 0:
        raise ValueError("No inhomogeneous samples available (n_inhomogen == 0)")

    # Cartesian workload (t_idx, freq_idx)
    all_pairs = [(it, jf) for it in range(n_t) for jf in range(n_inhom)]
    total_pairs = len(all_pairs)
    pair_chunks = np.array_split(np.arange(total_pairs), n_batches)
    if batch_idx < 0 or batch_idx >= len(pair_chunks):
        raise ValueError(f"batch_idx {batch_idx} out of range for n_batches {n_batches}")
    sel_idx = pair_chunks[batch_idx]
    selected_pairs = [all_pairs[i] for i in sel_idx]

    from collections import defaultdict

    work_by_t: dict[int, list[int]] = defaultdict(list)
    for t_i, f_j in selected_pairs:
        work_by_t[t_i].append(f_j)

    if work_by_t:
        t_indices_sorted = sorted(work_by_t.keys())
        it_min, it_max = t_indices_sorted[0], t_indices_sorted[-1]
        total_t_points = len(work_by_t)
        total_work_pairs = sum(len(v) for v in work_by_t.values())
        print(
            f"📊 Processing {total_t_points} coherence points in index range [{it_min},{it_max}] → {total_work_pairs} (t_coh,freq) pairs"
        )
        # Compact preview
        preview_items = ", ".join(f"t{ti}:{len(work_by_t[ti])}" for ti in t_indices_sorted[:6])
        if len(t_indices_sorted) > 6:
            preview_items += ", …"
        print(f"    Example load: {preview_items}")
    else:
        print("ℹ️  No workload assigned to this batch (nothing to save).")
        return

    saved_paths: list[str] = []
    start_time = time.time()
    for idx, t_i in enumerate(sorted(work_by_t.keys())):
        t_coh_val = float(t_coh_vals[t_i])
        freq_idx_subset = np.asarray(work_by_t[t_i], dtype=int)
        print(
            f"\n--- t_idx={t_i} ({idx+1}/{len(work_by_t)}) : t_coh={t_coh_val:.2f} fs with {len(freq_idx_subset)} freq samples ---"
        )
        avg_E, contribs = _avg_E_over_freqs_for_tcoh(
            sim_oqs, t_coh_val, freq_samples, freq_idx_subset, time_cut=time_cut
        )
        if contribs == 0:
            print("    ⚠️  No contributions for this t_coh (skipping).")
            continue

        sim_cfg.t_coh = t_coh_val
        metadata = {
            "signal_types": sim_cfg.signal_types,
            "stacked": False,
            "n_batches": n_batches,
            "batch_idx": batch_idx,
            "n_inhomogen_in_batch": int(contribs),
            "t_idx": int(t_i),
            "t_coh_value": float(t_coh_val),
        }
        out_path = save_simulation_data(sim_oqs, metadata, avg_E, t_det=sim_oqs.times_det)
        saved_paths.append(str(out_path))
        print(f"    ✅ Saved {out_path}")

    elapsed = time.time() - start_time
    print(f"\n✅ Batch {batch_idx + 1}/{n_batches} finished in {elapsed:.2f} s")
    if saved_paths:
        print("\n🎯 Next: stack into 2D (after all batches finished) with e.g.:")
        example = saved_paths[-1]
        print(f"python stack_1dto2d.py --abs_path '{example}' --skip_if_exists")
    else:
        print("ℹ️  No files saved in this batch.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 1D or batched multi-t_coh (2d-style) spectroscopy simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:\n  python calc_datas.py --simulation_type 1d\n  python calc_datas.py --simulation_type 1d --n_batches 4 --batch_idx 1\n  python calc_datas.py --simulation_type 2d --n_batches 8 --batch_idx 0""",
    )
    parser.add_argument(
        "--simulation_type",
        type=str,
        default="1d",
        choices=["1d", "2d"],
        help="Execution mode (default: 1d)",
    )
    parser.add_argument("--batch_idx", type=int, default=0, help="Batch index (0 .. n_batches-1)")
    parser.add_argument("--n_batches", type=int, default=1, help="Total number of batches (>=1)")
    args = parser.parse_args()

    print("=" * 80)
    print("SPECTROSCOPY SIMULATION")
    print(f"Mode: {args.simulation_type} | batch {args.batch_idx}/{args.n_batches}")

    if args.n_batches <= 0:
        raise ValueError("--n_batches must be positive")
    if args.batch_idx < 0 or args.batch_idx >= args.n_batches:
        raise ValueError("--batch_idx out of range")

    if args.simulation_type == "1d":
        run_1d_mode(args)
    else:  # 2d
        run_2d_mode(args)

    print("=" * 80)
    print("DONE")


if __name__ == "__main__":  # pragma: no cover
    main()
