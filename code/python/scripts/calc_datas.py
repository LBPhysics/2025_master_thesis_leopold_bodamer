"""Flexible 1D / simple multi-``t_coh`` electronic spectroscopy runner (no inhomogeneity).

Two execution modes sharing the same underlying simulation object:
    1d mode:
        Processes a single coherence time (``t_coh`` from the config) and saves one file.

    2d mode:
        Treats every detection time value ``t_det`` as a coherence time point ``t_coh``
        and computes one 1D trace per point. Each processed ``t_coh`` produces an
        individual file which can later be stacked into a 2D dataset using ``stack_1dto2d.py``.

The resulting files are stored via ``save_simulation_data`` and contain
metadata keys required by downstream stacking & plotting scripts:
    - signal_types
    - t_coh_value

Examples:
    python calc_datas.py --simulation_type 1d
    python calc_datas.py --simulation_type 2d
    # 2D in batches (N batches, pick batch i)
    python calc_datas.py --simulation_type 2d --n_batches 8 --batch_idx 0
"""

from __future__ import annotations

import argparse
import time
import warnings
import numpy as np

from project_config.paths import SCRIPTS_DIR
from qspectro2d.spectroscopy.one_d_field import parallel_compute_1d_e_comps
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
# Helper function
# ---------------------------------------------------------------------------
def _compute_e_components_for_tcoh(
    sim_oqs: SimulationModuleOQS, t_coh_val: float, *, time_cut: float
) -> list[np.ndarray]:
    """Compute polarization/field components for a single coherence time.

    This mutates the simulation object to set the current ``t_coh`` and
    corresponding pulse delay, then evaluates the 1D field components.
    """
    sim_cfg = sim_oqs.simulation_config
    sim_cfg.t_coh = float(t_coh_val)
    sim_oqs.reset_times_local()  # recompute local time axes
    new_delays = sim_oqs.laser.pulse_delays
    new_delays[0] = t_coh_val
    sim_oqs.laser.pulse_delays = new_delays
    E_list = parallel_compute_1d_e_comps(sim_oqs=sim_oqs, time_cut=time_cut)
    return E_list


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------
def run_1d_mode(args) -> None:
    config_path = SCRIPTS_DIR / "config.yaml"
    sim_oqs, time_cut = create_base_sim_oqs(config_path=config_path)

    t_coh_val = float(sim_oqs.simulation_config.t_coh)
    print(f"🎯 Running 1D mode with t_coh = {t_coh_val:.2f} fs (from config)")

    E_sigs = _compute_e_components_for_tcoh(sim_oqs, t_coh_val, time_cut=time_cut)

    # Persist dataset
    metadata = {
        "signal_types": sim_oqs.simulation_config.signal_types,
        "t_coh_value": float(t_coh_val),
        "time_cut": float(time_cut),
    }
    abs_data_path = save_simulation_data(sim_oqs, metadata, E_sigs, t_det=sim_oqs.t_det)

    print(f"✅ Saved 1D result for t_coh={t_coh_val:.2f} fs.")
    print(f'\n🎯 To plot run: \npython plot_datas.py --abs_path "{abs_data_path}"')


def run_2d_mode(args) -> None:
    config_path = SCRIPTS_DIR / "config.yaml"
    sim_oqs, time_cut = create_base_sim_oqs(config_path=config_path)
    sim_cfg = sim_oqs.simulation_config
    print("🎯 Running 2D mode (iterate over t_det as t_coh)")

    # Reuse detection times as coherence-axis grid
    t_coh_vals = sim_oqs.t_det
    N_total = len(t_coh_vals)

    # Determine index subset for batching
    batch_idx: int = int(getattr(args, "batch_idx", 0))
    n_batches: int = int(getattr(args, "n_batches", 1))

    if n_batches == 1:
        indices = np.arange(N_total)
        batch_note = "all"
    else:
        # Split into contiguous chunks as evenly as possible
        # Use numpy to avoid off-by-one; ensures full coverage
        chunks = np.array_split(np.arange(N_total), n_batches)
        indices = chunks[batch_idx]
        batch_note = f"batch {batch_idx+1}/{n_batches} (size={indices.size})"

    if indices.size:
        span = (float(t_coh_vals[indices[0]]), float(t_coh_vals[indices[-1]]))
        print(
            f"📦 Batching: {batch_note}; total t_coh points={N_total}; "
            f"this job covers indices [{indices[0]}..{indices[-1]}] → t_coh in [{span[0]:.3g}, {span[1]:.3g}] fs"
        )
    else:
        print(
            f"📦 Batching: {batch_note}; total t_coh points={N_total}; this job covers no indices (empty chunk)"
        )

    saved_paths: list[str] = []
    start_time = time.time()
    for t_i in indices.tolist():
        t_coh_val = float(t_coh_vals[t_i])
        print(f"\n--- t_coh={t_coh_val:.2f} fs  [{t_i} / {N_total}]---")
        E_sigs = _compute_e_components_for_tcoh(sim_oqs, t_coh_val, time_cut=time_cut)

        sim_cfg.t_coh = t_coh_val
        metadata = {
            "signal_types": sim_cfg.signal_types,
            "t_coh_value": float(t_coh_val),
            "time_cut": float(time_cut),
        }
        out_path = save_simulation_data(sim_oqs, metadata, E_sigs, t_det=sim_oqs.t_det)
        saved_paths.append(str(out_path))
        print(f"    ✅ Saved {out_path}")

    elapsed = time.time() - start_time
    print(f"\n✅ Finished computing {len(saved_paths)} t_coh points in {elapsed:.2f} s")
    if saved_paths:
        if n_batches == 1:
            print("\n🎯 Next steps:")
            example = saved_paths[-1]
            print("  1. Stack per-t_coh files into a 2D dataset:")
            print(f"     python stack_1dto2d.py --abs_path '{example}' --skip_if_exists")
            print("  2. Plot a single 1D file (example):")
            print(f"     python plot_datas.py --abs_path '{example}'")
        else:
            print("\n🎯 Next step:")
            print(f"     python hpc_plot_datas.py --abs_path '{example}'")
    else:
        print("ℹ️  No files saved.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 1D or multi-t_coh (2d-style) spectroscopy simulations (no inhomogeneity).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            """Examples:\n  python calc_datas.py --simulation_type 1d\n  python calc_datas.py --simulation_type 2d\n  python calc_datas.py --simulation_type 2d --n_batches 8 --batch_idx 0"""
        ),
    )
    parser.add_argument(
        "--simulation_type",
        type=str,
        default="1d",
        choices=["1d", "2d"],
        help="Execution mode (default: 1d)",
    )
    # Batching options (used for 2d mode only)
    parser.add_argument(
        "--n_batches",
        type=int,
        default=1,
        help="Split the 2d run into N batches (default: 1, i.e., no batching)",
    )
    parser.add_argument(
        "--batch_idx",
        type=int,
        default=0,
        help="Zero-based index selecting which batch to run (0..n_batches-1)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SPECTROSCOPY SIMULATION")
    print(f"Mode: {args.simulation_type}")

    if args.simulation_type == "1d":
        run_1d_mode(args)
    else:  # 2d
        run_2d_mode(args)

    print("=" * 80)
    print("DONE")


if __name__ == "__main__":  # pragma: no cover
    main()
