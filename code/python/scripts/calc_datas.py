"""
1D Electronic Spectroscopy Simulation Script – Flexible Execution Mode

This script runs 1D spectroscopy datas for a given set of simulation parameters.
It supports two modes of execution:

# Determine whether to use batch mode or single t_coh mode based on the provided arguments.
    --simulation_type <type>: Type of simulation (default: "1d")
        -> "1d" Run the simulation for one specific coherence time
        -> "2d" Run the simulation for a range of coherence times, splitted into coh_batches

# other arguments:
    --coh_batches <total>: Total number of coh_batches (default: 1)
    --coh_idx <index>: Batch index for the current job (0 to coh_batches-1, default: 0)

This script is designed for both local development and HPC batch execution.
Results are saved automatically using the qspectro2d I/O framework.

# usage
python calc_datas.py --simulation_type 1d
python calc_datas.py --simulation_type 2d
"""

import time
import argparse
import warnings
import numpy as np
from pathlib import Path

from project_config.paths import SCRIPTS_DIR

from qspectro2d.spectroscopy.one_d_field import (
    parallel_compute_1d_E_with_inhomogenity,
)
from qspectro2d.utils import (
    save_data_file,
    save_info_file,
    generate_unique_data_filename,
)
from qspectro2d.config.create_sim_obj import (
    create_base_sim_oqs,
)
from qspectro2d.core.simulation import SimulationModuleOQS

# Suppress noisy but harmless warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
# Silence QuTiP FutureWarning about keyword-only args in brmesolve (qutip >=5.3)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*c_ops, e_ops, args and options will be keyword only from qutip 5\.3.*",
    module=r"qutip\.solver\.brmesolve",
)


def _select_inhom_indices(n_inhom: int, inhom_batches: int | None, inhom_idx: int | None) -> np.ndarray:
    """Compute explicit inhomogeneity sample indices for this run.

    Precedence mirrors coherence batching: if no batching provided, return all indices.
    """
    all_indices = np.arange(int(n_inhom), dtype=int)
    if inhom_batches is None:
        return all_indices
    if inhom_idx is None:
        raise ValueError("When providing inhom_batches, also provide inhom_idx")
    if inhom_batches <= 0:
        raise ValueError("inhom_batches must be positive")
    if not (0 <= int(inhom_idx) < int(inhom_batches)):
        raise ValueError("inhom_idx must satisfy 0 <= idx < inhom_batches")
    parts = np.array_split(all_indices, int(inhom_batches))
    return parts[int(inhom_idx)]


def run_single_t_coh_with_sim(
    sim_oqs: SimulationModuleOQS,
    t_coh: float,
    save_info: bool = False,
    time_cut: float = -np.inf,
    inhom_indices: np.ndarray | None = None,
    inhom_batches: int | None = None,
    inhom_idx: int | None = None,
    average_over_inhom: bool = True,
    coh_batches_meta: int = 1,
    coh_idx_meta: int = 0,
) -> Path:
    """
    Run a single 1D simulation for a specific coherence time using existing SimulationModuleOQS.

    Parameters:
        sim_oqs (SimulationModuleOQS): Pre-configured simulation instance
        t_coh (float): Coherence time between 2 pump pulses [fs]
        save_info (bool): Whether to save simulation info
        time_cut (float): Time cutoff for solver validation

    Returns:
        Path: absolute path to the saved datas directory
    """
    print(f"\n=== Starting t_coh = {t_coh:.2f} fs ===")

    # Update coherence time and recompute derived timing + pulse delays
    sim_config_obj = sim_oqs.simulation_config
    sim_config_obj.t_coh = t_coh
    t_wait = sim_config_obj.t_wait
    t_det_max = sim_config_obj.t_det_max

    # Recompute t_max (SimulationConfig does this only in __post_init__)
    sim_config_obj.t_max = t_wait + 2 * t_det_max

    # Clear cached global / local time arrays so they rebuild with new t_max
    if hasattr(sim_oqs, "_times_global"):
        delattr(sim_oqs, "_times_global")
    sim_oqs.reset_times_local()

    # update pulse delays
    sim_oqs.laser.update_delays([t_coh, t_wait])  # Note hard to extend to n pulses

    # Run the simulation
    start_time = time.time()
    print("Computing 1D polarization with parallel processing...")
    try:
        if inhom_indices is None:
            n_inhom = int(sim_oqs.simulation_config.n_inhomogen)
            inhom_indices = _select_inhom_indices(n_inhom, inhom_batches, inhom_idx)
        datas = parallel_compute_1d_E_with_inhomogenity(
            sim_oqs=sim_oqs,
            time_cut=time_cut,
            inhom_indices=inhom_indices,
            average_over_inhom=average_over_inhom,
        )
        print("✅ Parallel computation completed successfully!")
    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise

    # Save datas
    abs_path = generate_unique_data_filename(sim_oqs.system, sim_config_obj)
    abs_data_path = Path(f"{abs_path}_data.npz")

    signal_types = sim_config_obj.signal_types
    # Attach batch metadata for later combination if needed
    metadata = {
        "inhom_batches": inhom_batches if inhom_batches is not None else 1,
        "inhom_idx": inhom_idx if inhom_idx is not None else 0,
        "coh_batches": int(coh_batches_meta),
        "coh_idx": int(coh_idx_meta),
        "average_over_inhom": average_over_inhom,
    }
    save_data_file(abs_data_path, datas, sim_oqs.times_det, signal_types=signal_types, metadata=metadata)

    if save_info:
        abs_info_path = Path(f"{abs_path}_info.pkl")
        save_info_file(
            abs_info_path,
            sim_oqs.system,
            bath=sim_oqs.bath,
            laser=sim_oqs.laser,
            sim_config=sim_config_obj,
        )

        print(f"{'='*60}")
        print(f"\n🎯 To plot this datas, run:")
        print(f'python plot_datas.py --abs_path "{abs_data_path}"')

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    return abs_data_path


def run_1d_mode(args):
    """Run single 1D simulation for a specific coherence time."""
    config_path = SCRIPTS_DIR / "config.yaml"

    # Build base simulation (applies CLI overrides inside)
    sim_oqs, time_cut = create_base_sim_oqs(config_path=config_path)

    t_coh_print = sim_oqs.simulation_config.t_coh
    print(f"🎯 Running 1D mode with t_coh = {t_coh_print:.2f} fs (from config)")

    # Centralize inhom indices selection for 1D as well
    n_inhom = int(sim_oqs.simulation_config.n_inhomogen)
    inhom_indices = _select_inhom_indices(n_inhom, args.inhom_batches, args.inhom_idx)

    run_single_t_coh_with_sim(
        sim_oqs,
        t_coh_print,
        save_info=True,
        time_cut=time_cut,
        inhom_indices=inhom_indices,
        inhom_batches=args.inhom_batches,
        inhom_idx=args.inhom_idx,
        average_over_inhom=not args.sum_inhom,
        coh_batches_meta=1,
        coh_idx_meta=0,
    )


def run_2d_mode(args):
    """Run 2D mode with batch processing for multiple coherence times."""
    config_path = SCRIPTS_DIR / "config.yaml"
    coh_batches = args.coh_batches
    coh_idx = args.coh_idx

    # Build base simulation (applies CLI overrides inside)
    sim_oqs, time_cut = create_base_sim_oqs(config_path=config_path)

    print(f"🎯 Running 2D mode - batch {coh_idx + 1}/{coh_batches}")

    # Generate t_coh values for the full range (reuse detection times array)
    t_coh_vals = sim_oqs.times_det

    # Split into coh_batches
    subarrays = np.array_split(t_coh_vals, coh_batches)
    if coh_idx >= len(subarrays):
        raise ValueError(f"Batch index {coh_idx} exceeds number of coh_batches {coh_batches}")

    t_coh_subarray = subarrays[coh_idx]
    print(
        f"📊 Processing {len(t_coh_subarray)} t_coh values: [{t_coh_subarray[0]:.1f}, {t_coh_subarray[-1]:.1f}] fs"
    )

    # Centralize inhom indices selection for this whole job (applies to all t_coh in the batch)
    n_inhom = int(sim_oqs.simulation_config.n_inhomogen)
    all_indices = np.arange(n_inhom, dtype=int)
    if args.inhom_batches is None:
        inhom_indices = all_indices
    else:
        if args.inhom_idx is None:
            raise ValueError("When providing --inhom_batches, also provide --inhom_idx")
        if args.inhom_batches <= 0:
            raise ValueError("--inhom_batches must be positive")
        if not (0 <= int(args.inhom_idx) < int(args.inhom_batches)):
            raise ValueError("--inhom_idx must satisfy 0 <= idx < --inhom_batches")
        subarrays_inh = np.array_split(all_indices, int(args.inhom_batches))
        inhom_indices = subarrays_inh[int(args.inhom_idx)]

    abs_data_path = None
    start_time = time.time()
    for i, t_coh in enumerate(t_coh_subarray):
        print(f"\n--- Progress: {i+1}/{len(t_coh_subarray)} ---")
        # Save info only for first run
        save_info = i == 0
        abs_data_path = run_single_t_coh_with_sim(
            sim_oqs,
            t_coh,
            save_info=save_info,
            time_cut=time_cut,
            inhom_indices=inhom_indices,
            inhom_batches=args.inhom_batches,
            inhom_idx=args.inhom_idx,
            average_over_inhom=not args.sum_inhom,
            coh_batches_meta=coh_batches,
            coh_idx_meta=coh_idx,
        )
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    print(f"\n✅ Batch {coh_idx + 1}/{coh_batches} completed!")
    print(f"\n🎯 To stack this datas into 2D (skips automatically if already stacked), run:")
    print(f'python stack_1dto2d.py --abs_path "{abs_data_path}" --skip_if_exists')


def main():
    """
    Main function with argument parsing and execution logic.
    """
    parser = argparse.ArgumentParser(
        description="1D Electronic Spectroscopy Simulation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""cd 
Examples:
  # Run single 1D simulation
  python calc_datas.py --simulation_type 1d

  # Run 2D batch mode
  python calc_datas.py --simulation_type 2d --coh_idx 0 --coh_batches 10
        """,
    )

    parser.add_argument(
        "--simulation_type",
        type=str,
        default="1d",
        choices=["1d", "2d"],
        help="Type of simulation: '1d' for single t_coh, '2d' for batch processing (default: 1d)",
    )

    parser.add_argument(
        "--coh_idx",
        type=int,
        default=0,
        help="Batch index for the current job (0 to coh_batches-1)",
    )
    parser.add_argument(
        "--coh_batches",
        type=int,
        default=1,
        help="Total number of coh_batches >= 1 (only for 2D mode)",
    )
    # Inhomogeneity batching (applies to both 1D and 2D)
    parser.add_argument(
        "--inhom_batches",
        type=int,
        default=1,
        help="Split inhomogeneous sampling into this many batches; if omitted, process all samples in one run.",
    )
    parser.add_argument(
        "--inhom_idx",
        type=int,
        default=0,
        help="Index of the inhomogeneity batch for this job (0..inhom_batches-1).",
    )
    parser.add_argument(
        "--sum_inhom",
        action="store_true",
        help="Do not average over inhom samples in this run; return sum to allow external combination across batches.",
    )
    args = parser.parse_args()

    # ARGUMENT VALIDATION

    if args.simulation_type == "2d":
        if args.coh_batches is not None and args.coh_batches <= 0:
            raise ValueError("Number of coh_batches must be positive for 2D mode")
        if args.coh_idx is not None and args.coh_idx < 0:
            raise ValueError("Batch index must be non-negative")
    # Validate inhom batching if provided
    if (args.inhom_batches is None) ^ (args.inhom_idx is None):
        raise ValueError("Provide both --inhom_batches and --inhom_idx, or neither.")
    if args.inhom_batches is not None:
        if args.inhom_batches <= 0:
            raise ValueError("--inhom_batches must be positive")
        if not (0 <= args.inhom_idx < args.inhom_batches):
            raise ValueError("--inhom_idx must satisfy 0 <= idx < --inhom_batches")

    print("=" * 80)
    print("1D ELECTRONIC SPECTROSCOPY SIMULATION")
    print(f"Simulation type: {args.simulation_type}")

    if args.simulation_type == "1d":
        run_1d_mode(args)
    elif args.simulation_type == "2d":
        run_2d_mode(args)

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETED")


if __name__ == "__main__":
    main()
