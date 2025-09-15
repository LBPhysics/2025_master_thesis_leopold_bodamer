"""
1D Electronic Spectroscopy Simulation Script – Flexible Execution Mode

This script runs 1D spectroscopy datas for a given set of simulation parameters.
It supports two modes of execution:

# Determine whether to use batch mode or single t_coh mode based on the provided arguments.
    --simulation_type <type>: Type of simulation (default: "1d")
        -> "1d" Run the simulation for one specific coherence time
        -> "2d" Run the simulation for a range of coherence times, splitted into n_batches

# other arguments:
    --n_batches <total>: Total number of n_batches (default: 1)
    --batch_idx <index>: Batch index for the current job (0 to n_batches-1, default: 0)

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

from qspectro2d.spectroscopy.calculations import (
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

warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")


def run_single_t_coh_with_sim(
    sim_oqs: SimulationModuleOQS,
    t_coh: float,
    save_info: bool = False,
    time_cut: float = -np.inf,
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
        datas = parallel_compute_1d_E_with_inhomogenity(
            sim_oqs=sim_oqs,
            time_cut=time_cut,
        )  # can also be a tuple (P_k_Rephasing, P_k_NonRephasing)
        print("✅ Parallel computation completed successfully!")
    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise

    # Save datas
    abs_path = generate_unique_data_filename(sim_oqs.system, sim_config_obj)
    abs_data_path = Path(f"{abs_path}_data.npz")

    signal_types = sim_config_obj.signal_types
    save_data_file(abs_data_path, datas, sim_oqs.times_det, signal_types=signal_types)

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

    run_single_t_coh_with_sim(sim_oqs, t_coh_print, save_info=True, time_cut=time_cut)


def run_2d_mode(args):
    """Run 2D mode with batch processing for multiple coherence times."""
    config_path = SCRIPTS_DIR / "config.yaml"
    n_batches = args.n_batches
    batch_idx = args.batch_idx

    # Build base simulation (applies CLI overrides inside)
    sim_oqs, time_cut = create_base_sim_oqs(config_path=config_path)

    print(f"🎯 Running 2D mode - batch {batch_idx + 1}/{n_batches}")

    # Generate t_coh values for the full range (reuse detection times array)
    t_coh_vals = sim_oqs.times_det

    # Split into n_batches
    subarrays = np.array_split(t_coh_vals, n_batches)
    if batch_idx >= len(subarrays):
        raise ValueError(f"Batch index {batch_idx} exceeds number of n_batches {n_batches}")

    t_coh_subarray = subarrays[batch_idx]
    print(
        f"📊 Processing {len(t_coh_subarray)} t_coh values: [{t_coh_subarray[0]:.1f}, {t_coh_subarray[-1]:.1f}] fs"
    )

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
        )
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    print(f"\n✅ Batch {batch_idx + 1}/{n_batches} completed!")
    print(f"\n🎯 To stack this datas into 2D, run:")
    print(f'python stack_1dto2d.py --abs_path "{abs_data_path.parent}"')


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
  python calc_data.py --simulation_type 1d

  # Run 2D batch mode
  python calc_data.py --simulation_type 2d --batch_idx 0 --n_batches 10
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
        "--batch_idx",
        type=int,
        default=0,
        help="Batch index for the current job (0 to n_batches-1)",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=1,
        help="Total number of n_batches >= 1 (only for 2D mode)",
    )
    args = parser.parse_args()

    # ARGUMENT VALIDATION

    if args.simulation_type == "2d":
        if args.n_batches is not None and args.n_batches <= 0:
            raise ValueError("Number of n_batches must be positive for 2D mode")
        if args.batch_idx is not None and args.batch_idx < 0:
            raise ValueError("Batch index must be non-negative")

    # EXECUTION LOGIC

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
