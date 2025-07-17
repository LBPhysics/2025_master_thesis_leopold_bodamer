"""
1D Electronic Spectroscopy Simulation Script – Flexible Execution Mode

This script runs 1D spectroscopy data for a given set of simulation parameters.
It supports two modes of execution:

# Determine whether to use batch mode or single t_coh mode based on the provided arguments.
    --simulation_type <type>: Type of simulation (default: "1d")
        -> "1d" Run the simulation for one specific coherence time
        -> "2d" Run the simulation for a range of coherence times, splitted into batches

# other arguments:
    --batch_idx <index>: Batch index for the current job (0 to n_batches-1, default: 0)
    --n_batches <total>: Total number of batches (default: 1)
    --t_det_max <fs>   : Maximum detection time (default: 600.0 fs)
    --t_wait <fs>      : Waiting time between pump and probe pulses (default: 0.0 fs)
    --t_coh <value>    : Coherence time between 2 pump pulses (default: 0.0 fs)
    --dt <fs>          : Spacing between t_coh and t_det values (default: 10.0 fs)

This script is designed for both local development and HPC batch execution.
Results are saved automatically using the qspectro2d I/O framework.

# testing
python calc_datas.py --t_coh 10.0 --t_wait 25.0 --dt 2.0
python calc_datas.py --simulation_type 2d --t_det_max 100.0 --t_coh 300.0 --dt 20
# full simulations
python calc_datas.py --t_coh 300.0 --t_det_max 1000.0 --dt 2
python calc_datas.py --simulation_type 2d --t_det_max 100.0 --t_coh 300.0 --dt 0.1
"""

import time
import argparse
import numpy as np
from pathlib import Path

from qspectro2d.config import DATA_DIR
from qspectro2d.spectroscopy.calculations import (
    parallel_compute_1d_E_with_inhomogenity,
)
from qspectro2d.utils import (
    save_data_file,
    save_info_file,
    generate_unique_data_filename,
    print_simulation_summary,
)
from qspectro2d.utils.simulation_utils import create_base_sim_oqs
from qspectro2d.core.simulation_class import SimulationModuleOQS


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
        Path: Relative path to the saved data directory
    """
    print(f"\n=== Starting t_coh = {t_coh:.2f} fs ===")

    # Update t_coh in the simulation config
    sim_oqs.simulation_config.t_coh = t_coh
    t_wait = sim_oqs.simulation_config.t_wait
    sim_oqs.laser.update_delays = [t_coh, t_wait]

    start_time = time.time()

    # Run the simulation
    print("Computing 1D polarization with parallel processing...")
    try:
        data = parallel_compute_1d_E_with_inhomogenity(
            sim_oqs=sim_oqs,
            time_cut=time_cut,
        )
        print("✅ Parallel computation completed successfully!")
    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise

    # Save data
    simulation_config_dict = sim_oqs.simulation_config.to_dict()
    abs_path = Path(
        generate_unique_data_filename(sim_oqs.system, simulation_config_dict)
    )
    data_path = Path(f"{abs_path}_data.npz")
    print(f"\nSaving data to: {data_path}")

    save_data_file(data_path, data, sim_oqs.times_det)

    rel_path = abs_path.relative_to(DATA_DIR)

    if save_info:
        # all_infos_as_dict = sim_oqs.to_dict() TODO update the saving to incorporate all the info data in one dict?
        info_path = Path(f"{abs_path}_info.pkl")
        save_info_file(
            info_path,
            sim_oqs.system,
            bath=sim_oqs.bath,
            laser=sim_oqs.laser,
            info_config=simulation_config_dict,
        )

        print(f"{'='*60}")
        print(f"\n🎯 To plot this data, run:")
        print(f'python plot_datas.py --rel_path "{rel_path}"')

    elapsed = time.time() - start_time
    print_simulation_summary(elapsed, data, rel_path, "1d")

    return rel_path.parent


def run_1d_mode(args):
    """
    Run single 1D simulation for a specific coherence time.

    Parameters:
        args: Parsed command line arguments
    """
    print(f"🎯 Running 1D mode with t_coh = {args.t_coh:.2f} fs")

    # Create base simulation and validate solver once
    sim_oqs, time_cut = create_base_sim_oqs(args)

    # Run single simulation
    rel_path = run_single_t_coh_with_sim(
        sim_oqs, args.t_coh, save_info=True, time_cut=time_cut
    )

    print(f"\n✅ 1D simulation completed!")


def run_2d_mode(args):
    """
    Run 2D mode with batch processing for multiple coherence times.

    Parameters:
        args: Parsed command line arguments
    """
    print(f"🎯 Running 2D mode - batch {args.batch_idx + 1}/{args.n_batches}")

    # Create base simulation and validate solver once
    sim_oqs, time_cut = create_base_sim_oqs(args)

    # Generate t_coh values for the full range
    t_coh_vals = sim_oqs.times_det

    # Split into batches
    subarrays = np.array_split(t_coh_vals, args.n_batches)

    if args.batch_idx >= len(subarrays):
        raise ValueError(
            f"Batch index {args.batch_idx} exceeds number of batches {args.n_batches}"
        )

    t_coh_subarray = subarrays[args.batch_idx]

    print(
        f"📊 Processing {len(t_coh_subarray)} t_coh values: [{t_coh_subarray[0]:.1f}, {t_coh_subarray[-1]:.1f}] fs"
    )

    rel_path = None
    start_time = time.time()
    for i, t_coh in enumerate(t_coh_subarray):
        print(f"\n--- Progress: {i+1}/{len(t_coh_subarray)} ---")
        # Save info only for first run
        save_info = t_coh == 0
        rel_path = run_single_t_coh_with_sim(
            sim_oqs,
            t_coh,
            save_info=save_info,
            time_cut=time_cut,
        )
    elapsed = time.time() - start_time
    dummy_data = np.zeros((len(t_coh_subarray), len(sim_oqs.times_det)))
    print_simulation_summary(elapsed, dummy_data, rel_path, "2d")

    print(f"\n✅ Batch {args.batch_idx + 1}/{args.n_batches} completed!")
    print(f"\n🎯 To stack this data into 2D, run:")
    print(f'python stack_t_coh_to_2d.py --rel_path "{rel_path}"')


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
  python calc_data.py --simulation_type 1d --t_coh 50.0

  # Run 2D batch mode
  python calc_data.py --simulation_type 2d --batch_idx 0 --n_batches 10
        """,
    )

    # =============================
    # SIMULATION TYPE ARGUMENT
    # =============================
    parser.add_argument(
        "--simulation_type",
        type=str,
        default="1d",
        choices=["1d", "2d"],
        help="Type of simulation: '1d' for single t_coh, '2d' for batch processing (default: 1d)",
    )

    # =============================
    # BATCH PROCESSING ARGUMENTS
    # =============================
    parser.add_argument(
        "--batch_idx",
        type=int,
        default=0,
        help="Batch index for the current job (0 to n_batches-1, default: 0)",
    )

    parser.add_argument(
        "--n_batches", type=int, default=1, help="Total number of batches (default: 1)"
    )

    # =============================
    # TIME PARAMETERS
    # =============================
    parser.add_argument(
        "--t_det_max",
        type=float,
        default=600.0,
        help="Maximum detection time (default: 600.0 fs)",
    )

    parser.add_argument(
        "--t_wait",
        type=float,
        default=0.0,
        help="Waiting time between pump and probe pulses (default: 0.0 fs)",
    )

    parser.add_argument(
        "--t_coh",
        type=float,
        default=0.0,
        help="Coherence time between 2 pump pulses (default: 0.0 fs)",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=10.0,
        help="Spacing between t_coh and t_det values (default: 10.0 fs)",
    )

    args = parser.parse_args()

    # =============================
    # ARGUMENT VALIDATION
    # =============================
    if args.simulation_type == "2d":
        args.t_coh = 0.0  # force default value for 2D mode
        if args.n_batches <= 0:
            raise ValueError("Number of batches must be positive for 2D mode")
        elif args.batch_idx < 0:
            raise ValueError("Batch index must be non-negative")

    if args.dt <= 0:
        raise ValueError("Time step dt must be positive")

    if args.t_det_max <= 0:
        raise ValueError("Detection time t_det_max must be positive")

    # =============================
    # EXECUTION LOGIC
    # =============================
    print("=" * 80)
    print("1D ELECTRONIC SPECTROSCOPY SIMULATION")
    print(f"Simulation type: {args.simulation_type}")
    print(f"Detection time window: {args.t_det_max} fs")
    print(f"Time step: {args.dt} fs")
    print(f"Waiting time: {args.t_wait} fs")

    if args.simulation_type == "1d":
        print(f"Coherence time: {args.t_coh} fs")
        run_1d_mode(args)

    elif args.simulation_type == "2d":
        print(f"Batch processing: {args.batch_idx + 1}/{args.n_batches}")
        run_2d_mode(args)

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETED")


if __name__ == "__main__":
    main()
