"""
1D Electronic Spectroscopy Simulation Script – Flexible Execution Mode

This script runs 1D spectroscopy data for a given set of simulation parameters.
It supports two modes of execution:

# Determine whether to use batch mode or single t_coh mode based on the provided arguments.
    --simulation_type <type>: Type of simulation (default: "1d")
        -> "1d" Run the simulation for one specific coherence time
        -> "2d" Run the simulation for a range of coherence times, splitted into n_batches

# other arguments:
    --config <path>: Path to the configuration file (default: scripts/config.yaml)
                     COVERS all the simulation parameters / nothing else needed,
                     however with the next couple of parameters, these defaults can be overridden:

    --n_batches <total>: Total number of n_batches (default: 1)
    --batch_idx <index>: Batch index for the current job (0 to n_batches-1, default: 0)
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
python calc_datas.py --t_coh 300.0 --t_det_max 1000.0 --dt 0.1
python calc_datas.py --simulation_type 2d --t_det_max 100.0 --t_coh 300.0 --dt 0.1
"""

import time
import argparse
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
import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in exp"
)


def _resolve_config_path(args) -> Path | None:
    """Resolve optional config path from CLI or default scripts directory.

    Precedence: CLI --config > scripts/config.yaml (if exists) > None (defaults)
    """
    if getattr(args, "config", None):
        p = Path(args.config).expanduser()
        return p if p.exists() else None
    candidate = SCRIPTS_DIR / "config.yaml"
    return candidate if candidate.exists() else None


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
        Path: absolute path to the saved data directory
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
    sim_config_obj = sim_oqs.simulation_config
    abs_path = generate_unique_data_filename(sim_oqs.system, sim_config_obj)
    abs_data_path = Path(f"{abs_path}_data.npz")

    save_data_file(abs_data_path, data, sim_oqs.times_det)

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
        print(f"\n🎯 To plot this data, run:")
        print(f'python plot_datas.py --abs_path "{abs_data_path}"')

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    return abs_data_path


def run_1d_mode(args):
    """Run single 1D simulation for a specific coherence time."""
    config_path = _resolve_config_path(args)

    # Build base simulation (applies CLI overrides inside)
    sim_oqs, time_cut = create_base_sim_oqs(
        args, config_path=str(config_path) if config_path else None
    )

    # Determine coherence time for printing (already overridden in sim if provided)
    t_coh_print = (
        args.t_coh if args.t_coh is not None else sim_oqs.simulation_config.t_coh
    )
    print(f"🎯 Running 1D mode with t_coh = {t_coh_print:.2f} fs")

    # Run single simulation
    abs_data_path = run_single_t_coh_with_sim(
        sim_oqs, t_coh_print, save_info=True, time_cut=time_cut
    )


def run_2d_mode(args):
    """Run 2D mode with batch processing for multiple coherence times."""
    config_path = _resolve_config_path(args)

    # Build base simulation (applies CLI overrides inside)
    sim_oqs, time_cut = create_base_sim_oqs(
        args, config_path=str(config_path) if config_path else None
    )

    # Resolve batching with precedence: CLI > defaults (no YAML fields now)
    n_batches = args.n_batches if args.n_batches is not None else 1
    batch_idx = args.batch_idx if args.batch_idx is not None else 0

    if n_batches <= 0:
        raise ValueError("Number of n_batches must be positive")
    if batch_idx < 0:
        raise ValueError("Batch index must be non-negative")

    print(f"🎯 Running 2D mode - batch {batch_idx + 1}/{n_batches}")

    # Generate t_coh values for the full range (reuse detection times array)
    t_coh_vals = sim_oqs.times_det

    # Split into n_batches
    subarrays = np.array_split(t_coh_vals, n_batches)
    if batch_idx >= len(subarrays):
        raise ValueError(
            f"Batch index {batch_idx} exceeds number of n_batches {n_batches}"
        )

    t_coh_subarray = subarrays[batch_idx]
    print(
        f"📊 Processing {len(t_coh_subarray)} t_coh values: [{t_coh_subarray[0]:.1f}, {t_coh_subarray[-1]:.1f}] fs"
    )

    abs_data_path = None
    start_time = time.time()
    for i, t_coh in enumerate(t_coh_subarray):
        print(f"\n--- Progress: {i+1}/{len(t_coh_subarray)} ---")
        # Save info only for first run
        save_info = t_coh == 0
        abs_data_path = run_single_t_coh_with_sim(
            sim_oqs,
            t_coh,
            save_info=save_info,
            time_cut=time_cut,
        )
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    print(f"\n✅ Batch {batch_idx + 1}/{n_batches} completed!")
    print(f"\n🎯 To stack this data into 2D, run:")
    print(f'python stack_t_coh_to_2d.py --abs_path "{abs_data_path}"')


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
  python calc_data.py --simulation_type 1d --t_coh 50.0  # uses scripts_dir/config.yaml if present
  python calc_data.py --simulation_type 1d --t_coh 50.0 --config path/to/config.yaml

  # Run 2D batch mode
  python calc_data.py --simulation_type 2d --batch_idx 0 --n_batches 10  # uses scripts_dir/config.yaml if present
  python calc_data.py --simulation_type 2d --batch_idx 0 --n_batches 10 --config path/to/config.yaml
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
    # CONFIGURATION FILE (optional)
    # =============================
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to configuration file (YAML/TOML/JSON). If omitted, config.yaml next to this script is used if present; otherwise built-in defaults are used."
        ),
    )

    # =============================
    # BATCH PROCESSING ARGUMENTS
    # =============================
    parser.add_argument(
        "--batch_idx",
        type=int,
        default=None,  # None => take from YAML/defaults
        help="Batch index for the current job (0 to n_batches-1)",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=None,  # None => take from YAML/defaults
        help="Total number of n_batches",
    )

    # =============================
    # TIME PARAMETERS
    # =============================
    parser.add_argument(
        "--t_det_max",
        type=float,
        default=None,  # None => take from YAML/defaults
        help="Maximum detection time (fs).",
    )
    parser.add_argument(
        "--t_wait",
        type=float,
        default=None,  # None => take from YAML/defaults
        help="Waiting time between pump and probe pulses (fs)",
    )
    parser.add_argument(
        "--t_coh",
        type=float,
        default=None,  # None => take from YAML/defaults (0.0 in 2D mode)
        help="Coherence time between 2 pump pulses (fs)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,  # None => take from YAML/defaults
        help="Spacing between t_coh and t_det values (fs).",
    )

    args = parser.parse_args()

    # =============================
    # ARGUMENT VALIDATION
    # =============================
    if args.simulation_type == "2d":
        if args.n_batches is not None and args.n_batches <= 0:
            raise ValueError("Number of n_batches must be positive for 2D mode")
        if args.batch_idx is not None and args.batch_idx < 0:
            raise ValueError("Batch index must be non-negative")
        if args.t_coh is None:
            args.t_coh = 0.0  # default for 2D mode
    if args.dt is not None and args.dt <= 0:
        raise ValueError("Time step dt must be positive")
    if args.t_det_max is not None and args.t_det_max <= 0:
        raise ValueError("Detection time t_det_max must be positive")

    # =============================
    # EXECUTION LOGIC
    # =============================
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
