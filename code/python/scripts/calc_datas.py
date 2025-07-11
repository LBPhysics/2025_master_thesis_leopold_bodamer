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
python3 calc_datas.py --t_coh 10.0 --t_wait 25.0 --dt 2.0
python3 calc_datas.py --simulation_type 2d --t_det_max 100.0 --t_coh 300.0 --dt 20
# full simulations
python3 calc_datas.py --t_coh 300.0 --t_det_max 1000.0 --dt 2
python3 calc_datas.py --simulation_type 2d --t_det_max 100.0 --t_coh 300.0 --dt 0.1
"""

import time
import argparse
from unittest.mock import DEFAULT
import numpy as np
from pathlib import Path

from qspectro2d.core.bath_system.bath_class import BathSystem
from qspectro2d.config import DATA_DIR
from qspectro2d.spectroscopy import (
    get_max_workers,
    print_simulation_summary,
)
from qspectro2d.spectroscopy.calculations import (
    parallel_compute_1d_E_with_inhomogenity,
    check_the_solver,
)
from qspectro2d.data import (
    save_data_file,
    save_info_file,
    generate_unique_data_filename,
)
from qspectro2d.core.simulation_class import (
    AtomicSystem,
    LaserPulseSequence,
    SimulationConfig,
    SimulationModuleOQS,
)

N_ATOMS = 1
DEFAULT_ODE_SOLVER = "Paper_eqs"
DEFAULT_RWA_SL = True

DEFAULT_BATH_TYPE = "paper"
DEFAULT_BATH_TEMP = 1e-5
DEFAULT_BATH_CUTOFF = 1e2
DEFAULT_BATH_GAMMA_0 = 1 / 300.0
DEFAULT_BATH_GAMMA_PHI = 1 / 100.0
DEFAULT_N_FREQS = 1
DEFAULT_PHASES = 1  # Number of phase cycles for the simulation
DEFAULT_DELTA_CM = 0.0  # Inhomogeneous broadening [cm⁻¹]
DEFAULT_APPLY_IFFT = (
    True  # Whether to apply inverse Fourier transform in the simulation
)


def create_simulation_module_from_configs(
    atom_config: dict,
    laser_config: dict,
    bath_config: dict,
    simulation_config: dict,
) -> SimulationModuleOQS:
    """
    Create a simulation module from the provided configuration dictionaries.

    Parameters:
        atom_config (dict): Atomic system configuration.
        laser_config (dict): Laser pulse sequence configuration.
        bath_config (dict): Bath parameters configuration.
        simulation_config (dict): Simulation parameters configuration.

    Returns:
        SimulationModuleOQS: Configured simulation class instance.
    """
    system = AtomicSystem.from_dict(atom_config)
    laser = LaserPulseSequence.from_delays(**laser_config)
    bath = BathSystem.from_dict(bath_config)

    return SimulationModuleOQS(
        simulation_config=SimulationConfig(**simulation_config),
        system=system,
        laser=laser,
        bath=bath,
    )


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

    # Update laser delays with new t_coh
    t_wait = sim_oqs.simulation_config.t_wait
    sim_oqs.laser.update_delays = [0.0, t_coh, t_coh + t_wait]

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
    detection_times = sim_oqs.times_det[sim_oqs.times_det_actual < time_cut]
    save_data_file(data_path, data, detection_times)

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


def create_base_sim_oqs(args) -> tuple[SimulationModuleOQS, float]:
    """
    Create base simulation instance and perform solver validation once.

    Parameters:
        args: Parsed command line arguments

    Returns:
        tuple: (SimulationModuleOQS instance, time_cut from solver validation)
    """
    print("🔧 Creating base simulation configuration...")

    atomic_config = {
        "N_atoms": N_ATOMS,
        "freqs_cm": [16000],  # Frequency of atom A [cm⁻¹]
        "dip_moments": [1.0] * N_ATOMS,  # Dipole moments for each atom
        "Delta_cm": DEFAULT_DELTA_CM,  # inhomogeneous broadening [cm⁻¹]
    }
    if N_ATOMS >= 2:
        atomic_config["J_cm"] = 300.0

    # Use dummy t_coh=0 for initial setup and solver check
    pulse_config = {
        "pulse_fwhm": 15.0 if N_ATOMS == 1 else 5.0,
        "base_amplitude": 0.005,
        "envelope_type": "gaussian",
        "carrier_freq_cm": np.mean(atomic_config["freqs_cm"]),
        "delays": [
            0.0,
            args.t_coh,
            args.t_coh + args.t_wait,
        ],  # dummy delays, will be updated
    }

    max_workers = get_max_workers()
    simulation_config_dict = {
        "simulation_type": "1d",
        "max_workers": max_workers,
        "apply_ift": DEFAULT_APPLY_IFFT,
        ### Simulation parameters
        "ODE_Solver": DEFAULT_ODE_SOLVER,
        "RWA_SL": DEFAULT_RWA_SL,
        "keep_track": "basis",
        # times
        "t_coh": args.t_coh,  # dummy value, will be updated
        "t_wait": args.t_wait,
        "t_det_max": args.t_det_max,
        "dt": args.dt,
        # phase cycling
        "n_phases": DEFAULT_PHASES,
        # inhomogeneous broadening
        "n_freqs": DEFAULT_N_FREQS,
    }

    bath_config = {
        ### Bath parameters
        "bath_type": DEFAULT_BATH_TYPE,
        "Temp": DEFAULT_BATH_TEMP,  # zero temperature
        "cutoff_": DEFAULT_BATH_CUTOFF,
        "gamma_0": DEFAULT_BATH_GAMMA_0,
        "gamma_phi": DEFAULT_BATH_GAMMA_PHI,
    }

    # Create the simulation class instance
    sim_oqs = create_simulation_module_from_configs(
        atom_config=atomic_config,
        laser_config=pulse_config,
        bath_config=bath_config,
        simulation_config=simulation_config_dict,
    )

    print(sim_oqs.simulation_config)

    ### Validate solver once at the beginning
    time_cut = -np.inf
    t_max = sim_oqs.simulation_config.t_max
    print("🔍 Validating solver with dummy t_coh=0...")
    try:
        _, time_cut = check_the_solver(sim_oqs)
        print("#" * 60)
        print(
            f"✅ Solver validation worked: Evolution becomes unphysical at "
            f"({time_cut / t_max:.2f} × t_max)"
        )
        print("#" * 60)
    except Exception as e:
        print(f"⚠️  WARNING: Solver validation failed: {e}")

    if time_cut < t_max:
        print(
            f"⚠️  WARNING: Time cut {time_cut} is less than the last time point "
            f"{t_max}. This may affect the simulation results.",
            flush=True,
        )

    return sim_oqs, time_cut


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
    rel_dir = run_single_t_coh_with_sim(
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

    t_wait = args.t_wait
    rel_dir = None
    for i, t_coh in enumerate(t_coh_subarray):
        print(f"\n--- Progress: {i+1}/{len(t_coh_subarray)} ---")
        save_info = t_coh == 0  # Only save info for first simulation
        sim_oqs.simulation_config.t_coh = t_coh
        sim_oqs.laser.update_delays = [0.0, t_coh, t_coh + t_wait]
        rel_dir = run_single_t_coh_with_sim(
            sim_oqs,
            t_coh,
            save_info=save_info,
            time_cut=time_cut,
        )

    print(f"\n✅ Batch {args.batch_idx + 1}/{args.n_batches} completed!")
    print(f"\n🎯 To stack this data into 2D, run:")
    print(f'python stack_t_coh_to_2d.py --rel_path "{rel_dir}"')


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
    if args.simulation_type == "2d" and args.n_batches <= 0:
        raise ValueError("Number of batches must be positive for 2D mode")

    if args.simulation_type == "2d" and args.batch_idx < 0:
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
