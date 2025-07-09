"""
1D Electronic Spectroscopy Simulation Script – Flexible Execution Mode

This script computes 1D spectroscopy data for a given set of simulation parameters.
It supports two modes of execution:

1. Single tau_coh mode:
   Run the simulation for one specific coherence time using:
       --tau_coh <value>

2. Batch mode:
   Split the tau_coh range [0, t_det_max) into n_batches equal parts.
   The specified batch index processes only its assigned subarray using:
       --batch_idx <index> --n_batches <total>

Additional optional arguments:
   --t_det_max <fs>   : Maximum detection time (default: 600.0 fs)
   --dt <fs>          : Spacing between tau_coh values (default: 10.0 fs)
   --t_wait <fs>      : Waiting time between pump and probe pulses (default: 0.0 fs)

This script is designed for both local development and HPC batch execution.
Results are saved automatically using the qspectro2d I/O framework.
"""

# ==============
# ANALYSIS: for f in batch_*.slurm; do sbatch "$f"; done
# ==============
# FOR N_ATOMS = 1, ODE_SOLVER=BR, RWA=TRUE
# for t_det_max=100, dt=0.1, one 1d calculation takes about 2-3 seconds, so
# -> 1000 tau_coh_vals: -> split into 20 batches=20jobs of 50 each, each job will take 3s * 50 = 150s = 2.5 minutes
# -> 1000 tau_coh_vals: -> split into 10 batches=10jobs of 100 each, each job will take 3s * 100 = 300s = 5 minutes
# one 1d calculation (t_det_max=600, dt=0.1) takes about 3-4 seconds, so
# -> 6000 tau_coh_vals: -> split into 20 batches=20jobs of 300 each, each job will take 5s * 300 = 1500s = 25 minutes
# -> 6000 tau_coh_vals: -> split into 10 batches=10jobs of 600 each, each job will take 5s * 600 = 3000s = 50 minutes

# FOR N_ATOMS = 2, ODE_SOLVER=BR, RWA=TRUE
# for t_det_max=100, dt=0.1, one 1d calculation takes about 5-6 second, so
# -> 1000 tau_coh_vals: -> split into 20 batches=20jobs of 50 each, each job will take 6s * 50 = 300s = 5 minutes
# -> 1000 tau_coh_vals: -> split into 10 batches=10jobs of 100 each, each job will take 6s * 100 = 600s = 10 minutes
# one 1d calculation (t_det_max=600, dt=0.1) takes about 18-19 seconds, so
# -> 6000 tau_coh_vals: -> split into 20 batches=20jobs of 300 each, each job will take 20s * 300 = 6000s = 100 minutes < 2 hours
# -> 6000 tau_coh_vals: -> split into 10 batches=10jobs of 600 each, each job will take 20s * 600 = 12000s = 200 minutes < 4 hours, actually on proteus it ran for 6,... h
# Estimation for RAM: 6000 tau vals * 10 batches < 40,000,000.0 complex64 numbers < 40 MB << 1 GB

from encodings.punycode import T
import time
import argparse
import numpy as np
from pathlib import Path

from qspectro2d.core.bath_system.bath_class import BathClass
from qspectro2d.config import DATA_DIR
from qspectro2d.spectroscopy import (
    create_system_parameters,
    run_1d_simulation,
    get_max_workers,
    print_simulation_header,
    print_simulation_summary,
)
from qspectro2d.data import (
    save_data_file,
    save_info_file,
    generate_unique_data_filename,
)
from spectroscopy import simulation
from qspectro2d.core.simulation_class import (
    AtomicSystem,
    LaserPulseSystem,
    SimulationConfigClass,
    SimClassOQS,
)

N_ATOMS = 1  # Number of atoms in the system, can be changed to 1 or 2


def run_single_tau(
    tau_coh: float, t_det_max: float, dt: float, t_wait: float, save_info: bool = False
) -> Path:
    print(f"\n=== Starting tau_coh = {tau_coh:.2f} fs ===")

    atomic_config = {
        ### Atomic System parameters
        "simulation_type": "1d",
        # solver parameters
        "N_atoms": N_ATOMS,
        "J_cm": 300,  # Coupling strength [cm⁻¹]
        "freqs_cm": [16000],  # Frequency of atom A [cm⁻¹]
    }
    system = AtomicSystem(**atomic_config)
    pulse_config = {
        "pulse_fwhm": 15.0 if N_ATOMS == 1 else 5.0,
        "base_amplitude": 0.005,  # Rename "E0" to match function signature
        "pulse_type": "gaussian",  # fix typo: "pulse_types" → "pulse_type"
        "carrier_freq_cm": atomic_config["omega_A_cm"],
        "delays": [0.0, tau_coh, tau_coh + t_wait],
    }
    laser = LaserPulseSystem.from_delays(**pulse_config)

    simulation_config = {
        ### Simulation parameters
        # solver parameters
        "ODE_Solver": "Paper_eqs",
        "RWA_SL": True,
        "keep_track": "eigenstates",  # or "basis" to track basis states
        # times
        "tau_coh": tau_coh,
        "t_wait": 0.0,
        "t_det_max": t_det_max,
        "dt": dt,
        # phase cycling -> 16 parallel jobs
        "n_phases": 4,
        # inhomogeneous broadening
        "n_freqs": 100,
        "Delta_cm": 300,
    }

    info_config = SimulationConfigClass(
        **simulation_config,
    )
    bath_config = {
        ### Bath parameters
        "bath_type": "paper",
        # Temperature / cutoff of the bath
        "Temp": 1e-5,  # zero temperature
        "cutoff_": 1e2,  # later * omega_A
        # decay  rates
        "gamma_0": 1 / 300.0,  # default value 1/300
        "gamma_phi": 1 / 100.0,  # default value 1e-2
    }
    bath = BathClass(**bath_config)

    # Create the simulation class instance
    sim_oqs = SimClassOQS(
        simulation_config=info_config,
        system=system,
        laser=laser,
        bath=bath,
        keep_track="eigenstates",  # or "basis" to track basis states
    )

    start_time = time.time()
    max_workers = get_max_workers()
    print_simulation_header(info_config, max_workers)

    system = create_system_parameters(info_config)

    t_det, data = run_1d_simulation(info_config, system, max_workers)
    abs_path = Path(generate_unique_data_filename(system, info_config))
    data_path = Path(f"{abs_path}_data.npz")
    print(f"\nSaving data to: {data_path}")
    save_data_file(data_path, data, t_det)

    rel_path = abs_path.relative_to(DATA_DIR)

    if save_info:
        info_path = Path(f"{abs_path}_info.pkl")
        save_info_file(info_path, system, info_config)

        elapsed = time.time() - start_time

        print_simulation_summary(
            elapsed, data, rel_path, "1d"
        )  # Print the paths for feed-forward to plotting script

        print(f"{'='*60}")
        print(f"\n🎯 To plot this data, run:")
        print(f'python plot_datas.py --rel_path "{rel_path}"')
        print(f"{'='*60}")

        # For shell scripts, we need absolute paths for file existence checks

    return rel_path.parent


def main():
    parser = argparse.ArgumentParser(description="Run 1D spectroscopy.")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--tau_coh", type=float, help="Single tau_coh value (fs)")
    group.add_argument("--batch_idx", type=int, help="Batch index for tau_coh sweep")

    parser.add_argument("--n_batches", type=int, default=1, help="Number of batches")
    parser.add_argument(
        "--t_det_max", type=float, default=600.0, help="Detection time window (fs)"
    )
    parser.add_argument("--dt", type=float, default=10.0, help="tau_coh spacing (fs)")

    args = parser.parse_args()

    # =============================
    # Robust argument handling
    # =============================
    if args.tau_coh is not None:
        rel_dir = run_single_tau(args.tau_coh, args.t_det_max, args.dt, save_info=True)

    elif args.batch_idx is not None:
        tau_coh_vals = np.linspace(
            0, args.t_det_max, int((args.t_det_max - 0) / args.dt) + 1
        )
        subarrays = np.array_split(tau_coh_vals, args.n_batches)
        tau_subarray = subarrays[args.batch_idx]

        print(
            f"🎯 Running batch {args.batch_idx + 1}/{args.n_batches} with {len(tau_subarray)} tau_coh values..."
        )
        for tau_coh in tau_subarray:
            save_info = tau_coh == 0
            rel_dir = run_single_tau(
                tau_coh, args.t_det_max, args.dt, save_info=save_info
            )

        print(f"\n🎯 To stack this data, run:")
        print(f'python stack_tau_coh_to_2d.py --rel_path "{rel_dir}"')

    else:
        # Default: run a standard single tau_coh simulation
        print(
            "No arguments specified. Running default single tau_coh simulation with tau_coh=0.0 fs."
        )
        rel_dir = run_single_tau(0.0, args.t_det_max, args.dt, save_info=True)


if __name__ == "__main__":
    main()
