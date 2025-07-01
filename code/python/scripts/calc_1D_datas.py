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

This script is designed for both local development and HPC batch execution.
Results are saved automatically using the qspectro2d I/O framework.
"""
# ==============
# ANALYSIS:
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
# -> 6000 tau_coh_vals: -> split into 10 batches=10jobs of 600 each, each job will take 20s * 600 = 12000s = 200 minutes < 4 hours
# Estimation for RAM: 6000 tau vals * 10 batches < 40,000,000.0 complex64 numbers < 40 MB << 1 GB


import time
import argparse
import numpy as np

from qspectro2d.simulation import (
    create_system_parameters,
    run_1d_simulation,
    get_max_workers,
    print_simulation_header,
    print_simulation_summary,
)
from qspectro2d.data import save_simulation_data
N_ATOMS = 1  # Number of atoms in the system, can be changed to 1 or 2

def run_single_tau(tau_coh: float, t_det_max: float, dt: float):
    print(f"\n=== Starting tau_coh = {tau_coh:.2f} fs ===")

    config = {
        ### Main system configuration
        "simulation_type": "1d",
        # solver parameters
        "ODE_Solver": "Paper_eqs",
        "RWA_laser": True,        
        
        "N_atoms": N_ATOMS,
        "J_cm": 300 if N_ATOMS == 2 else 0,  # Coupling strength [cm⁻¹]
        # time parameters
        "tau_coh": float(tau_coh),
        "T_wait": 0.0,
        "t_det_max": t_det_max,
        "dt": dt,

        "pulse_fwhm": 15.0 if N_ATOMS == 1 else 5.0,  # Pulse FWHM for single atom [fs]
        "E0": 0.005,
        "pulse_type": "gaussian",
        # phase cycling -> 16 parallel jobs
        "n_phases": 4,
        # inhomogeneous broadening
        "n_freqs": 1,
        "Delta_cm": 0,
    }

    start_time = time.time()
    max_workers = get_max_workers()
    print_simulation_header(config, max_workers)

    system = create_system_parameters(config)
    system.summary()

    t_det, data = run_1d_simulation(config, system, max_workers)
    rel_path = save_simulation_data(system, config, data, axs1=t_det)
    elapsed = time.time() - start_time
    print_simulation_summary(elapsed, data, rel_path, "1d")


def main():
    parser = argparse.ArgumentParser(description="Run 1D spectroscopy.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tau_coh", type=float, help="Single tau_coh value (fs)")
    group.add_argument("--batch_idx", type=int, help="Batch index for tau_coh sweep")

    parser.add_argument("--n_batches", type=int, default=1, help="Number of batches")

    # additional arguments
    parser.add_argument(
        "--t_det_max", type=float, default=600.0, help="Detection time window (fs)"
    )
    parser.add_argument("--dt", type=float, default=10.0, help="tau_coh spacing (fs)")

    args = parser.parse_args()

    if args.tau_coh is not None:
        run_single_tau(args.tau_coh, args.t_det_max, args.dt)

    elif args.batch_idx is not None:
        tau_coh_vals = np.linspace(0, args.t_det_max, int((args.t_det_max - 0) / args.dt) + 1)
        subarrays = np.array_split(tau_coh_vals, args.n_batches)
        tau_subarray = subarrays[args.batch_idx]

        print(
            f"🎯 Running batch {args.batch_idx + 1}/{args.n_batches} with {len(tau_subarray)} tau_coh values..."
        )
        for tau_coh in tau_subarray:
            run_single_tau(tau_coh, args.t_det_max, args.dt)


if __name__ == "__main__":
    main()
