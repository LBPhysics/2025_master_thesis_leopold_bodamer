"""
1D Electronic Spectroscopy Simulation Script – Flexible Execution Mode

This script computes 1D spectroscopy data for a given set of simulation parameters.
It supports two modes of execution:

1. Single tau_coh mode:
   Run the simulation for one specific coherence time using:
       --tau-coh <value>

2. Batch mode:
   Split the tau_coh range [0, t_det_max) into n_batches equal parts.
   The specified batch index processes only its assigned subarray using:
       --batch-idx <index> --n-batches <total>

Additional optional arguments:
   --t-det-max <fs>   : Maximum detection time (default: 600.0 fs)
   --dt <fs>          : Spacing between tau_coh values (default: 10.0 fs)

This script is designed for both local development and HPC batch execution.
Results are saved automatically using the qspectro2d I/O framework.
"""

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


def run_single_tau(tau_coh: float, t_det_max: float, dt: float):
    print(f"\n=== Starting tau_coh = {tau_coh:.2f} fs ===")

    config = {
        "simulation_type": "1d",
        "N_atoms": 1,
        "ODE_Solver": "BR",
        "tau_coh": float(tau_coh),
        "T_wait": 0.0,
        "t_det_max": t_det_max,
        "dt": dt,
        "n_phases": 2,
        "n_freqs": 1,
        "Delta_cm": 0,
        "pulse_type": "gaussian",
        "E0": 0.005,
        "pulse_fwhm": 15.0,
        "RWA_laser": True,
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
    group.add_argument("--tau-coh", type=float, help="Single tau_coh value (fs)")
    group.add_argument("--batch-idx", type=int, help="Batch index for tau_coh sweep")

    parser.add_argument("--n-batches", type=int, default=1, help="Number of batches")
    parser.add_argument(
        "--t-det-max", type=float, default=600.0, help="Detection time window (fs)"
    )
    parser.add_argument("--dt", type=float, default=10.0, help="tau_coh spacing (fs)")

    args = parser.parse_args()

    if args.tau_coh is not None:
        run_single_tau(args.tau_coh, args.t_det_max, args.dt)

    elif args.batch_idx is not None:
        tau_vals = np.arange(0, args.t_det_max, args.dt)
        subarrays = np.array_split(tau_vals, args.n_batches)
        tau_subarray = subarrays[args.batch_idx]

        print(
            f"🎯 Running batch {args.batch_idx + 1}/{args.n_batches} with {len(tau_subarray)} tau_coh values..."
        )
        for tau_coh in tau_subarray:
            run_single_tau(tau_coh, args.t_det_max, args.dt)


if __name__ == "__main__":
    main()
