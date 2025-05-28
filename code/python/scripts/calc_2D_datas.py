# Import the outsourced settings / functions
from src.spectroscopy.post_processing import *
from src.core.system_parameters import SystemParameters
from src.spectroscopy.calculations import (
    batch_process_all_combinations_with_inhomogeneity,
)
from src.spectroscopy.inhomogenity import sample_from_sigma, check_the_solver

import numpy as np
import psutil
import copy
import time
import pickle
import sys
import os


def main():
    """
    Main function to run the 2D spectroscopy simulation.
    """
    start_time = time.time()

    # =============================
    # SIMULATION PARAMETERS -> determines the number of combinations -> number of processors needed to optimally perform the simulation -> Time of the simulation
    # =============================
    n_times_T = 1  # Number of T_wait values (pump-probe separation)
    n_phases = 2  # Number of phases for phase cycling
    n_freqs = 1  # Number of frequencies for inhomogeneous broadening

    # Generate random subset of phases from the full set
    all_phases = [k * np.pi / 2 for k in range(4)]  # [0, π/2, π, 3π/2]
    phases = np.random.choice(all_phases, size=n_phases, replace=False).tolist()
    max_workers = psutil.cpu_count(logical=True)

    print("=" * 60)
    print("2D ELECTRONIC SPECTROSCOPY SIMULATION")
    print("=" * 60)
    print(f"Configuration:")
    print(
        f"  Parameters: #T_wait={n_times_T}, #phases={n_phases}, #frequencies={n_freqs}"
    )
    print(
        f"  Total combinations processed: {n_times_T * n_phases * n_phases * n_freqs}"
    )
    print(f"  Parallel workers used: {max_workers}")

    print()

    # =============================
    # SYSTEM PARAMETERS
    # =============================
    system = SystemParameters(
        N_atoms=1,
        ODE_Solver="Paper_eqs",
        RWA_laser=True,
        t_max=10.0,  # determines Δω
        dt=0.2,  # determines ωₘₐₓ
        Delta_cm=200 if n_freqs > 1 else 0,
    )

    # Create time arrays
    Delta_ts = system.Delta_ts
    times = np.arange(-Delta_ts[0], system.t_max, system.dt)
    T_wait_max = times[-1] / 10
    times_T = np.linspace(0, T_wait_max, n_times_T)

    print(f"System configuration:")
    system.summary()

    # =============================
    # SOLVER VALIDATION
    # =============================
    print("\nValidating solver stability...")
    test_system = copy.deepcopy(system)
    test_system.t_max = 10 * system.t_max
    test_system.dt = 10 * system.dt
    times_test = np.arange(-Delta_ts[0], test_system.t_max, test_system.dt)

    global time_cut  # SOMEHOW THIS Variable MAKES A PROBLEM NOW!!!!! TODO
    _, time_cut = check_the_solver(times_test, test_system)
    print(f"Evolution remains physical until: {time_cut:.1f} fs")

    # =============================
    # FREQUENCY SAMPLING
    # =============================
    omega_ats = sample_from_sigma(
        n_freqs, system.Delta_cm, system.omega_A_cm, E_range=3
    )

    # =============================
    # RUN SIMULATION
    # =============================
    print(f"\nStarting 2D spectroscopy calculation...")

    kwargs = {"plot_example": False}

    two_d_datas = batch_process_all_combinations_with_inhomogeneity(
        omega_ats=omega_ats,
        phases=phases,
        times_T=times_T,
        times=times,
        system=system,
        max_workers=max_workers,
        **kwargs,
    )
    print(two_d_datas[0])
    # =============================
    # SAVE RESULTS
    # =============================
    output_dir = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "papers_with_proteus_output"
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    # Generate unique filename
    base_filename = f"data_tmax_{system.t_max:.0f}_dt_{system.dt}.pkl"
    save_path = os.path.join(output_dir, base_filename)

    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(
            output_dir, f"data_tmax_{system.t_max:.0f}_dt_{system.dt}_{counter}.pkl"
        )
        counter += 1

    with open(save_path, "wb") as f:  # TODO CHANGE THIS TO h5py?
        pickle.dump(
            {
                "system": system,
                "times": times,
                "times_T": times_T,
                "two_d_datas": two_d_datas,
            },
            f,
        )

    # =============================
    # SIMULATION SUMMARY
    # =============================
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    # Calculate largest 2D data size
    max_data_size = 0
    for data in two_d_datas:
        if data is not None:
            max_data_size = max(max_data_size, data.size)

    # Estimate memory usage in MB
    estimated_memory_usage = 2 * max_data_size * n_times_T * 8 / (1024**2)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()
    print(f"Data characteristics:")
    print(f"  Time parameters: t_max={system.t_max:.1f} fs, dt={system.dt:.1f} fs")
    print(f"  Largest 2D array size: {max_data_size:,} elements")
    print(f"  Time grid points: {len(times):,}")
    print()
    print(f"Performance:")
    print(f"  Execution time: {hours}h {minutes}m {seconds:.1f}s")
    print(f"  Estimated memory usage: {estimated_memory_usage:.2f} MB")
    print()
    print(f"  Data saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
