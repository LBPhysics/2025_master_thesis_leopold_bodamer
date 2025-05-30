"""
script showing how to use the new parallel 1D polarization function.
This is a parallelized version for better performance.
"""

import numpy as np
from src.spectroscopy.calculations import (
    parallel_compute_1d_polarization_with_inhomogenity,
)
from src.spectroscopy.inhomogenity import sample_from_sigma
from src.core.system_parameters import SystemParameters
import psutil
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
    n_phases = 2  # Number of phases for phase cycling
    n_freqs = 2  # Number of frequencies for inhomogeneous broadening

    # Phase cycling
    all_phases = [k * np.pi / 2 for k in range(4)]  # [0, π/2, π, 3π/2]
    phases = np.random.choice(all_phases, size=n_phases, replace=False).tolist()
    max_workers = psutil.cpu_count(logical=True)
    # =============================
    # Setup parameters (adjust as needed)
    # =============================
    print("=" * 60)
    print("1D ELECTRONIC SPECTROSCOPY SIMULATION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Parameters: #phases={n_phases}, #frequencies={n_freqs}")
    print(f"  Total combinations processed: {n_phases * n_phases * n_freqs}")
    print(f"  Parallel workers used: {max_workers}")

    print()

    # Spectroscopy parameters
    tau_coh = 300.0
    T_wait = 1000.0

    # =============================
    # SYSTEM PARAMETERS
    # =============================
    system = SystemParameters(
        ODE_Solver="Paper_eqs",
        RWA_laser=True,
        t_max=tau_coh + T_wait + 600.0,  # determines Δω
        dt=2,  # determines ωₘₐₓ
        Delta_cm=200 if n_freqs > 1 else 0,
    )

    # Create time arrays
    FWHMs = system.FWHMs
    times = np.arange(-FWHMs[0], system.t_max, system.dt)

    print(f"System configuration:")
    system.summary()

    # Inhomogeneous broadening
    omega_ats = sample_from_sigma(n_freqs, FWHM=system.Delta_cm, mu=system.omega_A_cm)

    # =============================
    # Method 1: Direct function call
    # =============================
    print("Computing 1D polarization with parallel processing...")
    t_det_vals, data_avg = parallel_compute_1d_polarization_with_inhomogenity(
        omega_ats=omega_ats,
        phases=phases,
        tau_coh=tau_coh,
        T_wait=T_wait,
        times=times,
        system=system,
        max_workers=max_workers,
    )

    print(f"Results shape: {data_avg.shape}")
    print(f"Time range: {t_det_vals[0]:.1f} to {t_det_vals[-1]:.1f}")
    print("Parallel computation completed successfully!")


if __name__ == "__main__":
    main()
