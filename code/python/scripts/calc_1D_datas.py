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
from config.paths import DATA_DIR
import psutil
import time
import pickle
import sys
import os
from pathlib import Path
from datetime import datetime


def generate_unique_filename(
    output_dir: Path, n_phases: int, n_freqs: int, system
) -> Path:
    """
    Generate a unique filename for 1D polarization data.

    Parameters
    ----------
    output_dir : Path
        Directory where the file will be saved
    n_phases : int
        Number of phases used in the simulation
    n_freqs : int
        Number of frequencies used for inhomogeneous broadening
    system : SystemParameters
        System parameters object containing simulation settings

    Returns
    -------
    Path
        Unique file path for saving the data
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = (
        f"1d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
        f"ph{n_phases}_freq{n_freqs}_{timestamp}.pkl"
    )

    save_path = output_dir / base_filename

    # Ensure unique filename by adding counter if needed
    counter = 1
    while save_path.exists():
        name_with_counter = (
            f"1d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
            f"ph{n_phases}_freq{n_freqs}_{timestamp}_{counter}.pkl"
        )
        save_path = output_dir / name_with_counter
        counter += 1

    return save_path


def save_1d_data(
    t_det_vals: np.ndarray,
    data_avg: np.ndarray,
    tau_coh: float,
    T_wait: float,
    system,
    omega_ats: np.ndarray,
    n_phases: int,
    n_freqs: int,
    output_subdir: str = "1d_spectroscopy",
) -> Path:
    """
    Save 1D polarization simulation data to a pickle file.

    Parameters
    ----------
    t_det_vals : np.ndarray
        Detection time values
    data_avg : np.ndarray
        Averaged polarization data
    tau_coh : float
        Coherence time
    T_wait : float
        Waiting time
    system : SystemParameters
        System parameters object
    omega_ats : np.ndarray
        Atomic transition frequencies
    n_phases : int
        Number of phases used in the simulation
    n_freqs : int
        Number of frequencies used for inhomogeneous broadening
    output_subdir : str, optional
        Subdirectory for organizing data, by default "1d_spectroscopy"

    Returns
    -------
    Path
        Path where the data was saved
    """
    # Create output directory
    output_dir = DATA_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    save_path = generate_unique_filename(output_dir, n_phases, n_freqs, system)

    # Package data into dictionary
    data = {
        "t_det_vals": t_det_vals,
        "data_avg": data_avg,
        "tau_coh": tau_coh,
        "T_wait": T_wait,
        "system": system,
        "omega_ats": omega_ats,
        "n_phases": n_phases,
        "n_freqs": n_freqs,
    }

    # Save as pickle file
    try:
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"✓ Data saved successfully to: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ ERROR: Failed to save data: {e}")
        raise


def main():
    """
    Main function to run the 2D spectroscopy simulation.
    """
    start_time = time.time()

    # =============================
    # SIMULATION PARAMETERS -> determines the number of combinations -> number of processors needed to optimally perform the simulation -> Time of the simulation
    # =============================
    n_phases = 4  # Number of phases for phase cycling
    n_freqs = 1  # Number of frequencies for inhomogeneous broadening

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
        envelope_type="gaussian",  # Use Gaussian envelope
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

    # =============================
    # SAVE DATA
    # =============================
    print("\nSaving simulation data...")
    save_path = save_1d_data(
        t_det_vals=t_det_vals,
        data_avg=data_avg,
        tau_coh=tau_coh,
        T_wait=T_wait,
        system=system,
        omega_ats=omega_ats,
        n_phases=n_phases,
        n_freqs=n_freqs,
    )

    # =============================
    # SIMULATION SUMMARY
    # =============================
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Data shape: {data_avg.shape}")
    print(f"Data saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
