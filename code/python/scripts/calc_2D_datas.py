"""
2D Electronic Spectroscopy Simulation Script - Direct Parameter Version

This script computes 2D electronic spectroscopy data using parallel processing
and saves results in pickle format. All parameters are defined directly in main().
"""

import time
from pathlib import Path
from common_fcts import (
    create_system_parameters,
    run_simulation,
    get_max_workers,
    print_simulation_header,
    print_simulation_summary,
    save_data_with_unique_path,
)


def main():
    """Main function to run the 2D spectroscopy simulation."""

    # =============================
    # SIMULATION PARAMETERS - MODIFY HERE
    # =============================

    ### Main system configuration
    N_atoms = 2  # Number of atoms (1 or 2)
    t_max = 20  # Maximum time [fs]
    dt = 0.1  # Time step [fs]
    ODE_Solver = "BR"  # ODE solver type

    ### System-specific parameters
    if N_atoms == 1:
        pulse_fwhm = 15.0  # Pulse FWHM for single atom [fs]
    elif N_atoms == 2:
        pulse_fwhm = 5.0  # Pulse FWHM for two atoms [fs]
    else:
        raise ValueError(f"Unsupported number of atoms: {N_atoms}")

    ### 2D spectroscopy parameters
    RWA_laser = False  # Use RWA for laser interaction
    T_wait_max = t_max / 2  # Maximum waiting time [fs]
    n_times_T = 1  # Number of T_wait values
    n_phases = 2  # Number of phases for phase cycling
    n_freqs = 1  # Number of frequencies for inhomogeneous broadening
    Delta_cm = 0  # Inhomogeneous broadening [cm⁻¹]
    envelope_type = "gaussian"  # Pulse envelope type
    E0 = 0.005  # Electric field amplitude

    # =============================
    # BUILD CONFIGURATION DICTIONARY
    # =============================
    config = {
        "simulation_type": "2d",  # Explicitly specify simulation type
        "N_atoms": N_atoms,
        "t_max": t_max,
        "dt": dt,
        "ODE_Solver": ODE_Solver,
        "pulse_fwhm": pulse_fwhm,
        "RWA_laser": RWA_laser,
        "T_wait_max": T_wait_max,
        "n_times_T": n_times_T,
        "n_phases": n_phases,
        "n_freqs": n_freqs,
        "Delta_cm": Delta_cm,
        "envelope_type": envelope_type,
        "E0": E0,
    }

    # =============================
    # PRINT CONFIGURATION SUMMARY
    # =============================
    print(f"Running 2D spectroscopy simulation with:")
    print(f"  N_atoms: {N_atoms}")
    print(f"  Solver: {ODE_Solver}")
    print(f"  Time: {t_max} fs (dt = {dt} fs)")
    print(f"  Pulse FWHM: {pulse_fwhm} fs")
    print("")

    # =============================
    # RUN SIMULATION
    # =============================
    start_time = time.time()

    # Get parallel processing configuration
    max_workers = get_max_workers()

    # Print simulation header
    print_simulation_header(config, max_workers, "2d")

    # Create system parameters
    system = create_system_parameters(config)
    print(f"System configuration:")
    system.summary()

    # Run simulation (returns standardized payload)
    payload = run_simulation(config, system, "2d")

    # Save data using the new workflow
    print("\nSaving simulation data...")
    relative_dir = save_data_with_unique_path(payload, config, system)

    # Print simulation summary
    elapsed_time = time.time() - start_time
    print_simulation_summary(elapsed_time, payload["data"], relative_dir, "2d")

    # Print the relative path for feed-forward to plotting script
    print(f"SAVED_DATA_PATH:{relative_dir}")


if __name__ == "__main__":
    main()
