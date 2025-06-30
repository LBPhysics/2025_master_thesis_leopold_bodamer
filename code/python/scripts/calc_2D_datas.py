"""
2D Electronic Spectroscopy Simulation Script - Direct Parameter Version

This script computes 2D electronic spectroscopy data using parallel processing
and saves results in pickle format. All parameters are defined directly in main().
"""

import os
import sys
import time

# Modern imports from reorganized package structure
from qspectro2d.simulation import (
    create_system_parameters,
    run_2d_simulation,
    get_max_workers,
    print_simulation_header,
    print_simulation_summary,
)
from qspectro2d.data import save_simulation_data
from config.paths import DATA_DIR


def main():
    """Main function to run the 2D spectroscopy simulation."""

    # =============================
    # SIMULATION PARAMETERS - MODIFY HERE
    # =============================

    ### Main system configuration
    N_atoms = 2  # Number of atoms (1 or 2)
    ODE_Solver = "BR"  # ODE solver type
    RWA_laser = True  # Use RWA for laser interaction
    t_det_max = 300  # Additional time buffer [fs]
    dt = 1  # Time step [fs]

    ### System-specific parameters
    if N_atoms == 1:
        pulse_fwhm = 15.0  # Pulse FWHM for single atom [fs]
    elif N_atoms == 2:
        pulse_fwhm = 5.0  # Pulse FWHM for two atoms [fs]
    else:
        raise ValueError(f"Unsupported number of atoms: {N_atoms}")

    ### 2D spectroscopy parameters
    T_wait = 0  # Number of T_wait values
    n_phases = 4  # Number of phases for phase cycling
    n_freqs = 1  # Number of frequencies for inhomogeneous broadening
    Delta_cm = 0  # Inhomogeneous broadening [cm⁻¹]
    envelope_type = "gaussian"  # Pulse envelope type
    E0 = 0.005  # Electric field amplitude

    # =============================
    # BUILD CONFIGURATION DICTIONARY
    # =============================
    data_config = {
        "simulation_type": "2d",  # Explicitly specify simulation type
        "N_atoms": N_atoms,
        "dt": dt,
        "t_det_max": t_det_max,
        "ODE_Solver": ODE_Solver,
        "pulse_fwhm": pulse_fwhm,
        "RWA_laser": RWA_laser,
        "T_wait": T_wait,
        "n_phases": n_phases,
        "n_freqs": n_freqs,
        "Delta_cm": Delta_cm,
        "envelope_type": envelope_type,
        "E0": E0,
    }

    # =============================
    # PRINT CONFIGURATION SUMMARY
    # =============================
    t_max = data_config["T_wait"] + 2 * data_config["t_det_max"]
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
    print_simulation_header(data_config, max_workers, "2d")

    # Create system parameters
    system = create_system_parameters(data_config)
    print(f"System configuration:")
    system.summary()

    # Run simulation (returns standardized payload)
    tau_coh, t_det, data = run_2d_simulation(data_config, system, max_workers)

    # Save data using the unified save function    print("\nSaving simulation data...")
    data_path, info_path = save_simulation_data(
        system=system, data_config=data_config, data=data, axs2=tau_coh, axs1=t_det
    )

    # Print simulation summary
    elapsed_time = time.time() - start_time
    print_simulation_summary(
        elapsed_time, data, data_path, "2d"
    )  # Print the paths for feed-forward to plotting script
    # For shell scripts, we need absolute paths for file existence checks
    from config.paths import DATA_DIR

    abs_data_path = DATA_DIR / data_path
    abs_info_path = DATA_DIR / info_path

    print(f"\n{'='*60}")
    print("DATA SAVED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Data file: {abs_data_path}")
    print(f"Info file: {abs_info_path}")
    print(f"\n🎯 To plot this data, run:")
    print(
        f'python plot_2D_datas.py --data-path "{data_path}" --info-path "{info_path}"'
    )
    print(f"{'='*60}")

    return data_path, info_path


if __name__ == "__main__":
    main()
