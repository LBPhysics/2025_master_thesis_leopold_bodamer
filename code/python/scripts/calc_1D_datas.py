"""
1D Electronic Spectroscopy Simulation Script - Direct Parameter Version

This script computes 1D electronic spectroscopy data using parallel processing
and saves results in pickle format. All parameters are defined directly in main().
"""

import os
import sys
import time

# Change the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add the script's directory to sys.path to ensure imports work
sys.path.append(script_dir)

from common_fcts import (
    create_system_parameters,
    run_1d_simulation,
    get_max_workers,
    print_simulation_header,
    print_simulation_summary,
    save_simulation_data,
)
from config.paths import DATA_DIR


def main():
    """Main function to run the 1D spectroscopy simulation."""

    # =============================
    # SIMULATION PARAMETERS - MODIFY HERE
    # =============================

    ### Main system configuration
    N_atoms = 1  # Number of atoms
    ODE_Solver = "Paper_eqs"  # ODE solver type ("BR" or "Paper_eqs")
    RWA_laser = True  # Use RWA for laser interaction

    ### Time parameters
    tau_coh = 300.0  # Coherence time [fs]
    T_wait = 1000.0  # Waiting time [fs]
    t_det_max = 600.0  # Additional time buffer [fs]
    dt = 2.0  # Time step [fs]

    ### System-specific parameters
    if N_atoms == 1:
        pulse_fwhm = 15.0  # Pulse FWHM for single atom [fs]
    elif N_atoms == 2:
        pulse_fwhm = 5.0  # Pulse FWHM for two atoms [fs]
    else:
        raise ValueError(f"Unsupported number of atoms: {N_atoms}")

    ### Spectroscopy parameters
    n_phases = 2  # Number of phases for phase cycling
    n_freqs = 1  # Number of frequencies for inhomogeneous broadening
    Delta_cm = 200  # Inhomogeneous broadening [cm⁻¹]
    envelope_type = "gaussian"  # Pulse envelope type ('cos2' or 'gaussian')
    E0 = 0.005  # Electric field amplitude

    # =============================
    # BUILD CONFIGURATION DICTIONARY
    # =============================
    config = {
        "simulation_type": "1d",  # Explicitly specify simulation type
        "N_atoms": N_atoms,
        "ODE_Solver": ODE_Solver,
        "tau_coh": tau_coh,
        "T_wait": T_wait,
        "t_det_max": t_det_max,
        "dt": dt,
        "n_phases": n_phases,
        "n_freqs": n_freqs,
        "Delta_cm": Delta_cm,
        "envelope_type": envelope_type,
        "E0": E0,
        "pulse_fwhm": pulse_fwhm,
        "RWA_laser": RWA_laser,
    }

    # =============================
    # PRINT CONFIGURATION SUMMARY
    # =============================
    print(f"Running 1D spectroscopy simulation with:")
    print(f"  N_atoms: {N_atoms}")
    print(f"  Solver: {ODE_Solver}")
    print(f"  Times: τ_coh={tau_coh} fs, T_wait={T_wait} fs, dt={dt} fs")
    print(f"  Total time: {tau_coh + T_wait + t_det_max} fs")
    print(f"  Pulse FWHM: {pulse_fwhm} fs")
    print("")

    # =============================
    # RUN SIMULATION
    # =============================
    start_time = time.time()

    # Get parallel processing configuration
    max_workers = get_max_workers()

    # Print simulation header
    print_simulation_header(config, max_workers, "1d")

    # Create system parameters
    system = create_system_parameters(config)
    print(f"System configuration:")
    system.summary()

    # Run simulation (returns standardized payload)
    t_det, data = run_1d_simulation(config, system, max_workers)

    # Save data using the new workflow
    print("\nSaving simulation data...")
    data_path, info_path = save_simulation_data(
        system=system, config=config, data=data, axs1=t_det
    )  # Print simulation summary
    elapsed_time = time.time() - start_time
    print_simulation_summary(elapsed_time, data, data_path, "1d")

    # Print the paths for feed-forward to plotting script
    print(f"\n{'='*60}")
    print("DATA SAVED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Data file: {data_path}")
    print(f"Info file: {info_path}")
    print(f"\n🎯 To plot this data, run:")
    print(
        f'python plot_1D_datas.py --data-path "{data_path}" --info-path "{info_path}"'
    )
    print(f"{'='*60}")

    return data_path, info_path


if __name__ == "__main__":
    main()
