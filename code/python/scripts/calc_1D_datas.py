"""
1D Electronic Spectroscopy Simulation Script - Direct Parameter Version

This script computes 1D electronic spectroscopy data using parallel processing
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
    """Main function to run the 1D spectroscopy simulation."""

    # =============================
    # SIMULATION PARAMETERS - MODIFY HERE
    # =============================

    ### Main system configuration
    N_atoms = 1  # Number of atoms
    ODE_Solver = "Paper_eqs"  # ODE solver type ("BR" or "Paper_eqs")

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
    n_phases = 4  # Number of phases for phase cycling
    n_freqs = 1  # Number of frequencies for inhomogeneous broadening
    Delta_cm = 200  # Inhomogeneous broadening [cm⁻¹]
    envelope_type = "gaussian"  # Pulse envelope type ('cos2' or 'gaussian')
    E0 = 0.005  # Electric field amplitude
    RWA_laser = True  # Use RWA for laser interaction

    ### Generate dynamic output path
    output_subdir = f"{T_wait:.0f}_{tau_coh:.0f}fs"

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
        "output_subdir": output_subdir,
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
    print(f"  Output: {output_subdir}")
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
    payload = run_simulation(config, system, "1d")

    # Save data using the new workflow
    print("\nSaving simulation data...")
    relative_dir = save_data_with_unique_path(payload, config, system)

    # Print simulation summary
    elapsed_time = time.time() - start_time
    print_simulation_summary(elapsed_time, payload["data"], relative_dir, "1d")

    # Print the relative path for feed-forward to plotting script
    print(f"SAVED_DATA_PATH:{relative_dir}")


if __name__ == "__main__":
    main()
