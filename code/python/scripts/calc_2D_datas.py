"""
2D Electronic Spectroscopy Simulation Script - Direct Parameter Version

This script computes 2D electronic spectroscopy data using parallel processing
and saves results in pickle format. All parameters are defined directly in main().
"""

import time
from qspectro2d.spectroscopy import (
    create_system_parameters,
    run_2d_simulation,
    get_max_workers,
    print_simulation_header,
    print_simulation_summary,
)
from qspectro2d.data import save_simulation_data


def main():
    """Main function to run the 2D spectroscopy simulation."""

    # =============================
    # SIMULATION PARAMETERS - MODIFY HERE
    # =============================

    ### Main system configuration
    N_atoms = 1  # Number of atoms (1 or 2)
    ODE_Solver = "BR"  # ODE solver type
    RWA_laser = True  # Use RWA for laser interaction
    t_det_max = 4  # Additional time buffer [fs]
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
    n_phases = 2  # Number of phases for phase cycling
    n_freqs = 1  # Number of frequencies for inhomogeneous broadening
    Delta_cm = 0  # Inhomogeneous broadening [cm⁻¹]
    pulse_type = "gaussian"  # Pulse envelope type
    E0 = 0.005  # Electric field amplitude

    # =============================
    # BUILD CONFIGURATION DICTIONARY
    # =============================
    info_config = {
        "simulation_type": "2d",  # Explicitly specify simulation type
        "N_atoms": N_atoms,
        "dt": dt,
        "t_det_max": t_det_max,
        "ODE_Solver": ODE_Solver,
        "pulse_fwhm": pulse_fwhm,
        "RWA_laser": RWA_laser,
        "t_wait": T_wait,
        "n_phases": n_phases,
        "n_freqs": n_freqs,
        "Delta_cm": Delta_cm,
        "pulse_type": pulse_type,
        "E0": E0,
    }

    # =============================
    # PRINT CONFIGURATION SUMMARY
    # =============================
    t_max = info_config["t_wait"] + 2 * info_config["t_det_max"]
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
    print_simulation_header(info_config, max_workers)

    # Create system parameters
    system = create_system_parameters(info_config)
    print(f"System configuration:")
    system.summary()

    # Run simulation (returns standardized payload)
    tau_coh, t_det, data = run_2d_simulation(info_config, system, max_workers)

    # Save data using the unified save function    print("\nSaving simulation data...")
    rel_path = save_simulation_data(
        system=system, info_config=info_config, data=data, axis2=tau_coh, axis1=t_det
    )

    # Print simulation summary
    elapsed_time = time.time() - start_time
    print_simulation_summary(
        elapsed_time, data, rel_path, "2d"
    )  # Print the paths for feed-forward to plotting script
    # For shell scripts, we need absolute paths for file existence checks

    print(f"{'='*60}")
    print(f"\n🎯 To plot this data, run:")
    print(f'python plot_datas.py --rel_path "{rel_path} --dim 2"')
    print(f"{'='*60}")

    return rel_path


if __name__ == "__main__":
    main()
