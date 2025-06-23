"""
2D Electronic Spectroscopy Simulation Script - Direct Parameter Version

This script computes 2D electronic spectroscopy data using parallel processing
and saves results in pickle format. All parameters are defined directly in main().
"""

from common_fcts import run_2d_simulation_with_config


def main():
    """Main function to run the 2D spectroscopy simulation."""

    # =============================
    # SIMULATION PARAMETERS - MODIFY HERE
    # =============================

    ### Main system configuration
    N_atoms = 1  # Number of atoms (1 or 2)
    t_max = 10  # Maximum time [fs]
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
    RWA_laser = True  # Use RWA for laser interaction
    T_wait_max = t_max / 2  # Maximum waiting time [fs]
    n_times_T = 1  # Number of T_wait values
    n_phases = 4  # Number of phases for phase cycling
    n_freqs = 1  # Number of frequencies for inhomogeneous broadening
    Delta_cm = 200  # Inhomogeneous broadening [cm⁻¹]
    envelope_type = "gaussian"  # Pulse envelope type
    E0 = 0.005  # Electric field amplitude

    ### Generate dynamic output path
    output_subdir = f"N_{N_atoms}/{ODE_Solver.lower()}/t_max_{t_max:.0f}fs"

    # =============================
    # BUILD CONFIGURATION DICTIONARY
    # =============================
    config = {
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
        "output_subdir": output_subdir,
    }

    # =============================
    # PRINT CONFIGURATION SUMMARY
    # =============================
    print(f"Running 2D spectroscopy simulation with:")
    print(f"  N_atoms: {N_atoms}")
    print(f"  Solver: {ODE_Solver}")
    print(f"  Time: {t_max} fs (dt = {dt} fs)")
    print(f"  Pulse FWHM: {pulse_fwhm} fs")
    print(f"  Output: {output_subdir}")
    print("")

    # =============================
    # RUN SIMULATION
    # =============================
    run_2d_simulation_with_config(config)


if __name__ == "__main__":
    main()
