"""
1D Electronic Spectroscopy Simulation Script - Direct Parameter Version

This script computes 1D electronic spectroscopy data using parallel processing
and saves results in pickle format. All parameters are defined directly in main().
"""

# =============================
# IMPORTS
# =============================
from common_fcts import run_1d_simulation_with_config


# =============================
# MAIN FUNCTION
# =============================
def main():
    """Main function to run the 1D spectroscopy simulation."""

    # =============================
    # SIMULATION PARAMETERS - MODIFY HERE
    # =============================

    ### Main system configuration
    N_atoms = 1  # Number of atoms
    ODE_Solver = "BR"  # ODE solver type ("BR" or "Paper_eqs")

    ### Time parameters
    tau_coh = 300.0  # Coherence time [fs]
    T_wait = 1000.0  # Waiting time [fs]
    t_det_max = 600.0  # Additional time buffer [fs]
    dt = 2.0  # Time step [fs]

    ### Spectroscopy parameters
    n_phases = 4  # Number of phases for phase cycling
    n_freqs = 1  # Number of frequencies for inhomogeneous broadening
    Delta_cm = 200  # Inhomogeneous broadening [cm⁻¹]
    envelope_type = "gaussian"  # Pulse envelope type ('cos2' or 'gaussian')
    E0 = 0.005  # Electric field amplitude
    pulse_fwhm = 15.0  # Pulse FWHM for Gaussian envelope [fs]
    RWA_laser = True  # Use RWA for laser interaction

    ### Generate dynamic output path
    output_subdir = f"N_{N_atoms}/{ODE_Solver.lower()}_tau{tau_coh:.0f}_T{T_wait:.0f}"

    # =============================
    # BUILD CONFIGURATION DICTIONARY
    # =============================
    config = {
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
    run_1d_simulation_with_config(config)

    # Print the output subdirectory for SLURM script to capture
    print(f"OUTPUT_SUBDIR:{output_subdir}")

    return output_subdir


if __name__ == "__main__":
    main()
