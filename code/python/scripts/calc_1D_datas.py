"""
1D Electronic Spectroscopy Simulation Script

This script computes 1D electronic spectroscopy data using parallel processing
and saves results in pickle format for simple data storage.
"""

# =============================
# IMPORTS
# =============================
import numpy as np
import psutil
import time
import pickle
from pathlib import Path
from datetime import datetime

### Project-specific imports
from qspectro2d.spectroscopy.calculations import (
    parallel_compute_1d_E_with_inhomogenity,
)
from qspectro2d.core.system_parameters import SystemParameters
from config.paths import DATA_DIR


# =============================
# SIMULATION CONFIGURATION
# =============================
def get_simulation_config():
    """Get default simulation configuration for 1D spectroscopy."""
    return {
        "N_atoms": 1,  # Number of atoms in the system
        "n_phases": 4,  # Number of phases for phase cycling
        "n_freqs": 1,  # Number of frequencies for inhomogeneous broadening
        "tau_coh": 300.0,  # Coherence time [fs]
        "T_wait": 1000.0,  # Waiting time [fs]
        "t_det_max": 600.0,  # Additional time buffer [fs]
        "dt": 2.0,  # Time step [fs]
        "Delta_cm": 200,  # Inhomogeneous broadening [cm⁻¹]
        "envelope_type": "gaussian",  # 'cos2' or 'gaussian'
        "output_subdir": "1d_spectroscopy/inhomogeneity",
        "E0": 0.005,
        "ODE_Solver": "BR",  # ODE solver type
        "pulse_fwhm": 15.0,  # Pulse fwhm for Gaussian envelope [fs]
        "RWA_laser": True,  # Use RWA for laser interaction
    }


# =============================
# DATA SAVING FUNCTIONS
# =============================
def generate_unique_filename(output_dir: Path, config: dict, system) -> Path:
    """Generate a unique filename for 1D polarization data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = (
        f"1d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
        f"ph{config['n_phases']}_freq{config['n_freqs']}_{timestamp}.pkl"
    )

    save_path = output_dir / base_filename

    ### Ensure unique filename by adding counter if needed
    counter = 1
    while save_path.exists():
        name_with_counter = (
            f"1d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
            f"ph{config['n_phases']}_freq{config['n_freqs']}_{timestamp}_{counter}.pkl"
        )
        save_path = output_dir / name_with_counter
        counter += 1

    return save_path


def save_1d_data(
    t_det_vals: np.ndarray,
    data_avg: np.ndarray,
    config: dict,
    system,
) -> Path:
    """Save 1D polarization simulation data to a pickle file."""
    ### Create output directory
    output_dir = DATA_DIR / config["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Generate unique filename
    save_path = generate_unique_filename(output_dir, config, system)

    ### Package data into dictionary
    data = {
        "t_det_vals": t_det_vals,
        "data_avg": data_avg,
        "tau_coh": config["tau_coh"],
        "T_wait": config["T_wait"],
        "system": system,
        "n_phases": config["n_phases"],
        "n_freqs": config["n_freqs"],
    }

    ### Save as pickle file
    try:
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Data saved successfully to: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ ERROR: Failed to save data: {e}")
        raise


# =============================
# MAIN SIMULATION FUNCTION
# =============================
def main():
    """Main function to run the 1D spectroscopy simulation."""
    start_time = time.time()

    ### Get simulation configuration
    config = get_simulation_config()
    max_workers = psutil.cpu_count(logical=True)

    # =============================
    # SIMULATION PARAMETERS
    # =============================
    print("=" * 60)
    print("1D ELECTRONIC SPECTROSCOPY SIMULATION")
    print("=" * 60)
    print(f"Configuration:")
    print(
        f"  Parameters: #phases={config['n_phases']}, #frequencies={config['n_freqs']}"
    )
    print(
        f"  Total combinations processed: {config['n_phases'] * config['n_phases'] * config['n_freqs']}"
    )
    print(f"  Parallel workers used: {max_workers}")
    print()

    # =============================
    # SYSTEM PARAMETERS
    # =============================
    system = SystemParameters(
        N_atoms=config["N_atoms"],
        ODE_Solver=config["ODE_Solver"],
        t_max=config["tau_coh"] + config["T_wait"] + config["t_det_max"],
        dt=config["dt"],
        Delta_cm=config["Delta_cm"],  #  if config["n_freqs"] > 1 else 0,
        envelope_type=config["envelope_type"],
        pulse_fwhm=config["pulse_fwhm"] if "pulse_fwhm" in config else 100.0,
        E0=config["E0"],
        RWA_laser=config["RWA_laser"],
    )

    print(f"System configuration:")
    system.summary()

    ### Create time arrays
    fwhms = system.fwhms
    times = np.arange(-2 * fwhms[0], system.t_max, system.dt)

    # =============================
    # RUN SIMULATION
    # =============================
    print("Computing 1D polarization with parallel processing...")
    try:
        t_det_vals, data_avg = parallel_compute_1d_E_with_inhomogenity(
            n_freqs=config["n_freqs"],
            n_phases=config["n_phases"],
            tau_coh=config["tau_coh"],
            T_wait=config["T_wait"],
            times=times,
            system=system,
            max_workers=max_workers,
        )

        print("✅ Parallel computation completed successfully!")

    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise

    # =============================
    # SAVE DATA
    # =============================
    print("\nSaving simulation data...")
    save_path = save_1d_data(
        t_det_vals=t_det_vals,
        data_avg=data_avg,
        config=config,
        system=system,
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
