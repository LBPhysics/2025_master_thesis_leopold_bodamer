"""
2D Electronic Spectroscopy Simulation Script

This script computes 2D electronic spectroscopy data using parallel processing
and saves results in pickle format for simple data storage.
"""

# =============================
# IMPORTS
# =============================
import numpy as np
import psutil
import time
import pickle
import copy
from pathlib import Path
from datetime import datetime

### Project-specific imports
from src.spectroscopy.calculations import (
    parallel_compute_2d_E_with_inhomogenity,
    check_the_solver,
)
from src.core.system_parameters import SystemParameters
from config.paths import DATA_DIR


# =============================
# SIMULATION CONFIGURATION
# =============================
def get_simulation_config():
    """Get default simulation configuration for 2D spectroscopy."""
    # Default number of atoms
    N_atoms = 2

    # =============================
    # GENERAL SIMULATION PARAMETERS
    # =============================
    config = {}

    # Time and output directory configuration
    time_config = {
        "t_max": 1,  # Maximum time [fs]
        "dt": 0.1,  # Time step [fs]
    }
    config.update(time_config)

    # =============================
    # ATOM-SPECIFIC CONFIGURATION
    # =============================
    if N_atoms == 1:
        system_config = {
            "N_atoms": 1,  # Number of atoms in the system
            "pulse_FWHM": 15.0,  # Pulse FWHM for Gaussian envelope [fs]
            "output_subdir": "2d_spectroscopy/N_1/600fs",
        }
    elif N_atoms == 2:
        system_config = {
            "N_atoms": 2,  # Number of atoms in the system
            "pulse_FWHM": 5.0,  # Pulse FWHM for Gaussian envelope [fs]
            "output_subdir": "2d_spectroscopy/N_2/600fs",
        }
    config.update(system_config)

    # =============================
    # 2D SPECTROSCOPY PARAMETERS
    # =============================
    spectroscopy_config = {
        "ODE_Solver": "Paper_eqs",  # ODE solver type
        "RWA_laser": True,  # Use RWA for laser interaction
        "T_wait_max": time_config["t_max"] / 2,  # Maximum waiting time [fs]
        "n_times_T": 1,  # Number of T_wait values
        "n_phases": 4,  # Number of phases for phase cycling
        "n_freqs": 1,  # Number of frequencies for inhomogeneous broadening
        "Delta_cm": 200,  # Inhomogeneous broadening [cm⁻¹]
        "envelope_type": "gaussian",  # Pulse envelope type
        "E0": 0.005,  # Electric field amplitude
    }
    config.update(spectroscopy_config)

    return config


# =============================
# DATA SAVING FUNCTIONS
# =============================
def generate_unique_filename(output_dir: Path, config: dict, system) -> Path:
    """Generate a unique filename for 2D polarization data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = (
        f"2d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
        f"T{config['n_times_T']}_ph{config['n_phases']}_"
        f"freq{config['n_freqs']}_{timestamp}.pkl"
    )

    save_path = output_dir / base_filename

    ### Ensure unique filename by adding counter if needed
    counter = 1
    while save_path.exists():
        name_with_counter = (
            f"2d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
            f"T{config['n_times_T']}_ph{config['n_phases']}_"
            f"freq{config['n_freqs']}_{timestamp}_{counter}.pkl"
        )
        save_path = output_dir / name_with_counter
        counter += 1

    return save_path


def save_2d_data(
    two_d_datas: list, times: np.ndarray, times_T: np.ndarray, config: dict, system
) -> Path:
    """Save 2D polarization simulation data to a pickle file."""
    ### Create output directory
    output_dir = DATA_DIR / config["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Generate unique filename
    save_path = generate_unique_filename(output_dir, config, system)

    ### Package data into dictionary
    data = {
        "times": times,
        "times_T": times_T,
        "system": system,
        "two_d_datas": two_d_datas,
        "n_times_T": config["n_times_T"],
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
    """Main function to run the 2D spectroscopy simulation."""
    start_time = time.time()

    ### Get simulation configuration
    config = get_simulation_config()

    # Use SLURM environment variable if available, otherwise detect automatically
    import os

    # take the max number of these 2:
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    local_cpus = psutil.cpu_count(logical=True)
    max_workers = max(slurm_cpus, local_cpus)
    if max_workers < 1:
        max_workers = 1

    # =============================
    # SIMULATION PARAMETERS
    # =============================
    print("=" * 60)
    print("2D ELECTRONIC SPECTROSCOPY SIMULATION")
    print("=" * 60)
    print(f"Configuration:")
    print(
        f"  Parameters: #times_T={config['n_times_T']}, #phases={config['n_phases']}, #frequencies={config['n_freqs']}"
    )
    print(
        f"  Total combinations processed: {config['n_times_T'] * config['n_phases'] * config['n_phases'] * config['n_freqs']}"
    )
    print(f"  Parallel workers used: {max_workers}")
    print()

    # =============================
    # SYSTEM PARAMETERS
    # =============================
    system = SystemParameters(
        N_atoms=config["N_atoms"],
        ODE_Solver=config["ODE_Solver"],
        RWA_laser=config["RWA_laser"],
        t_max=config["t_max"],
        dt=config["dt"],
        Delta_cm=config["Delta_cm"],  #  if config["n_freqs"] > 1 else 0,
        envelope_type=config["envelope_type"],
        E0=config["E0"],
        pulse_FWHM=config["pulse_FWHM"] if "pulse_FWHM" in config else 100.0,
    )
    print(f"System configuration:")
    system.summary()

    ### Create time arrays
    FWHMs = system.FWHMs
    times = np.arange(-1 * FWHMs[0], system.t_max, system.dt)
    times_T = np.linspace(0, config["T_wait_max"], config["n_times_T"])

    # =============================
    # SOLVER VALIDATION
    # =============================
    test_system = copy.deepcopy(system)
    test_system.t_max = 10 * system.t_max
    test_system.dt = 10 * system.dt
    times_test = np.arange(-FWHMs[0], test_system.t_max, test_system.dt)

    try:
        _, time_cut = check_the_solver(times_test, test_system)
    except Exception as e:
        print(f"⚠️  WARNING: Solver validation failed: {e}")
        time_cut = 0

    # =============================
    # RUN SIMULATION
    # =============================
    print("Computing 2D polarization with parallel processing...")
    kwargs = {"plot_example": False, "time_cut": time_cut}

    try:
        two_d_datas = parallel_compute_2d_E_with_inhomogenity(
            n_freqs=config["n_freqs"],
            n_phases=config["n_phases"],
            times_T=times_T,
            times=times,
            system=system,
            max_workers=max_workers,
            **kwargs,
        )

        print("✅ Parallel computation completed successfully!")

    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise

    # =============================
    # SAVE DATA
    # =============================
    print("\\nSaving simulation data...")
    save_path = save_2d_data(
        two_d_datas=two_d_datas,
        times=times,
        times_T=times_T,
        config=config,
        system=system,
    )

    # =============================
    # SIMULATION SUMMARY
    # =============================
    elapsed_time = time.time() - start_time

    print("\\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Data shape: {two_d_datas[0].shape}")
    print(f"Data saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
