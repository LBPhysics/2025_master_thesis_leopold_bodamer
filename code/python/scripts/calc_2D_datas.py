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
    parallel_compute_2d_polarization_with_inhomogenity,
    check_the_solver,
)
from src.spectroscopy.inhomogenity import sample_from_sigma
from src.core.system_parameters import SystemParameters
from config.paths import DATA_DIR


# =============================
# SIMULATION CONFIGURATION
# =============================
def get_simulation_config():
    """Get default simulation configuration for 2D spectroscopy."""
    return {
        "N_atoms": 2,  # Number of atoms in the system
        "n_times_T": 1,  # Number of T_wait values
        "n_phases": 4,  # Number of phases for phase cycling
        "n_freqs": 1,  # Number of frequencies for inhomogeneous broadening
        "t_max": 5.0,  # Maximum time [fs]
        "dt": 0.1,  # Time step [fs]
        "T_wait_max": 10.0,  # Maximum waiting time [fs]
        "Delta_cm": 200,  # Inhomogeneous broadening [cm⁻¹]
        "envelope_type": "gaussian",
        "output_subdir": "2d_spectroscopy",
    }


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
    max_workers = psutil.cpu_count(logical=True)

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
        ODE_Solver="Paper_eqs",
        RWA_laser=True,
        t_max=config["t_max"],
        dt=config["dt"],
        Delta_cm=config["Delta_cm"] if config["n_freqs"] > 1 else 0,
        envelope_type=config["envelope_type"],
    )

    print(f"System configuration:")
    system.summary()

    ### Create time arrays
    FWHMs = system.FWHMs
    times = np.arange(-FWHMs[0], system.t_max, system.dt)
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
        two_d_datas = parallel_compute_2d_polarization_with_inhomogenity(
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
    total_data_points = sum(
        data.size if data is not None else 0 for data in two_d_datas
    )
    file_size_mb = save_path.stat().st_size / (1024**2) if save_path.exists() else 0

    print("\\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Saved: {save_path.name} ({file_size_mb:.2f} MB)")
    print(f"Data saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
