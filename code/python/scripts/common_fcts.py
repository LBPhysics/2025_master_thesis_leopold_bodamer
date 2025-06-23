"""
Common functions for Electronic Spectroscopy Simulations

This module contains shared functionality for 1D and 2D spectroscopy
calculation scripts, including data saving, validation, and common utilities.
"""

# =============================
# IMPORTS
# =============================
import numpy as np
import psutil
import time
import pickle
import copy
import os
from pathlib import Path
from datetime import datetime

### Project-specific imports
from qspectro2d.spectroscopy.calculations import (
    parallel_compute_2d_E_with_inhomogenity,
    check_the_solver,
)
from qspectro2d.core.system_parameters import SystemParameters
from config.paths import DATA_DIR


# =============================
# GENERALIZED DATA SAVING FUNCTIONS
# =============================
def generate_unique_filename(
    output_dir: Path, config: dict, system, simulation_type: str = "spectroscopy"
) -> Path:
    """Generate a unique filename for spectroscopy data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build filename components based on simulation type
    if simulation_type == "1d":
        base_filename = (
            f"1d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
            f"ph{config['n_phases']}_freq{config['n_freqs']}_{timestamp}.pkl"
        )
    elif simulation_type == "2d":
        base_filename = (
            f"2d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
            f"T{config.get('n_times_T', 1)}_ph{config['n_phases']}_"
            f"freq{config['n_freqs']}_{timestamp}.pkl"
        )
    else:
        base_filename = (
            f"{simulation_type}_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
            f"{timestamp}.pkl"
        )

    save_path = output_dir / base_filename

    ### Ensure unique filename by adding counter if needed
    counter = 1
    while save_path.exists():
        name_parts = base_filename.split(".pkl")[0]
        name_with_counter = f"{name_parts}_{counter}.pkl"
        save_path = output_dir / name_with_counter
        counter += 1

    return save_path


def save_spectroscopy_data(
    data_dict: dict, config: dict, system, simulation_type: str = "spectroscopy"
) -> Path:
    """Save spectroscopy simulation data to a pickle file."""
    ### Create output directory
    output_dir = DATA_DIR / config["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Generate unique filename
    save_path = generate_unique_filename(output_dir, config, system, simulation_type)

    ### Add common metadata to data
    data_dict.update(
        {
            "system": system,
            "n_phases": config["n_phases"],
            "n_freqs": config["n_freqs"],
            "simulation_type": simulation_type,
            "timestamp": datetime.now().isoformat(),
        }
    )

    ### Save as pickle file
    try:
        with open(save_path, "wb") as f:
            pickle.dump(data_dict, f)
        print(f"✅ Data saved successfully to: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ ERROR: Failed to save data: {e}")
        raise


# =============================
# WRAPPER FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================
def save_1d_data(
    t_det_vals: np.ndarray, data_avg: np.ndarray, config: dict, system
) -> Path:
    """Save 1D polarization simulation data - wrapper for generalized function."""
    data_dict = {
        "t_det_vals": t_det_vals,
        "data_avg": data_avg,
        "tau_coh": config["tau_coh"],
        "T_wait": config["T_wait"],
    }
    return save_spectroscopy_data(data_dict, config, system, "1d")


def save_2d_data(
    two_d_datas: list, times: np.ndarray, times_T: np.ndarray, config: dict, system
) -> Path:
    """Save 2D polarization simulation data - wrapper for generalized function."""
    data_dict = {
        "times": times,
        "times_T": times_T,
        "two_d_datas": two_d_datas,
        "n_times_T": config["n_times_T"],
    }
    return save_spectroscopy_data(data_dict, config, system, "2d")


# =============================
# SIMULATION UTILITIES
# =============================
def get_max_workers() -> int:
    """Get the maximum number of workers for parallel processing."""
    # Use SLURM environment variable if available, otherwise detect automatically
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    local_cpus = psutil.cpu_count(logical=True)
    max_workers = max(slurm_cpus, local_cpus)
    if max_workers < 1:
        max_workers = 1
    return max_workers


def validate_solver(system: SystemParameters, times: np.ndarray) -> int:
    """Validate the ODE solver and return time_cut parameter."""
    fwhms = system.fwhms
    test_system = copy.deepcopy(system)
    test_system.t_max = 10 * system.t_max
    test_system.dt = 10 * system.dt
    times_test = np.arange(-fwhms[0], test_system.t_max, test_system.dt)

    try:
        _, time_cut = check_the_solver(times_test, test_system)
        return time_cut
    except Exception as e:
        print(f"⚠️  WARNING: Solver validation failed: {e}")
        return 0


def create_system_parameters(config: dict) -> SystemParameters:
    """Create SystemParameters object from configuration."""
    # Handle different time configurations for 1D vs 2D
    if "tau_coh" in config and "T_wait" in config:
        # 1D configuration
        t_max = config["tau_coh"] + config["T_wait"] + config["t_det_max"]
    else:
        # 2D configuration
        t_max = config["t_max"]

    return SystemParameters(
        N_atoms=config["N_atoms"],
        ODE_Solver=config["ODE_Solver"],
        RWA_laser=config["RWA_laser"],
        t_max=t_max,
        dt=config["dt"],
        Delta_cm=config["Delta_cm"],
        envelope_type=config["envelope_type"],
        E0=config["E0"],
        pulse_fwhm=config.get("pulse_fwhm", 100.0),
    )


# =============================
# GENERALIZED SIMULATION FUNCTIONS
# =============================
def run_simulation(
    config: dict, system: SystemParameters, simulation_type: str
) -> tuple:
    """Run spectroscopy simulation based on type."""
    max_workers = get_max_workers()

    if simulation_type == "1d":
        return run_1d_simulation(config, system, max_workers)
    elif simulation_type == "2d":
        return run_2d_simulation(config, system, max_workers)
    else:
        raise ValueError(f"Unsupported simulation type: {simulation_type}")


def run_1d_simulation(
    config: dict, system: SystemParameters, max_workers: int
) -> tuple:
    """Run 1D spectroscopy simulation."""
    from qspectro2d.spectroscopy.calculations import (
        parallel_compute_1d_E_with_inhomogenity,
    )

    ### Create time arrays
    fwhms = system.fwhms
    times = np.arange(-2 * fwhms[0], system.t_max, system.dt)

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
        return t_det_vals, data_avg
    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise


def run_2d_simulation(config: dict, system: SystemParameters, max_workers: int) -> list:
    """Run 2D spectroscopy simulation."""
    ### Create time arrays
    fwhms = system.fwhms
    times = np.arange(-1 * fwhms[0], system.t_max, system.dt)
    times_T = np.linspace(0, config["T_wait_max"], config["n_times_T"])

    ### Validate solver
    time_cut = validate_solver(system, times)

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
        return two_d_datas, times, times_T
    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise


# =============================
# GENERALIZED PRINTING FUNCTIONS
# =============================
def print_simulation_header(config: dict, max_workers: int, simulation_type: str):
    """Print simulation header with configuration info."""
    title = f"{simulation_type.upper()} ELECTRONIC SPECTROSCOPY SIMULATION"
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"Configuration:")

    if simulation_type == "1d":
        print(
            f"  Parameters: #phases={config['n_phases']}, #frequencies={config['n_freqs']}"
        )
        print(f"  Times: τ_coh={config['tau_coh']} fs, T_wait={config['T_wait']} fs")
    elif simulation_type == "2d":
        print(
            f"  Parameters: #times_T={config['n_times_T']}, #phases={config['n_phases']}, #frequencies={config['n_freqs']}"
        )
        print(f"  Time: t_max={config['t_max']} fs")

    print(
        f"  Total combinations: {config['n_phases'] * config['n_phases'] * config['n_freqs']}"
    )
    print(f"  Parallel workers: {max_workers}")
    print()


def print_simulation_summary(
    elapsed_time: float, result_data, save_path: Path, simulation_type: str
):
    """Print simulation completion summary."""
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    if simulation_type == "1d":
        print(f"Data shape: {result_data.shape}")
    elif simulation_type == "2d":
        print(f"Data shape: {result_data[0].shape}")

    print(f"Data saved to: {save_path}")
    print("=" * 60)


# =============================
# MAIN SIMULATION RUNNERS
# =============================
def run_simulation_with_config(config: dict, simulation_type: str):
    """Main simulation runner that takes a configuration dictionary."""
    start_time = time.time()

    ### Get parallel processing configuration
    max_workers = get_max_workers()

    ### Print simulation header
    print_simulation_header(config, max_workers, simulation_type)

    # =============================
    # SYSTEM PARAMETERS
    # =============================
    system = create_system_parameters(config)
    print(f"System configuration:")
    system.summary()

    # =============================
    # RUN SIMULATION
    # =============================
    result = run_simulation(config, system, simulation_type)

    # =============================
    # SAVE DATA
    # =============================
    print("\nSaving simulation data...")

    if simulation_type == "1d":
        t_det_vals, data_avg = result
        save_path = save_1d_data(t_det_vals, data_avg, config, system)
        result_data = data_avg
    elif simulation_type == "2d":
        two_d_datas, times, times_T = result
        save_path = save_2d_data(two_d_datas, times, times_T, config, system)
        result_data = two_d_datas

    # =============================
    # SIMULATION SUMMARY
    # =============================
    elapsed_time = time.time() - start_time
    print_simulation_summary(elapsed_time, result_data, save_path, simulation_type)

    return result


# =============================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================
def run_1d_simulation_with_config(config: dict):
    """Run 1D simulation - wrapper for generalized function."""
    return run_simulation_with_config(config, "1d")


def run_2d_simulation_with_config(config: dict):
    """Run 2D simulation - wrapper for generalized function."""
    return run_simulation_with_config(config, "2d")


# NOW FOR PLOTTING FUNCTIONS

# =============================
# IMPORTS
# =============================
import numpy as np
import psutil
import time
import pickle
import copy
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

### Project-specific imports
from qspectro2d.spectroscopy.calculations import (
    parallel_compute_2d_E_with_inhomogenity,
    check_the_solver,
)
from qspectro2d.core.system_parameters import SystemParameters
from config.paths import DATA_DIR


# =============================
# GENERALIZED PLOTTING FUNCTIONS
# =============================
def find_latest_file(data_subdir: str, file_pattern: str = "*.pkl") -> Optional[Path]:
    """Find the most recent file matching pattern in a data subdirectory.

    Args:
        data_subdir: Subdirectory within DATA_DIR (e.g., '2d_spectroscopy/N_1/paper_eqs/100fs')
        file_pattern: Glob pattern for file matching

    Returns:
        Path to the latest file or None if not found
    """
    data_dir = DATA_DIR / data_subdir

    if not data_dir.exists():
        print(f"❌ Data directory does not exist: {data_dir}")
        return None

    # Look for files matching the pattern
    files = list(data_dir.glob(file_pattern))

    # For 2D data, also look for compressed files
    if "2d_spectroscopy" in data_subdir:
        files.extend(list(data_dir.glob("*.pkl.gz")))

    if not files:
        print(f"❌ No files matching '{file_pattern}' found in {data_dir}")
        return None

    # Get the most recent file
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    print(f"✅ Found latest file: {latest_file.name}")
    return latest_file


def load_pickle_file(filepath: Path) -> Optional[dict]:
    """Load data from pickle file (supports both .pkl and .pkl.gz).

    Args:
        filepath: Path to the pickle file

    Returns:
        Dictionary containing the loaded data or None if error
    """
    print(f"Loading data from: {filepath.name}")

    try:
        if filepath.suffix == ".gz":
            # Handle compressed pickle files
            import gzip

            with gzip.open(filepath, "rb") as f:
                data = pickle.load(f)
        else:
            # Handle regular pickle files
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        print(f"✅ Data loaded successfully!")
        return data

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def create_output_directory(subdir: str) -> Path:
    """Create output directory for figures.

    Args:
        subdir: Subdirectory name within FIGURES_DIR

    Returns:
        Path to the created output directory
    """
    from config.paths import FIGURES_DIR
    import os

    output_dir = FIGURES_DIR / subdir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_spectroscopy_data(config: dict, simulation_type: str) -> None:
    """Generalized plotting function for spectroscopy data.

    Args:
        config: Dictionary containing plotting configuration
        simulation_type: "1d" or "2d"
    """
    print(f"# =============================")
    print(f"# LOAD AND PLOT {simulation_type.upper()} SPECTROSCOPY DATA")
    print(f"# =============================")

    ### Find and load data
    file_path = find_latest_file(config["data_subdir"], config["file_pattern"])

    if file_path is None:
        print(
            f"   Please run the {simulation_type.upper()} spectroscopy calculation first to generate data."
        )
        return

    loaded_data = load_pickle_file(file_path)

    if loaded_data is None:
        print(f"❌ Failed to load {simulation_type.upper()} spectroscopy data.")
        return

    ### Create output directory
    output_dir = create_output_directory(config["output_subdir"])

    ### Route to specific plotting function
    if simulation_type == "1d":
        _plot_1d_data(loaded_data, config, output_dir)
    elif simulation_type == "2d":
        _plot_2d_data(loaded_data, config, output_dir)
    else:
        raise ValueError(f"Unsupported simulation type: {simulation_type}")

    print(f"🎯 All {simulation_type.upper()} plots saved to: {output_dir}")


def _plot_1d_data(data: dict, config: dict, output_dir: Path) -> None:
    """Plot 1D spectroscopy data."""
    from qspectro2d.visualization.plotting import (
        Plot_fixed_tau_T,
        Plot_1d_frequency_spectrum,
    )
    from qspectro2d.spectroscopy.post_processing import compute_1d_fft_wavenumber
    import matplotlib.pyplot as plt
    import gc
    import psutil

    # Extract data
    t_det_vals = data["t_det_vals"]
    data_avg = data["data_avg"]
    system = data["system"]

    print(f"✅ Data loaded with shape: {data_avg.shape}")
    print(f"   Time points: {len(t_det_vals)}")
    print(f"   Time range: {t_det_vals[0]:.1f} to {t_det_vals[-1]:.1f} fs")

    ### Plot time domain data
    if config.get("plot_time_domain", True):
        print("📊 Plotting time domain data...")
        try:
            Plot_fixed_tau_T(
                t_det_vals=t_det_vals,
                data_avg=data_avg,
                tau_coh=data.get("tau_coh", 300),
                T_wait=data.get("T_wait", 1000),
                system=system,
                plot_types=config.get("plot_types", ["real", "imag", "abs", "phase"]),
                output_dir=output_dir,
            )
            print("✅ Time domain plots completed!")
        except Exception as e:
            print(f"❌ Error in time domain plotting: {e}")

    ### Plot frequency domain data
    if config.get("plot_frequency_domain", True):
        print("📊 Plotting frequency domain data...")
        try:
            frequencies, data_fft = compute_1d_fft_wavenumber(
                t_det_vals, data_avg, system
            )
            Plot_1d_frequency_spectrum(
                frequencies=frequencies,
                data_fft=data_fft,
                system=system,
                plot_types=config.get("plot_types", ["real", "imag", "abs", "phase"]),
                output_dir=output_dir,
            )
            print("✅ Frequency domain plots completed!")
        except Exception as e:
            print(f"❌ Error in frequency domain plotting: {e}")

    # Clean up memory
    plt.close("all")
    gc.collect()


def _plot_2d_data(data: dict, config: dict, output_dir: Path) -> None:
    """Plot 2D spectroscopy data."""
    from qspectro2d.spectroscopy.post_processing import extend_and_plot_results
    from qspectro2d.spectroscopy.calculations import get_tau_cohs_and_t_dets_for_T_wait
    from qspectro2d.visualization.plotting import Plot_2d_El_field
    import matplotlib.pyplot as plt
    import gc
    import psutil

    # Extract data
    times = data["times"]
    times_T = data["times_T"]
    two_d_datas = data["two_d_datas"]
    system_data = data["system"]

    print(f"✅ 2D data loaded successfully!")
    print(f"   Times shape: {times.shape}")
    print(f"   Times_T shape: {times_T.shape}")
    print(f"   Data shape: {two_d_datas[0].shape}")

    ### Plot time domain data if requested
    if config.get("plot_time_domain", False):
        print("📊 Plotting time domain data...")
        try:
            tau_cohs, t_dets = get_tau_cohs_and_t_dets_for_T_wait(times, times_T[0])
            Plot_2d_El_field(
                tau_cohs=tau_cohs,
                t_dets=t_dets,
                two_d_datas=two_d_datas,
                output_dir=output_dir,
            )
            print("✅ Time domain plots completed!")
        except Exception as e:
            print(f"❌ Error in time domain plotting: {e}")

    ### Plot frequency domain data
    extend_for = config.get("extend_for", (1, 2.3))
    section = config.get("section", (1.5, 1.7, 1.5, 1.7))
    plot_types = config.get("plot_types", ["imag", "abs", "real", "phase"])

    print(
        f"📊 Plotting frequency domain data with extend_for={extend_for}, section={section}"
    )

    for plot_type in plot_types:
        print(f"   Generating {plot_type} plot...")

        plot_args = {
            "plot_type": plot_type,
            "output_dir": output_dir,
            "section": section,
            "system": system_data,
        }

        try:
            extend_and_plot_results(
                two_d_datas,
                times_T=times_T,
                times=times,
                extend_for=extend_for,
                **plot_args,
            )
            print(f"✅ {plot_type} plot completed!")

            # Force garbage collection after each plot to free memory
            plt.close("all")
            gc.collect()

            # Check memory usage after each plot
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
            print(f"  Memory usage after {plot_type}: {memory_usage:.2f} GB")

        except Exception as e:
            print(f"❌ Error plotting {plot_type}: {e}")
            plt.close("all")
            gc.collect()


# =============================
# CONVENIENCE PLOTTING FUNCTIONS
# =============================
def plot_1d_spectroscopy_data(config: dict) -> None:
    """Plot 1D spectroscopy data - convenience wrapper."""
    plot_spectroscopy_data(config, "1d")


def plot_2d_spectroscopy_data(config: dict) -> None:
    """Plot 2D spectroscopy data - convenience wrapper."""
    plot_spectroscopy_data(config, "2d")
