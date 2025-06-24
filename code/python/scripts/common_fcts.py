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
import gc
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

### Project-specific imports
from qspectro2d.spectroscopy.calculations import (
    parallel_compute_1d_E_with_inhomogenity,
    parallel_compute_2d_E_with_inhomogenity,
    check_the_solver,
    get_tau_cohs_and_t_dets_for_T_wait,
)
from qspectro2d.core.system_parameters import SystemParameters
from config.paths import DATA_DIR, FIGURES_DIR
from typing import Optional

from qspectro2d.visualization.plotting import (
    plot_fixed_tau_t,
    plot_2d_el_field,
    plot_1d_frequency_spectrum,
)
from qspectro2d.spectroscopy.post_processing import (
    compute_1d_fft_wavenumber,
    extend_and_plot_results,
)


# =============================
# GENERALIZED DATA SAVING FUNCTIONS
# =============================
def generate_unique_data_path(
    output_dir: Path, config: dict, system, simulation_type: str = "spectroscopy"
) -> Path:
    """
    Args:
        output_dir: Directory path where file will be saved
        config: Dictionary containing simulation parameters
        system: System parameters object
        simulation_type: Type of simulation ("1d", "2d", or other)

    Returns:
        unique path
    """
    # Build filename components based on simulation type (no timestamp)
    if simulation_type == "1d":
        base_filename = (
            f"1d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
            f"ph{config['n_phases']}_freq{config['n_freqs']}.pkl"
        )
    elif simulation_type == "2d":
        base_filename = (
            f"2d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
            f"T{config.get('n_times_T', 1)}_ph{config['n_phases']}_"
            f"freq{config['n_freqs']}.pkl"
        )
    else:
        base_filename = (
            f"{simulation_type}_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}.pkl"
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
    ### Create output directory with simulation type
    output_dir = DATA_DIR / f"{simulation_type}_spectroscopy" / config["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Generate unique filename
    save_path = generate_unique_data_path(output_dir, config, system, simulation_type)

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
        return _run_1d_simulation(config, system, max_workers)
    elif simulation_type == "2d":
        return _run_2d_simulation(config, system, max_workers)
    else:
        raise ValueError(f"Unsupported simulation type: {simulation_type}")


def _run_1d_simulation(
    config: dict, system: SystemParameters, max_workers: int
) -> tuple:
    """Run 1D spectroscopy simulation."""

    ### Create time arrays
    fwhms = system.fwhms
    times = np.arange(-1 * fwhms[0], system.t_max, system.dt)

    ### Validate solver
    # TODO add time_cut validation
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


def _run_2d_simulation(
    config: dict, system: SystemParameters, max_workers: int
) -> list:
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

    # RUN SIMULATION
    result = run_simulation(config, system, simulation_type)  # SAVE DATA
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

    # Return relative path from DATA_DIR for feed-forward compatibility
    relative_path = save_path.relative_to(DATA_DIR)
    return relative_path


# =============================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================
def run_1d_simulation_with_config(config: dict):
    """Run 1D simulation - wrapper for generalized function."""
    return run_simulation_with_config(config, "1d")


def run_2d_simulation_with_config(config: dict):
    """Run 2D simulation - wrapper for generalized function."""
    return run_simulation_with_config(config, "2d")


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


def find_latest_file_with_counter(pkl_files: list[Path]) -> Path:
    """
    Find the file with the highest counter suffix (_1, _2, etc.) from a list of pkl files.

    Args:
        pkl_files: List of Path objects pointing to .pkl files

    Returns:
        Path object of the file with highest counter, or the base file if no counters exist
    """
    import re

    # Group files by base name (without counter)
    file_groups = {}

    for pkl_file in pkl_files:
        filename = pkl_file.stem  # Remove .pkl extension

        # Check if filename ends with _number pattern
        counter_match = re.search(r"_(\d+)$", filename)

        if counter_match:
            # File has counter suffix
            counter = int(counter_match.group(1))
            base_name = filename[: counter_match.start()]  # Remove _number part
        else:
            # File has no counter (original file)
            counter = 0
            base_name = filename

        if base_name not in file_groups:
            file_groups[base_name] = []

        file_groups[base_name].append((pkl_file, counter))

    # Find the file with highest counter for each base name
    latest_files = []
    for base_name, files_with_counters in file_groups.items():
        # Sort by counter and take the highest
        files_with_counters.sort(key=lambda x: x[1], reverse=True)
        latest_file, highest_counter = files_with_counters[0]
        latest_files.append(latest_file)

        print(
            f"  📁 Base: {base_name} → Latest: {latest_file.name} (counter: {highest_counter})"
        )

    # If multiple base names exist, return all latest files
    # If only one base name, return the single latest file
    if len(latest_files) == 1:
        return latest_files[0]
    else:
        # Multiple different file types - return the one with highest overall counter
        all_counters = []
        for pkl_file in pkl_files:
            filename = pkl_file.stem
            counter_match = re.search(r"_(\d+)$", filename)
            counter = int(counter_match.group(1)) if counter_match else 0
            all_counters.append((pkl_file, counter))

        all_counters.sort(key=lambda x: x[1], reverse=True)
        return all_counters[0][0]


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

    output_dir = FIGURES_DIR / subdir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_output_directory_from_data_path(data_relative_path: Path) -> Path:
    """Create output directory for plots based on data path structure.

    Args:
        data_relative_path: Relative path from DATA_DIR (e.g., '1d_spectroscopy/special_dir/filename.pkl')

    Returns:
        Output directory path in FIGURES_DIR with same structure
    """
    # Extract the directory part (without filename)
    special_dir = data_relative_path.parent

    # Create corresponding directory in FIGURES_DIR
    output_dir = FIGURES_DIR / "figures_from_python" / special_dir
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

    loaded_data = load_pickle_file(DATA_DIR / file_path)

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
            plot_fixed_tau_t(
                t_det_vals=t_det_vals,
                data_avg=data_avg,
                tau_coh=data.get("tau_coh", 300),
                T_wait=data.get("T_wait", 1000),
                system=system,
                spectral_components_to_plot=config.get(
                    "spectral_components_to_plot", ["real", "imag", "abs", "phase"]
                ),
                output_dir=output_dir,
                save=True,  # Add the missing save parameter
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
            plot_1d_frequency_spectrum(
                frequencies=frequencies,
                data_fft=data_fft,
                system=system,
                spectral_components_to_plot=config.get(
                    "spectral_components_to_plot", ["real", "imag", "abs", "phase"]
                ),
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
            tau_cohs_0, t_dets_0 = get_tau_cohs_and_t_dets_for_T_wait(times, times_T[0])
            plot_2d_el_field(
                data_xyz=(
                    tau_cohs_0,
                    t_dets_0,
                    two_d_datas[0],
                ),
                domain="time",
                output_dir=output_dir,
                system=system_data,  # Add the missing system parameter
                save=True,
            )
        except Exception as e:
            print(f"❌ Error in time domain plotting: {e}")

    ### Plot frequency domain data
    extend_for = config.get("extend_for", (1, 2.3))
    section = config.get("section", (1.5, 1.7, 1.5, 1.7))
    spectral_components_to_plot = config.get(
        "spectral_components_to_plot", ["imag", "abs", "real", "phase"]
    )

    print(
        f"📊 Plotting frequency domain data with extend_for={extend_for}, section={section}"
    )

    for component in spectral_components_to_plot:
        print(f"   Generating {component} plot...")

        plot_args = {
            "component": component,
            "output_dir": output_dir,
            "section": section,
            "system": system_data,
            "save": True,
        }

        try:
            extend_and_plot_results(
                two_d_datas,
                times_T=times_T,
                times=times,
                extend_for=extend_for,
                **plot_args,
            )
            print(f"✅ {component} plot completed!")

            # Force garbage collection after each plot to free memory
            plt.close("all")
            gc.collect()

            # Check memory usage after each plot
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
            print(f"  Memory usage after {component}: {memory_usage:.2f} GB")

        except Exception as e:
            print(f"❌ Error plotting {component}: {e}")
            plt.close("all")
            gc.collect()


def plot_2d_from_filepath(filepath: Path, config: dict) -> None:
    """Plot 2D spectroscopy data from a specific filepath.

    Args:
        filepath: Path to the pickle file containing 2D data
        config: Dictionary containing plotting configuration
    """
    print("# =============================")
    print("# LOAD AND PLOT 2D DATA FROM FILEPATH")
    print("# =============================")

    ### Load data from the specific file
    loaded_data = load_pickle_file(filepath)

    if loaded_data is None:
        print(f"❌ Failed to load 2D spectroscopy data from {filepath}")
        return

    ### Create output directory based on data file structure
    # Get relative path from DATA_DIR to maintain same directory structure
    try:
        data_relative_path = filepath.relative_to(DATA_DIR)
        output_dir = create_output_directory_from_data_path(data_relative_path)
    except ValueError:
        # Fallback to config if filepath is not under DATA_DIR
        output_dir = create_output_directory(
            config.get("output_subdir", "figures_from_python")
        )

    ### Route to 2D plotting function
    _plot_2d_data(loaded_data, config, output_dir)

    print(f"🎯 All 2D plots saved to: {output_dir}")


def plot_1d_from_filepath(filepath: Path, config: dict) -> None:
    """Plot 1D spectroscopy data from a specific filepath.

    Args:
        filepath: Path to the pickle file containing 1D data
        config: Dictionary containing plotting configuration
    """
    print("# =============================")
    print("# LOAD AND PLOT 1D DATA FROM FILEPATH")
    print("# =============================")

    ### Load data from the specific file
    loaded_data = load_pickle_file(filepath)

    if loaded_data is None:
        print(f"❌ Failed to load 1D spectroscopy data from {filepath}")
        return

    ### Create output directory based on data file structure
    # Get relative path from DATA_DIR to maintain same directory structure
    try:
        data_relative_path = filepath.relative_to(DATA_DIR)
        output_dir = create_output_directory_from_data_path(data_relative_path)
    except ValueError:
        # Fallback to config if filepath is not under DATA_DIR
        output_dir = create_output_directory(
            config.get("output_subdir", "figures_from_python")
        )

    ### Route to 1D plotting function
    _plot_1d_data(loaded_data, config, output_dir)

    print(f"🎯 All 1D plots saved to: {output_dir}")


def plot_2d_from_relative_path(relative_path_str: str, config: dict = None) -> None:
    """Plot 2D spectroscopy data from a relative path string (feed-forward compatible).

    Args:
        relative_path_str: Relative path from DATA_DIR (e.g., '2d_spectroscopy/special_dir/filename.pkl' or '2d_spectroscopy/special_dir/')
        config: Optional plotting configuration dictionary
    """
    # Convert string to Path object
    relative_path = Path(relative_path_str)
    full_path = DATA_DIR / relative_path

    # Use default config if none provided
    if config is None:
        config = {
            "spectral_components_to_plot": ["real", "imag", "abs", "phase"],
            "plot_time_domain": True,
            "extend_for": (1, 3),
            "section": (1.4, 1.8, 1.4, 1.8),
        }

    print(f"📊 Plotting from relative path: {relative_path_str}")
    print(f"📂 Full path: {full_path}")

    # Check if it's a directory or file
    if full_path.is_dir():
        # Find all .pkl files in the directory
        pkl_files = list(full_path.glob("*.pkl"))
        if not pkl_files:
            print(f"❌ No .pkl files found in directory: {full_path}")
            return

        print(f"Found {len(pkl_files)} .pkl files:")
        for pkl_file in pkl_files:
            print(f"  - {pkl_file.name}")

        # Find and plot only the latest file (highest counter)
        latest_file = find_latest_file_with_counter(pkl_files)
        print(f"\n📊 Plotting latest file: {latest_file.name}")
        plot_2d_from_filepath(latest_file, config)

    elif full_path.is_file():
        # Direct file path provided
        plot_2d_from_filepath(full_path, config)
    else:
        print(f"❌ Path does not exist: {full_path}")
        return


def plot_1d_from_relative_path(relative_path_str: str, config: dict = None) -> None:
    """Plot 1D spectroscopy data from a relative path string (feed-forward compatible).

    Args:
        relative_path_str: Relative path from DATA_DIR (e.g., '1d_spectroscopy/special_dir/filename.pkl')
        config: Optional plotting configuration dictionary
    """
    # Convert string to Path object
    relative_path = Path(relative_path_str)
    full_filepath = DATA_DIR / relative_path

    # Use default config if none provided
    if config is None:
        config = {
            "spectral_components_to_plot": ["real", "imag", "abs", "phase"],
            "plot_time_domain": True,
            "plot_frequency_domain": True,
        }

    print(f"📊 Plotting from relative path: {relative_path_str}")
    print(f"📂 Full path: {full_filepath}")

    # Call the main plotting function
    plot_1d_from_filepath(full_filepath, config)


# =============================
# CONVENIENCE PLOTTING FUNCTIONS
# =============================
def plot_1d_spectroscopy_data(config: dict) -> None:
    """Plot 1D spectroscopy data - convenience wrapper."""
    plot_spectroscopy_data(config, simulation_type="1d")


def plot_2d_spectroscopy_data(config: dict) -> None:
    """Plot 2D spectroscopy data - convenience wrapper."""
    plot_spectroscopy_data(config, simulation_type="2d")
