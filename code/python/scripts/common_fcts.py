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
from typing import Optional, Literal, Union, Dict, Any

### Project-specific imports
from qspectro2d.spectroscopy.calculations import (
    parallel_compute_1d_E_with_inhomogenity,
    parallel_compute_2d_E_with_inhomogenity,
    check_the_solver,
    get_tau_cohs_and_t_dets_for_T_wait,
)
from qspectro2d.core.system_parameters import SystemParameters
from config.paths import DATA_DIR, FIGURES_DIR
from config.mpl_tex_settings import DEFAULT_FIG_FORMAT, save_fig

from qspectro2d.visualization.plotting import (
    plot_1d_el_field,
    plot_2d_el_field,
)
from qspectro2d.spectroscopy.post_processing import (
    compute_1d_fft_wavenumber,
    extend_time_axes,
    compute_2d_fft_wavenumber,
)


# =============================
# FILENAME GENERATION AND DATA MANAGEMENT
# =============================
def get_data_subdir(config: dict, system: SystemParameters) -> Path:
    """
    Generate standardized subdirectory path based on system and configuration.

    Args:
        config: Dictionary containing simulation parameters
        system: System parameters object

    Returns:
        Path: Relative path for data storage
    """
    # Base directory based on number of atoms and solver type
    parts = []

    # Add simulation dimension (1d/2d)
    if "simulation_type" in config:
        parts.append(f"{config['simulation_type']}_spectroscopy")
    else:
        # Default to the most common case
        parts.append("spectroscopy")

    # Add system details
    parts.append(f"N{system.N_atoms}")

    # Add solver if available
    parts.append(system.ODE_Solver.lower())

    # Add RWA if available
    parts.append("RWA" if system.RWA_laser else "no_RWA")

    # Add output subdirectory if specified
    if "output_subdir" in config:
        parts.append(config["output_subdir"])

    # Join all parts with path separator
    return Path(*parts)


def generate_data_filename(system, config: dict) -> str:
    """
    Generate a standardized filename for data files.

    Args:
        system: System parameters object
        config: Dictionary containing simulation parameters

    Returns:
        str: Base filename without path
    """
    simulation_type = config.get("simulation_type", "spectroscopy")

    # Common parameters for all simulation types
    common_parts = [f"tmax_{system.t_max:.0f}", f"dt_{system.dt:.1f}"]

    if simulation_type == "1d":
        parts = (
            ["1d"]
            + common_parts
            + [f"ph{config.get('n_phases', 0)}", f"freq{config.get('n_freqs', 0)}"]
        )
    elif simulation_type == "2d":
        parts = (
            ["2d"]
            + common_parts
            + [
                f"T{config.get('n_times_T', 1)}",
                f"ph{config.get('n_phases', 0)}",
                f"freq{config.get('n_freqs', 0)}",
            ]
        )
    else:
        parts = [simulation_type] + common_parts

    return "_".join(parts) + ".pkl.gz"


def save_data_with_unique_path(payload: dict, config: dict, system) -> Path:
    """
    Save data to a uniquely named file and return the relative path.

    Args:
        payload: Dictionary containing data to save
        config: Dictionary containing simulation parameters
        system: System parameters object

    Returns:
        Path: Relative path to the saved file
    """
    import gzip
    import pickle
    from pathlib import Path

    # Get the relative directory for this data
    relative_dir = get_data_subdir(config, system)

    # Create full path to output directory
    output_dir = DATA_DIR / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate base filename
    base_filename = generate_data_filename(system, config)

    # Ensure unique filename by adding counter if needed
    counter = 1
    save_path = output_dir / base_filename

    while save_path.exists():
        name_parts = base_filename.split(".pkl.gz")[0]
        save_path = output_dir / f"{name_parts}_{counter}.pkl.gz"
        counter += 1

    # Save data
    try:
        with gzip.open(save_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"✅ Data saved successfully to: {save_path}")

        # Return relative path for future reference
        return relative_dir
    except Exception as e:
        print(f"❌ ERROR: Failed to save data: {e}")
        raise


def get_latest_data_file(relative_dir: Path) -> Path:
    """
    Find the most recent data file in the specified directory.

    Args:
        relative_dir: Relative path to the directory containing data files

    Returns:
        Path: Full path to the most recent data file
    """
    import glob
    import os

    # Create full path to data directory
    data_dir = DATA_DIR / relative_dir

    # Find all pickle files in the directory
    file_pattern = str(data_dir / "*.pkl.gz")
    files = glob.glob(file_pattern)

    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    # Sort by modification time (newest last)
    latest_file = max(files, key=os.path.getmtime)

    return Path(latest_file)


def load_latest_data(relative_dir: Path) -> dict:
    """
    Load the most recent data file from the specified directory.

    Args:
        relative_dir: Relative path to the directory containing data files

    Returns:
        dict: The loaded data
    """
    import gzip
    import pickle

    # Get the latest file
    latest_file = get_latest_data_file(relative_dir)

    # Load and return data
    try:
        with gzip.open(latest_file, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Loaded data from: {latest_file}")
        return data
    except Exception as e:
        print(f"❌ ERROR: Failed to load data from {latest_file}: {e}")
        raise


def build_plot_filename(
    system,
    domain: str,
    component: str = None,
    dimension: str = "1D",
    t_wait: float = None,
) -> str:
    """
    Build a standardized filename for plots.

    Args:
        system: System parameters object
        domain: Data domain ("time" or "freq")
        component: Optional component name ("real", "imag", "abs", "phase")
        dimension: Plot dimension ("1D" or "2D")
        t_wait: Optional waiting time for 2D plots

    Returns:
        str: Standardized base filename for the plot (without extension)
    """
    # Start with basic structure
    parts = [dimension, domain]

    # Add component if specified
    if component:
        parts.append(component)

    # Add system parameters
    parts.append(f"N{system.N_atoms}")
    parts.append(f"wA{system.omega_A:.2f}")

    # Add waiting time for 2D plots
    if dimension == "2D" and t_wait is not None and t_wait != np.inf:
        parts.append(f"T{t_wait:.0f}fs")

    # Add solver if available
    if hasattr(system, "ODE_Solver"):
        parts.append(system.ODE_Solver.lower())

    # Join with underscores
    return "_".join(parts)


# =============================
# UNIFIED DATA SAVING FUNCTIONS
# =============================
def unique_data_filename(
    output_dir: Path, config: dict, system, simulation_type: str = "spectroscopy"
) -> Path:
    """
    Generate unique data path with compressed pickle format.

    Args:
        output_dir: Directory path where file will be saved
        config: Dictionary containing simulation parameters
        system: System parameters object
        simulation_type: Type of simulation ("1d", "2d", or other)

    Returns:
        unique path with .pkl.gz extension
    """
    # Build filename components based on simulation type (no timestamp)
    if simulation_type == "1d":
        base_filename = (
            f"1d_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
            f"ph{config['n_phases']}_freq{config['n_freqs']}.pkl.gz"
        )
    elif simulation_type == "2d":
        base_filename = (
            f"2d_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
            f"T{config.get('n_times_T', 1)}_ph{config['n_phases']}_"
            f"freq{config['n_freqs']}.pkl.gz"
        )
    else:
        base_filename = (
            f"{simulation_type}_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}.pkl.gz"
        )

    save_path = output_dir / base_filename

    ### Ensure unique filename by adding counter if needed
    counter = 1
    while save_path.exists():
        name_parts = base_filename.split(".pkl.gz")[0]
        name_with_counter = f"{name_parts}_{counter}.pkl.gz"
        save_path = output_dir / name_with_counter
        counter += 1

    return save_path


def save_data(
    payload: dict,
    config: dict,
    system: SystemParameters,
    simulation_type: str = "spectroscopy",
) -> Path:
    """
    Unified function to save spectroscopy simulation data with standardized structure.

    Args:
        payload: Dictionary containing data, axes, system, config, metadata
        config: Dictionary containing simulation parameters
        system: System parameters object
        simulation_type: Type of simulation ("1d", "2d", or other)

    Returns:
        Path to saved file
    """
    import gzip

    ### Create output directory with simulation type
    output_dir = DATA_DIR / f"{simulation_type}_spectroscopy" / config["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Generate unique filename
    save_path = unique_data_filename(output_dir, config, system, simulation_type)

    ### Save as compressed pickle file
    try:
        with gzip.open(save_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"✅ Data saved successfully to: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ ERROR: Failed to save data: {e}")
        raise


# =============================
# UNIFIED DATA STRUCTURE BUILDERS
# =============================
def build_1d_payload(
    t_det_vals: np.ndarray, data_avg: np.ndarray, config: dict, system: SystemParameters
) -> dict:
    """Build standardized 1D data payload."""
    return {
        "data": data_avg,
        "axes": {
            "t_det": t_det_vals,
            "tau_coh": np.array([config["tau_coh"]]),  # Single value as array
            "T_wait": np.array([config["T_wait"]]),  # Single value as array
        },
        "system": system,
        "config": config,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "simulation_type": "1d",
            "n_phases": config["n_phases"],
            "n_freqs": config["n_freqs"],
        },
    }


def build_2d_payload(
    two_d_datas: list,
    times: np.ndarray,
    times_T: np.ndarray,
    config: dict,
    system: SystemParameters,
) -> dict:
    """Build standardized 2D data payload."""
    # Get tau_coh and t_det arrays for the first T_wait for axes info
    tau_coh_vals, t_det_vals = get_tau_cohs_and_t_dets_for_T_wait(times, times_T[0])

    return {
        "data": two_d_datas,
        "axes": {
            "t_det": t_det_vals,
            "tau_coh": tau_coh_vals,
            "T_wait": times_T,
        },
        "system": system,
        "config": config,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "simulation_type": "2d",
            "n_phases": config["n_phases"],
            "n_freqs": config["n_freqs"],
            "n_times_T": config["n_times_T"],
        },
    }


# =============================
# BACKWARD COMPATIBILITY WRAPPERS
# =============================
def save_1d_data(
    t_det_vals: np.ndarray, data_avg: np.ndarray, config: dict, system: SystemParameters
) -> Path:
    """Save 1D polarization simulation data - wrapper for unified function."""
    payload = build_1d_payload(t_det_vals, data_avg, config, system)
    return save_data(payload, config, system, "1d")


def save_2d_data(
    two_d_datas: list,
    times: np.ndarray,
    times_T: np.ndarray,
    config: dict,
    system: SystemParameters,
) -> Path:
    """Save 2D polarization simulation data - wrapper for unified function."""
    payload = build_2d_payload(two_d_datas, times, times_T, config, system)
    return save_data(payload, config, system, "2d")


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
    test_system.t_max = 1 * system.t_max
    test_system.dt = 20 * system.dt
    times_test = np.arange(-fwhms[0], test_system.t_max, test_system.dt)

    try:
        _, time_cut = check_the_solver(times_test, test_system)
        print("#" * 60)
        print(
            f"✅  Solver validation worked: Evolution becomes unphysical at"
            f"({time_cut / test_system.t_max:.2f} × t_max)"
        )
        print("#" * 60)
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
# HARMONIZED SIMULATION FUNCTIONS
# =============================
def run_simulation(
    config: dict, system: SystemParameters, simulation_type: str
) -> dict:
    """
    Run spectroscopy simulation and return standardized payload.

    Returns:
        dict: Standardized payload with keys: data, axes, system, config, metadata
    """
    max_workers = get_max_workers()

    if simulation_type == "1d":
        t_det_vals, data_avg = _run_1d_simulation(config, system, max_workers)
        return build_1d_payload(t_det_vals, data_avg, config, system)
    elif simulation_type == "2d":
        two_d_datas, times, times_T = _run_2d_simulation(config, system, max_workers)
        return build_2d_payload(two_d_datas, times, times_T, config, system)
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
    ### Validate solver
    time_cut = validate_solver(system, times)
    if time_cut < times[-1]:
        print(
            f"⚠️  WARNING: Time cut {time_cut} is less than the last time point {times[-1]}. "
            "This may affect the simulation results.",
            flush=True,
        )

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
# UNIFIED MAIN SIMULATION RUNNER
# =============================
def run_simulation_with_config(config: dict, simulation_type: str) -> str:
    """
    Unified simulation runner that returns standardized payload and path.

    Args:
        config: Dictionary containing simulation parameters
        simulation_type: "1d" or "2d"

    Returns:
        str: Relative path from DATA_DIR for feed-forward compatibility
    """
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
    # RUN SIMULATION (returns standardized payload)
    # =============================
    payload = run_simulation(config, system, simulation_type)

    # =============================
    # SAVE DATA
    # =============================
    print("\nSaving simulation data...")
    save_path = save_data(payload, config, system, simulation_type)

    # =============================
    # SIMULATION SUMMARY
    # =============================
    elapsed_time = time.time() - start_time
    print_simulation_summary(elapsed_time, payload["data"], save_path, simulation_type)

    # Return relative path from DATA_DIR for feed-forward compatibility
    relative_path = save_path.relative_to(DATA_DIR)
    return str(relative_path)


# =============================
# GENERALIZED PLOTTING FUNCTIONS
# =============================
def find_latest_file(data_subdir: str, file_pattern: str = "*.pkl*") -> Optional[Path]:
    """Find the most recent file matching pattern in a data subdirectory.

    Args:
        data_subdir: Subdirectory within DATA_DIR (e.g., '2d_spectroscopy/N_1/paper_eqs/100fs')
        file_pattern: Glob pattern for file matching (supports both .pkl and .pkl.gz)

    Returns:
        Path to the latest file or None if not found
    """
    data_dir = DATA_DIR / data_subdir
    if not data_dir.exists():
        print(f"❌ Data directory does not exist: {data_dir}")
        return None

    # Search for files matching the pattern(s)
    files = []
    patterns = [file_pattern]
    if file_pattern == "*.pkl*":  # Special case for pickle files
        patterns = ["*.pkl", "*.pkl.gz"]

    for pattern in patterns:
        files.extend(list(data_dir.glob(pattern)))

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

    # Simple approach: extract all counters, sort by counter value
    counters = []
    for pkl_file in pkl_files:
        filename = pkl_file.stem  # Remove extension
        counter_match = re.search(r"_(\d+)$", filename)
        counter = int(counter_match.group(1)) if counter_match else 0
        counters.append((pkl_file, counter))

    # Sort by counter (descending)
    counters.sort(key=lambda x: x[1], reverse=True)
    latest_file = counters[0][0]

    print(f"✅ Selected latest file: {latest_file.name} (counter: {counters[0][1]})")
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
        if str(filepath).endswith(".gz"):
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


def create_output_directory_from_data_path(data_path: Path) -> Path:
    """Create output directory for plots based on data path structure.

    Args:
        data_path: Path to data file (can be absolute or relative to DATA_DIR)

    Returns:
        Output directory path in FIGURES_DIR with same structure
    """
    # Handle both absolute and relative paths
    try:
        relative_path = data_path.relative_to(DATA_DIR)
    except ValueError:
        # Path is already relative or outside DATA_DIR
        relative_path = data_path

    # Create corresponding directory in FIGURES_DIR
    output_dir = FIGURES_DIR / "figures_from_python" / relative_path.parent
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_from_filepath(
    filepath: Path, config: dict = None, simulation_type: str = None
) -> None:
    """Plot spectroscopy data from a specific filepath.

    Args:
        filepath: Path to the pickle file containing data
        config: Dictionary containing plotting configuration
        simulation_type: Simulation type ("1d" or "2d"). If None, determined automatically.
    """
    print(f"# =============================")
    print(f"# LOAD AND PLOT DATA FROM {filepath}")
    print(f"# =============================")

    # Auto-detect simulation type if not provided
    if simulation_type is None:
        simulation_type = "2d" if "2d" in str(filepath).lower() else "1d"
        print(f"Auto-detected simulation type: {simulation_type}")

    # Use default config if none provided
    if config is None:
        config = {
            "spectral_components_to_plot": ["abs"],
            "plot_time_domain": True,
            "plot_frequency_domain": True,
        }
        # Add 2D-specific defaults
        if simulation_type == "2d":
            config.update(
                {
                    "extend_for": (1, 3),
                    "section": (1.4, 1.8, 1.4, 1.8),
                }
            )

    # Load data from the specific file
    loaded_data = load_pickle_file(filepath)
    if loaded_data is None:
        print(f"❌ Failed to load {simulation_type} spectroscopy data from {filepath}")
        return

    # Create output directory based on data file structure
    output_dir = create_output_directory_from_data_path(filepath)

    # Route to specific plotting function
    if simulation_type == "1d":
        _plot_1d_data(loaded_data, config, output_dir)
    else:  # simulation_type == "2d"
        _plot_2d_data(loaded_data, config, output_dir)

    print(f"🎯 All {simulation_type.upper()} plots saved to: {output_dir}")


def _plot_1d_data(data: dict, config: dict, output_dir: Path) -> None:
    """Plot 1D spectroscopy data using standardized data structure.

    Args:
        data: Dictionary with standardized 1D data structure
        config: Plotting configuration
        output_dir: Directory to save plots
    """
    # Extract data from standardized structure
    data_avg = data["data"]
    axes = data["axes"]
    t_det_vals = axes["t_det"]
    tau_coh = axes["tau_coh"][0] if len(axes["tau_coh"]) > 0 else 300.0
    T_wait = axes["T_wait"][0] if len(axes["T_wait"]) > 0 else 1000.0
    system = data["system"]

    print(f"✅ Data loaded with shape: {data_avg.shape}")
    print(f"   Time points: {len(t_det_vals)}")
    print(f"   Time range: {t_det_vals[0]:.1f} to {t_det_vals[-1]:.1f} fs")

    spectral_components = config.get("spectral_components_to_plot", ["abs"])

    ### Plot time domain data
    if config.get("plot_time_domain", True):
        print("📊 Plotting time domain data...")
        try:
            fig = plot_1d_el_field(  # TODO add the decision if P or E_k_s is plotted (in title or something like that)!
                data_x=t_det_vals,
                data_y=data_avg,
                domain="time",
                component="abs",  # Use first component for time domain
                tau_coh=tau_coh,
                T_wait=T_wait,
            )
            filename = build_plot_filename(
                system, domain="time", component="abs", dimension="1D"
            )
            save_fig(fig, filename=filename, output_dir=output_dir)
            plt.close(fig)
            print("✅ Time domain plots completed!")
        except Exception as e:
            print(f"❌ Error in time domain plotting: {e}")

    ### Plot frequency domain data
    if config.get("plot_frequency_domain", True):
        print("📊 Plotting frequency domain data...")
        extend_for = config.get("extend_for", (1, 1))
        # Extend time axes if needed
        if extend_for != (1, 1):
            extended_x, extended_data = extend_time_axes(
                data=data_avg,
                t_det=t_det_vals,
                pad_t_det=extend_for,
            )
        else:
            extended_x, extended_data = t_det_vals, data_avg

        frequencies, data_fft = compute_1d_fft_wavenumber(extended_x, extended_data)
        # Plot each spectral component separately
        for component in spectral_components:

            fig = plot_1d_el_field(
                data_x=frequencies,
                data_y=data_fft,
                domain="freq",
                component=component,
            )
            filename = build_plot_filename(
                system, domain="freq", component=component, dimension="1D"
            )
            save_fig(fig, filename=filename, output_dir=output_dir)
            plt.close(fig)

        print("✅ Frequency domain plots completed!")

    # Clean up memory
    plt.close("all")
    gc.collect()


def _plot_2d_data(data: dict, config: dict, output_dir: Path) -> None:
    """Plot 2D spectroscopy data using standardized data structure.

    Args:
        data: Dictionary with standardized 2D data structure
        config: Plotting configuration
        output_dir: Directory to save plots
    """
    # Extract data from standardized structure
    two_d_datas = data["data"]
    axes = data["axes"]
    times_T = axes["T_wait"]
    system_data = data["system"]

    # Get simulation parameters
    fwhm0 = system_data.fwhms[0] if hasattr(system_data, "fwhms") else 100  # Default
    times = np.arange(-1 * fwhm0, system_data.t_max, system_data.dt)

    # Get configuration values
    extend_for = config.get("extend_for", (1, 1))
    section = config.get("section", (1.4, 1.8, 1.4, 1.8))
    spectral_components = config.get("spectral_components_to_plot", ["abs"])

    print(f"✅ 2D data loaded successfully!")
    print(f"   Times shape: {times.shape}")
    print(f"   Times_T shape: {times_T.shape}")
    print(f"   Data shape: {two_d_datas[0].shape if two_d_datas else 'None'}")

    # Filter out None values from averaged_results
    valid_results = [res for res in two_d_datas if res is not None]
    valid_T_waits = [times_T[i] for i, res in enumerate(two_d_datas) if res is not None]

    if not valid_results:
        print("❌ No valid results to plot")
        return

    # Process each valid waiting time
    for i, data_array in enumerate(valid_results):
        T_wait = valid_T_waits[i]
        ts, taus = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait)

        # Plot time domain data
        if config.get("plot_time_domain", True):
            try:
                fig = plot_2d_el_field(
                    data_x=ts,
                    data_y=taus,
                    data_z=data_array,
                    t_wait=T_wait,
                    domain="time",
                    use_custom_colormap=True,
                )
                filename = build_plot_filename(
                    system=system_data, domain="time", dimension="2D", t_wait=T_wait
                )
                save_fig(fig, filename=filename, output_dir=output_dir)
                plt.close(fig)
            except Exception as e:
                print(f"❌ Error in 2D time domain plotting: {e}")

        # Handle frequency domain processing
        if config.get("plot_frequency_domain", True):
            try:
                # Extend time axes if needed
                if extend_for != (1, 1):
                    extended_ts, extended_taus, extended_data = extend_time_axes(
                        data=data_array,
                        t_det=ts,
                        tau_coh=taus,
                        pad_t_det=extend_for,
                        pad_tau_coh=extend_for,
                    )
                else:
                    extended_ts, extended_taus, extended_data = ts, taus, data_array

                # Compute FFT
                nu_ts, nu_taus, data_freq = compute_2d_fft_wavenumber(
                    extended_ts, extended_taus, extended_data
                )

                # Plot each component
                for component in spectral_components:
                    fig = plot_2d_el_field(
                        data_x=nu_ts,
                        data_y=nu_taus,
                        data_z=data_freq,
                        t_wait=T_wait,
                        domain="freq",
                        use_custom_colormap=True,
                        component=component,
                        section=section,
                    )
                    filename = build_plot_filename(
                        system=system_data,
                        domain="freq",
                        component=component,
                        dimension="2D",
                        t_wait=T_wait,
                    )
                    save_fig(fig, filename=filename, output_dir=output_dir)
                    plt.close(fig)

            except Exception as e:
                print(f"❌ Error plotting 2D {component} component: {e}")

    # Clean up memory
    plt.close("all")
    gc.collect()
