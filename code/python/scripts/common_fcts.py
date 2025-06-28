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
import gzip
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Optional, Literal, Union, Dict, Any, Tuple

### Project-specific imports
from qspectro2d.spectroscopy.calculations import (
    parallel_compute_1d_E_with_inhomogenity,
    parallel_compute_2d_E_with_inhomogenity,
    check_the_solver,
)
from qspectro2d.core.system_parameters import SystemParameters
from config.paths import DATA_DIR, FIGURES_DIR
from config.mpl_tex_settings import save_fig

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
def _generate_base_filename(system: SystemParameters, config: dict) -> str:
    """
    Generate a universal base filename for a calculation based on system and config parameters.

    Args:
        system: System parameters object
        config: Dictionary containing simulation parameters

    Returns:
        str: Base filename without path
    """
    simulation_type = config.get("simulation_type", "spectroscopy")
    N_atoms = system.N_atoms

    parts = [simulation_type]
    parts.append(f"N{N_atoms}")
    parts.append(f"wA{system.omega_A:.2f}")

    if N_atoms == 2:
        parts.append(f"T_wait{config.get('t_wait', 0)}fs")

    parts += [
        f"t_det_max_{config.get('t_det_max', 0)}",
        f"dt_{system.dt:.1f}",
        f"{config.get('n_phases', 0)}ph",
        f"{config.get('n_freqs', 1)}freq",
    ]

    return "_".join(parts)


def _generate_unique_filename(path: Union[str, Path], base_name: str) -> str:
    """
    Generate a unique filename in the specified directory based on system and config parameters.

    Args:
        path (str or Path): Directory where the file will be saved
        base_name (str): Base name for the file (without extension)
    Returns:
        str: Unique filename with full path
    """
    path = Path(path)
    save_path = path / base_name
    counter = 1

    while save_path.exists():
        save_path = path / f"{base_name}_{counter}"
        counter += 1

    return str(save_path)


def _generate_base_sub_dir(config: dict, system) -> Path:
    """
    Generate standardized subdirectory path based on system and configuration. WILL BE subdir of  DATA_DIR OR FIGURES_DIR

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
    parts.append(system.ODE_Solver)

    # Add RWA if available
    parts.append("RWA" if system.RWA_laser else "noRWA")

    # Join all parts with path separator
    return Path(*parts)


def _get_latest_data_file(relative_dir: Path) -> Path:
    """
    Find the most recent data file in the specified directory.

    Args:
        relative_dir: Relative path to the directory containing data files

    Returns:
        Path: Full path to the most recent data file
    """
    import glob

    # Create full path to data directory
    data_dir = DATA_DIR / relative_dir

    # Find all numpy compressed files in the directory
    file_pattern = str(data_dir / "*.npz")
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
    import numpy as np

    # Get the latest file
    latest_file = _get_latest_data_file(relative_dir)

    # Load and return data
    try:
        with np.load(latest_file, allow_pickle=True) as data:
            loaded_data = {key: data[key] for key in data.files}
        print(f"✅ Loaded data from: {latest_file}")
        return loaded_data
    except Exception as e:
        print(f"❌ ERROR: Failed to load data from {latest_file}: {e}")
        raise


def generate_unique_data_filename(
    system: SystemParameters,
    config: dict,
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
    realtive_path = _generate_base_sub_dir(config, system)
    path = DATA_DIR / realtive_path
    path.mkdir(parents=True, exist_ok=True)
    base_name = _generate_base_filename(system, config)
    filename = _generate_unique_filename(path, base_name)
    return filename


def generate_unique_plot_filename(
    system: SystemParameters,
    config: dict,
    domain: str,
    component: str = None,
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
    realtive_path = _generate_base_sub_dir(config, system)
    path = FIGURES_DIR / realtive_path
    base_name = _generate_base_filename(system, config)
    base_name += f"_{domain}_domain"
    if component:
        base_name += f"_{component}_comp"

    filename = _generate_unique_filename(path, base_name)
    return filename


# CHECKED
# =============================
# UNIFIED DATA SAVING FUNCTIONS
# =============================
def save_simulation_data(
    system: SystemParameters,
    config: dict,
    data: np.ndarray,
    axs1: np.ndarray,
    axs2: Optional[np.ndarray] = None,
) -> Tuple[Path, Path]:
    """
    Save spectroscopy simulation data (numpy arrays) along with known axes in one file,
    and system parameters and configuration in another file.

    Parameters:
        data (np.ndarray): Simulation results (1D or 2D data).
        axs1 (np.ndarray): First axis (e.g., time or frequency for 1D or 2D data).
        axs2 (Optional[np.ndarray]): Second axis (e.g., coherence time for 2D data).
        system (SystemParameters): System parameters object.
        config (dict): Simulation configuration dictionary.

    Returns:
        Tuple[Path, Path]: Paths to the saved numpy data file and info file.
    """
    # =============================
    # Generate unique filenames
    # =============================
    base_path = generate_unique_data_filename(system, config)
    data_path = Path(f"{base_path}_data.npz")
    info_path = Path(f"{base_path}_info.pkl")

    # =============================
    # Save data and axes to numpy file
    # =============================
    try:
        if axs2 is not None:
            np.savez_compressed(data_path, data=data, axis1=axs1, axis2=axs2)
        else:
            np.savez_compressed(data_path, data=data, axis1=axs1)
        print(f"✅ Data saved successfully to: {data_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save data: {e}")
        raise

    # =============================
    # Save system and config to info file
    # =============================
    try:
        with open(info_path, "wb") as info_file:
            pickle.dump({"system": system, "config": config}, info_file)
        print(f"✅ Info saved successfully to: {info_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save info: {e}")
        raise

    return data_path, info_path


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


def create_system_parameters(config: dict) -> SystemParameters:
    """
    Create a SystemParameters object from a configuration dictionary.

    Parameters:
    config (dict): Configuration dictionary containing simulation parameters.

    Returns:
    SystemParameters: Initialized SystemParameters object.

    Raises:
    KeyError: If required keys for time configuration are missing.
    """
    # Handle different time configurations for 1D vs 2D
    if "tau_coh" in config:
        # 1D configuration
        t_max = config["tau_coh"] + config["T_wait"] + config["t_det_max"]
    else:
        # 2D configuration
        t_max = config["T_wait"] + 2 * config["t_det_max"]

    # Set all parameters to defaults if not provided in config
    return SystemParameters(
        t_max=t_max,
        Temp=config.get("Temp", 0),
        cutoff_=config.get("cutoff_", 1),
        N_atoms=config.get("N_atoms", 1),
        ODE_Solver=config.get("ODE_Solver", "Paper_eqs"),
        RWA_laser=config.get("RWA_laser", True),
        dt=config.get("dt", 1),
        bath=config.get("bath", "paper"),
        E0=config.get("E0", 1),
        envelope_type=config.get("envelope_type", "gaussian"),
        pulse_fwhm=config.get("pulse_fwhm", 15),
        omega_laser_cm=config.get("omega_laser_cm", 16000),
        Delta_cm=config.get("Delta_cm", 0),
        omega_A_cm=config.get("omega_A_cm", 16000),
        mu_A=config.get("mu_A", 1),
        omega_B_cm=config.get("omega_B_cm", 16000),
        mu_B=config.get("mu_B", 1),
        J_cm=config.get("J_cm", 0),
        gamma_0=config.get("gamma_0", 1 / 300),
        gamma_phi=config.get("gamma_phi", 1 / 100),
    )


# =============================
# SIMULATION UTILITIES
# =============================


def run_1d_simulation(
    config: dict, system: SystemParameters, max_workers: int
) -> tuple:
    """
    Run 1D spectroscopy simulation with updated calculation structure.

    Parameters:
        config: Dictionary containing simulation parameters.
        system: System parameters object.
        max_workers: Number of parallel workers.

    Returns:
        tuple: Detection time values and averaged data.
    """
    ### Create time arrays
    tau_coh = config["tau_coh"]
    T_wait = config["T_wait"]
    t_det_max = config["t_det_max"]
    t_max = system.t_max
    ### Validate solver
    time_cut = -np.inf
    try:
        _, time_cut = check_the_solver(system)
        print("#" * 60)
        print(
            f"✅  Solver validation worked: Evolution becomes unphysical at"
            f"({time_cut / t_max:.2f} × t_max)"
        )
        print("#" * 60)
    except Exception as e:
        print(f"⚠️  WARNING: Solver validation failed: {e}")

    if time_cut < t_max:
        print(
            f"⚠️  WARNING: Time cut {time_cut} is less than the last time point {t_max}. "
            "This may affect the simulation results.",
            flush=True,
        )

    print("Computing 1D polarization with parallel processing...")

    try:
        t_det_vals, data = parallel_compute_1d_E_with_inhomogenity(
            n_freqs=config["n_freqs"],
            n_phases=config["n_phases"],
            tau_coh=tau_coh,
            T_wait=T_wait,
            t_det_max=t_det_max,
            system=system,
            max_workers=max_workers,
            time_cut=time_cut,
        )
        print("✅ Parallel computation completed successfully!")
        return t_det_vals, data
    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise


def run_2d_simulation(
    config: dict, system: SystemParameters, max_workers: int
) -> tuple:
    """
    Run 2D spectroscopy simulation with updated calculation structure.

    Parameters:
        config: Dictionary containing simulation parameters.
        system: System parameters object.
        max_workers: Number of parallel workers.

    Returns:
        tuple: Coherence times, detection times, and averaged 2D data.
    """
    ### Validate solver
    time_cut = -np.inf
    t_max = system.t_max
    try:
        _, time_cut = check_the_solver(system)
        print("#" * 60)
        print(
            f"✅  Solver validation worked: Evolution becomes unphysical at"
            f"({time_cut / t_max:.2f} × t_max)"
        )
        print("#" * 60)
    except Exception as e:
        print(f"⚠️  WARNING: Solver validation failed: {e}")

    if time_cut < t_max:
        print(
            f"⚠️  WARNING: Time cut {time_cut} is less than the last time point {t_max}. "
            "This may affect the simulation results.",
            flush=True,
        )

    print("Computing 2D polarization with parallel processing...")
    kwargs = {"plot_example": False, "time_cut": time_cut}

    try:
        tau_coh_vals, t_det_vals, data_2d = parallel_compute_2d_E_with_inhomogenity(
            n_freqs=config["n_freqs"],
            n_phases=config["n_phases"],
            T_wait=config["T_wait"],
            t_det_max=config["t_det_max"],
            system=system,
            max_workers=max_workers,
            **kwargs,
        )
        print("✅ Parallel computation completed successfully!")
        return tau_coh_vals, t_det_vals, data_2d
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
    print(
        f"  Parameters: #phases={config['n_phases']}, #frequencies={config['n_freqs']}"
    )
    print(f"  Times: t_det_max={config['t_det_max']} fs, dt={config['dt']} fs")
    if simulation_type == "1d":
        print(f"  Times: τ_coh={config['tau_coh']} fs, T_wait={config['T_wait']} fs")

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


# =============================
# GENERALIZED PLOTTING FUNCTIONS
# =============================
def plot_1d_data(
    ax1: np.array,
    data: np.array,
    data_dict: dict,
    plot_config: dict,
    output_dir: Path,
) -> None:
    """Plot 1D spectroscopy data using standardized data structure.

    Args:
    ax1: 1D array of detection times
        data: 1D array of averaged data values
        data_dict: Dictionary containing additional data (e.g., data_config and system parameters)
        plot_config: Plotting configuration
        output_dir: Directory to save plots
    """
    # Extract data from standardized structure
    t_det_vals = ax1
    system = data_dict["system"]
    data_config = data_dict["data_config"]
    tau_coh = data_config["tau_coh"]
    T_wait = data_config["T_wait"]

    print(f"✅ Data loaded with shape: {data.shape}")
    print(f"   Time points: {len(t_det_vals)}")
    print(f"   Time range: {t_det_vals[0]:.1f} to {t_det_vals[-1]:.1f} fs")

    spectral_components = plot_config.get("spectral_components_to_plot", ["abs"])
    extend_for = plot_config.get("extend_for", (1, 1))

    ### Plot time domain data
    if plot_config.get("plot_time_domain", True):
        print("📊 Plotting time domain data...")
        try:
            fig = plot_1d_el_field(  # TODO add the decision if P or E_k_s is plotted (in title or something like that)!
                data_x=t_det_vals,
                data_y=data,
                domain="time",
                component="abs",  # Use first component for time domain
                tau_coh=tau_coh,
                T_wait=T_wait,
            )
            filename = generate_unique_plot_filename(
                system,
                config=data_config,
                domain="time",
                component="abs",
            )
            save_fig(fig, filename=filename, output_dir=output_dir)
            plt.close(fig)
            print("✅ Time domain plots completed!")
        except Exception as e:
            print(f"❌ Error in time domain plotting: {e}")

    ### Plot frequency domain data
    if plot_config.get("plot_frequency_domain", True):
        print("📊 Plotting frequency domain data...")
        # Extend time axes if needed
        if extend_for != (1, 1):
            extended_x, extended_data = extend_time_axes(
                data=data,
                t_det=t_det_vals,
                pad_t_det=extend_for,
            )
        else:
            extended_x, extended_data = t_det_vals, data

        frequencies, data_fft = compute_1d_fft_wavenumber(extended_x, extended_data)
        # Plot each spectral component separately
        for component in spectral_components:

            fig = plot_1d_el_field(
                data_x=frequencies,
                data_y=data_fft,
                domain="freq",
                component=component,
            )
            filename = generate_unique_plot_filename(
                system,
                config=data_config,
                domain="freq",
                component=component,
            )

            save_fig(fig, filename=filename, output_dir=output_dir)
            plt.close(fig)

        print("✅ Frequency domain plots completed!")

    # Clean up memory
    plt.close("all")
    gc.collect()


#  TODO UPDATE THIS FUNCTION
def plot_2d_data(
    ax1: np.array,
    ax2: np.array,
    data: np.array,
    data_dict: dict,
    plot_config: dict,
    output_dir: Path,
) -> None:
    """Plot 2D spectroscopy data using standardized data structure.

    Args:
    ax1: 1D array of coherence times
    ax2: 1D array of detection times
    data: 1D array of averaged data values
    data_dict: Dictionary containing additional data (e.g., data_config and system parameters)
    plot_config: Plotting configuration
    output_dir: Directory to save plots
    """
    # Extract data from standardized structure
    tau_coh_vals = ax1
    t_det_vals = ax2
    system = data_dict["system"]
    data_config = data_dict["data_config"]
    tau_coh = data_config["tau_coh"]
    T_wait = data_config["T_wait"]

    # Extract data from standardized structure

    # Get configuration values
    spectral_components = plot_config.get("spectral_components_to_plot", ["abs"])
    extend_for = plot_config.get("extend_for", (1, 1))
    section = plot_config.get("section", (1.4, 1.8, 1.4, 1.8))

    print(f"✅ 2D data loaded successfully!")

    # Plot time domain data
    if plot_config.get("plot_time_domain", True):
        try:
            fig = plot_2d_el_field(
                data_x=t_det_vals,
                data_y=tau_coh_vals,
                data_z=data,
                t_wait=T_wait,
                domain="time",
                use_custom_colormap=True,
            )
            filename = generate_unique_plot_filename(
                system=system, config=data_config, domain="time"
            )
            save_fig(fig, filename=filename, output_dir=output_dir)
            plt.close(fig)
        except Exception as e:
            print(f"❌ Error in 2D time domain plotting: {e}")

    # Handle frequency domain processing
    if plot_config.get("plot_frequency_domain", True):
        try:
            # Extend time axes if needed
            if extend_for != (1, 1):
                extended_ts, extended_taus, extended_data = extend_time_axes(
                    data=data,
                    t_det=t_det_vals,
                    tau_coh=tau_coh_vals,
                    pad_t_det=extend_for,
                    pad_tau_coh=extend_for,
                )
            else:
                extended_ts, extended_taus, extended_data = (
                    t_det_vals,
                    tau_coh_vals,
                    data,
                )

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
                filename = generate_unique_plot_filename(
                    system=system,
                    config=data_config,
                    domain="freq",
                    component=component,
                )
                save_fig(fig, filename=filename, output_dir=output_dir)
                plt.close(fig)

        except Exception as e:
            print(f"❌ Error plotting 2D {component} component: {e}")

    # Clean up memory
    plt.close("all")
    gc.collect()


# FROM HERE ON: not used in the current workflow
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


def load_data_from_paths(data_path: Path, info_path: Path) -> dict:
    """
    Load simulation data from specific data and info file paths.

    Args:
        data_path: Path to the numpy data file (.npz)
        info_path: Path to the info file (.pkl)

    Returns:
        dict: Dictionary containing loaded data, axes, system, and config
    """

    # Load data file (numpy format)
    try:
        with np.load(data_path, allow_pickle=True) as data_file:
            data_dict = {key: data_file[key] for key in data_file.files}
        print(f"✅ Loaded data from: {data_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load data from {data_path}: {e}")
        raise

    # Load info file (pickle format)
    try:
        with open(info_path, "rb") as info_file:
            info_dict = pickle.load(info_file)
        print(f"✅ Loaded info from: {info_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load info from {info_path}: {e}")
        raise  # Combine data and info into standardized structure
    result = {
        "data": data_dict.get("data"),
        "axes": {
            "axs1": data_dict.get("axis1"),  # Note: saved as 'axis1', loaded as 'axs1'
        },
        "system": info_dict.get("system"),
        "config": info_dict.get("config"),
        "data_config": info_dict.get(
            "config"
        ),  # Alias for compatibility with plot_1d_data
    }

    # Add second axis if it exists (for 2D data)
    if "axis2" in data_dict:
        result["axes"]["axs2"] = data_dict["axis2"]

    return result
