"""
Simulation utilities for qspectro2d.

This module provides utility functions for simulation setup,
parallel processing configuration, and reporting.
"""

# =============================
# IMPORTS
# =============================
import os
import psutil
from pathlib import Path

### Project-specific imports
from qspectro2d.core.system_parameters import SystemParameters


# =============================
# SIMULATION SETUP UTILITIES
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


def create_system_parameters(data_config: dict) -> SystemParameters:
    """
    Create a SystemParameters object from a configuration dictionary.

    Parameters:
    data_config (dict): Configuration dictionary containing simulation parameters.

    Returns:
    SystemParameters: Initialized SystemParameters object.

    Raises:
    KeyError: If required keys for time configuration are missing.
    """
    # Handle different time configurations for 1D vs 2D
    if "tau_coh" in data_config:
        # 1D configuration
        t_max = (
            data_config["tau_coh"] + data_config["T_wait"] + data_config["t_det_max"]
        )
    else:
        # 2D configuration
        t_max = data_config["T_wait"] + 2 * data_config["t_det_max"]

    # Set all parameters to defaults if not provided in data_config
    return SystemParameters(
        t_max=t_max,
        Temp=data_config.get("Temp", 0),
        cutoff_=data_config.get("cutoff_", 1),
        N_atoms=data_config.get("N_atoms", 1),
        ODE_Solver=data_config.get("ODE_Solver", "Paper_eqs"),
        RWA_laser=data_config.get("RWA_laser", True),
        dt=data_config.get("dt", 1),
        bath=data_config.get("bath", "paper"),
        E0=data_config.get("E0", 1),
        envelope_type=data_config.get("envelope_type", "gaussian"),
        pulse_fwhm=data_config.get("pulse_fwhm", 15),
        omega_laser_cm=data_config.get("omega_laser_cm", 16000),
        Delta_cm=data_config.get("Delta_cm", 0),
        omega_A_cm=data_config.get("omega_A_cm", 16000),
        mu_A=data_config.get("mu_A", 1),
        omega_B_cm=data_config.get("omega_B_cm", 16000),
        mu_B=data_config.get("mu_B", 1),
        J_cm=data_config.get("J_cm", 0),
        gamma_0=data_config.get("gamma_0", 1 / 300),
        gamma_phi=data_config.get("gamma_phi", 1 / 100),
    )


# =============================
# REPORTING FUNCTIONS
# =============================
def print_simulation_header(data_config: dict, max_workers: int, simulation_type: str):
    """Print simulation header with configuration info."""
    title = f"{simulation_type.upper()} ELECTRONIC SPECTROSCOPY SIMULATION"
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"Configuration:")
    print(
        f"  Parameters: #phases={data_config['n_phases']}, #frequencies={data_config['n_freqs']}"
    )
    print(
        f"  Times: t_det_max={data_config['t_det_max']} fs, dt={data_config['dt']} fs"
    )
    if simulation_type == "1d":
        print(
            f"  Times: τ_coh={data_config['tau_coh']} fs, T_wait={data_config['T_wait']} fs"
        )

    print(
        f"  Total combinations: {data_config['n_phases'] * data_config['n_phases'] * data_config['n_freqs']}"
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
