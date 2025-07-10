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
import numpy as np

### Project-specific imports
from qspectro2d.core.atomic_system.system_class import AtomicSystem


# =============================
# SIMULATION SETUP UTILITIES
# =============================
def get_max_workers() -> int:
    """Get the maximum number of workers for parallel processing."""
    # Use SLURM environment variable if available, otherwise detect automatically
    try:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    except ValueError:
        slurm_cpus = 0

    local_cpus = psutil.cpu_count(logical=True) or 1
    return slurm_cpus if slurm_cpus > 0 else local_cpus


# NOT NEEDED with new SimulationConfigClass
def create_system_parameters(info_config: dict) -> AtomicSystem:
    """
    Create a AtomicSystem object from a configuration dictionary.

    Parameters:
    info_config (dict): Configuration dictionary containing simulation parameters.

    Returns:
    AtomicSystem: Initialized AtomicSystem object.

    Raises:
    KeyError: If required keys for time configuration are missing.
    """
    # Handle different time configurations for 1D vs 2D
    if "t_coh" in info_config:
        # 1D configuration
        t_max = info_config["t_coh"] + info_config["t_wait"] + info_config["t_det_max"]
    else:
        # 2D configuration
        t_max = info_config["t_wait"] + 2 * info_config["t_det_max"]

    # Set all parameters to defaults if not provided in info_config
    return AtomicSystem(
        t_max=t_max,
        Temp=info_config.get("Temp", 0),
        cutoff_=info_config.get("cutoff_", 1),
        N_atoms=info_config.get("N_atoms", 1),
        ODE_Solver=info_config.get("ODE_Solver", "Paper_eqs"),
        RWA_SL=info_config.get("RWA_SL", True),
        dt=info_config.get("dt", 1),
        bath=info_config.get("bath", "paper"),
        E0=info_config.get("E0", 1),
        envelope_type=info_config.get("envelope_type", "gaussian"),
        pulse_fwhm=info_config.get("pulse_fwhm", 15),
        omega_laser_cm=info_config.get("omega_laser_cm", 16000),
        Delta_cm=info_config.get("Delta_cm", 0),
        # omega_A_cm=info_config.get("omega_A_cm", 16000),
        mu_A=info_config.get("mu_A", 1),
        # omega_B_cm=info_config.get("omega_B_cm", 16000),
        mu_B=info_config.get("mu_B", 1),
        J_cm=info_config.get("J_cm", 0),
        gamma_0=info_config.get("gamma_0", 1 / 300),
        gamma_phi=info_config.get("gamma_phi", 1 / 100),
    )


# =============================
# REPORTING FUNCTIONS
# =============================
# NOT NEEDED with new SimulationConfigClass
def print_simulation_header(info_config: dict, max_workers: int):
    """Print simulation header with configuration info."""
    simulation_type = info_config.get("simulation_type", "spectroscopy")
    title = f"{simulation_type.upper()} ELECTRONIC SPECTROSCOPY SIMULATION"
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"Configuration:")
    print(
        f"  Parameters:\t#phases={info_config['n_phases']}, #frequencies={info_config['n_freqs']}"
    )
    print(
        f"  Times:\tt_det_max={info_config['t_det_max']} fs, dt={info_config['dt']} fs"
    )
    if simulation_type == "1d":
        print(f"\t\tτ_coh={info_config['t_coh']} fs")
    print(f"\t\tT_wait: {info_config['t_wait']} fs")
    print(
        f"  Total combinations: {info_config['n_phases'] * info_config['n_phases'] * info_config['n_freqs']}"
    )
    print(f"  Parallel workers: {max_workers}")
    print()


def print_simulation_summary(
    elapsed_time: float, result_data, rel_path: str, simulation_type: str
):
    """Print simulation completion summary."""
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    print("The data with shape: ")
    if simulation_type == "1d":
        print(f"{result_data.shape}")
    elif simulation_type == "2d":
        print(f"{result_data[0].shape}")
    print(f"was saved to: {rel_path}")


DEFAULT_SOLVER_OPTIONS = {  # very rough estimate, not optimized
    "nsteps": 200000,
    "atol": 1e-6,
    "rtol": 1e-4,
}
PHASE_CYCLING_PHASES = [0, np.pi / 2, np.pi, 3 * np.pi / 2]


# Define named constants for hardcoded values
NEGATIVE_EIGVAL_THRESHOLD = -1e-3
TRACE_TOLERANCE = 1e-6

# Fixed phase for detection pulse
DETECTION_PHASE = 0
