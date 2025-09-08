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
from pathlib import Path
from qutip import OhmicEnvironment
from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
from qspectro2d.core.simulation import SimulationConfig, SimulationModuleOQS
from qspectro2d.config.models import MasterConfig
from qspectro2d.config.loader import load_config


# =============================
# SIMULATION CONFIGURATION UTILITIES
# =============================
def create_base_sim_oqs(
    args,
    cfg: MasterConfig | None = None,
) -> tuple[SimulationModuleOQS, float]:
    """Create base simulation instance and perform solver validation once.

    Parameters:
        args: Parsed command line arguments

    Returns:
        tuple: (SimulationModuleOQS instance, time_cut from solver validation)
    """
    # Resolve configuration: use provided cfg, else load defaults
    CONFIG = cfg if cfg is not None else load_config()

    print("🔧 Creating base simulation configuration...")

    # Validate merged configuration (defaults or file-based)
    print("🔎 Validating configuration...")
    CONFIG.validate()

    # Resolve precedence: args > YAML > defaults
    t_coh_res = (
        args.t_coh
        if getattr(args, "t_coh", None) is not None
        else getattr(CONFIG.window, "t_coh", 0.0)
    )
    t_wait_res = (
        args.t_wait
        if getattr(args, "t_wait", None) is not None
        else getattr(CONFIG.window, "t_wait", 0.0)
    )
    tdet_res = (
        args.t_det_max
        if getattr(args, "t_det_max", None) is not None
        else CONFIG.window.t_det_max
    )
    dt_res = args.dt if getattr(args, "dt", None) is not None else CONFIG.window.dt

    atomic_config = {
        "n_atoms": CONFIG.atomic.n_atoms,
        "n_chains": getattr(CONFIG.atomic, "n_chains", None),
        "frequencies_cm": list(CONFIG.atomic.frequencies_cm),
        "dip_moments": list(CONFIG.atomic.dip_moments),
        "delta_cm": getattr(CONFIG.atomic, "delta_cm", None),
        "max_excitation": getattr(CONFIG.atomic, "max_excitation", 1),
    }
    if (
        CONFIG.atomic.n_atoms >= 2
        and getattr(CONFIG.atomic, "coupling_cm", None) is not None
    ):
        atomic_config["coupling_cm"] = CONFIG.atomic.coupling_cm

    # Use dummy t_coh for initial setup and solver check
    pulse_config = {
        "pulse_fwhm": CONFIG.laser.pulse_fwhm_fs,
        "base_amplitude": CONFIG.laser.base_amplitude,
        "envelope_type": CONFIG.laser.envelope_type,
        "carrier_freq_cm": CONFIG.laser.carrier_freq_cm,
        "relative_E0s": list(CONFIG.signal.relative_e0s),
        "delays": [t_coh_res, t_wait_res],
    }

    max_workers = get_max_workers()
    simulation_config_dict = {
        "simulation_type": "1d",
        "max_workers": max_workers,
        "IFT_component": list(CONFIG.signal.ift_component),
        ### Simulation parameters
        "ode_solver": CONFIG.solver.solver,
        "rwa_sl": CONFIG.laser.rwa_sl,
        "keep_track": "basis",
        # times
        "t_coh": t_coh_res,
        "t_wait": t_wait_res,
        "t_det_max": tdet_res,
        "dt": dt_res,
        # phase cycling
        "n_phases": CONFIG.signal.n_phases,
        # inhomogeneous broadening
        "n_freqs": CONFIG.atomic.n_freqs,
    }

    bath_config = {
        "bath_type": CONFIG.bath.bath_type,
        "Temp": CONFIG.bath.temperature,
        "cutoff": CONFIG.bath.cutoff,
        "alpha": CONFIG.bath.coupling,
    }

    # Create the simulation class instance (inline, simple and explicit)
    system = AtomicSystem.from_dict(atomic_config)
    laser = LaserPulseSequence.from_delays(**pulse_config)
    bath = OhmicEnvironment(
        T=bath_config["Temp"],
        alpha=bath_config["alpha"] / bath_config["cutoff"], # NOTE this is now exactly the paper implementation 
        wc=bath_config["cutoff"],
        s=1.0,
        tag=bath_config["bath_type"],
    )
    sim_oqs = SimulationModuleOQS(
        simulation_config=SimulationConfig(**simulation_config_dict),
        system=system,
        laser=laser,
        bath=bath,
    )

    # print(sim_oqs.simulation_config)

    ### Validate solver once at the beginning
    time_cut = -np.inf
    t_max = sim_oqs.simulation_config.t_max
    print("🔍 Validating solver...")
    try:
        # Import here to avoid circular import
        from qspectro2d.spectroscopy.calculations import check_the_solver
        _, time_cut = check_the_solver(sim_oqs)
        print("#" * 60)
        print(
            f"✅ Solver validation worked: Evolution becomes unphysical at "
            f"({time_cut / t_max:.2f} × t_max)"
        )
    except Exception as e:
        print(f"⚠️  WARNING: Solver validation failed: {e}")

    if time_cut < t_max:
        print(
            f"⚠️  WARNING: Time cut {time_cut} is less than the last time point "
            f"{t_max}. This may affect the simulation results.",
            flush=True,
        )

    return sim_oqs, time_cut


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


def print_simulation_summary(
    elapsed_time: float, result_data, abs_path: str, simulation_type: str
):
    """Print simulation completion summary."""
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    print("The data with shape: ")
    if simulation_type == "1d":
        print(f"{result_data.shape}")
    elif simulation_type == "2d":
        print(f"{result_data[0].shape}")
    parent_dir = Path(abs_path).parent
    print(f"was saved to: {parent_dir}")
