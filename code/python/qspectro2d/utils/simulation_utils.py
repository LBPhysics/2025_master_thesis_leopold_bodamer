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

from qspectro2d.core.bath_system.bath_class import BathSystem
from qspectro2d.core.simulation_class import (
    AtomicSystem,
    LaserPulseSequence,
    SimulationConfig,
    SimulationModuleOQS,
)


# =============================
# SIMULATION CONFIGURATION UTILITIES
# =============================
def create_simulation_module_from_configs(
    atom_config: dict,
    laser_config: dict,
    bath_config: dict,
    simulation_config: dict,
) -> SimulationModuleOQS:
    """
    Create a simulation module from the provided configuration dictionaries.

    Parameters:
        atom_config (dict): Atomic system configuration.
        laser_config (dict): Laser pulse sequence configuration.
        bath_config (dict): Bath parameters configuration.
        simulation_config (dict): Simulation parameters configuration.

    Returns:
        SimulationModuleOQS: Configured simulation class instance.
    """
    system = AtomicSystem.from_dict(atom_config)
    laser = LaserPulseSequence.from_delays(**laser_config)
    bath = BathSystem.from_dict(bath_config)

    return SimulationModuleOQS(
        simulation_config=SimulationConfig(**simulation_config),
        system=system,
        laser=laser,
        bath=bath,
    )


def create_base_sim_oqs(
    args, overwrite_config: dict = None
) -> tuple[SimulationModuleOQS, float]:
    """TODO write this function such that i can also overwrite the default values with a optional dictionary
    Create base simulation instance and perform solver validation once.

    Parameters:
        args: Parsed command line arguments

    Returns:
        tuple: (SimulationModuleOQS instance, time_cut from solver validation)
    """
    # Import config values here to avoid circular imports
    from qspectro2d.config import (
        N_ATOMS,
        ODE_SOLVER,
        RWA_SL,
        BATH_TYPE,
        BATH_TEMP,
        BATH_CUTOFF,
        BATH_GAMMA_0,
        BATH_GAMMA_PHI,
        N_FREQS,
        N_PHASES,
        DELTA_CM,
        IFT_COMPONENT,
        RELATIVE_E0S,
        PULSE_FWHM,
        BASE_AMPLITUDE,
        ENVELOPE_TYPE,
        CARRIER_FREQ_CM,
        DIP_MOMENTS,
        FREQS_CM,
        J_COUPLING_CM,
    )

    print("🔧 Creating base simulation configuration...")

    atomic_config = {
        "n_atoms": N_ATOMS,
        "freqs_cm": FREQS_CM,  # Frequency of atom A [cm⁻¹]
        "dip_moments": DIP_MOMENTS,  # Dipole moments for each atom
        "delta_cm": DELTA_CM,  # inhomogeneous broadening [cm⁻¹]
    }
    if N_ATOMS >= 2:
        atomic_config["J_cm"] = J_COUPLING_CM

    # Use dummy t_coh=0 for initial setup and solver check
    pulse_config = {
        "pulse_fwhm": PULSE_FWHM,
        "base_amplitude": BASE_AMPLITUDE,
        "envelope_type": ENVELOPE_TYPE,
        "carrier_freq_cm": CARRIER_FREQ_CM,
        "relative_E0s": RELATIVE_E0S,  # relative amplitudes for each pulse
        "delays": [
            args.t_coh,
            args.t_wait,
        ],  # dummy delays, will be updated
    }

    max_workers = get_max_workers()
    simulation_config_dict = {
        "simulation_type": "1d",
        "max_workers": max_workers,
        "IFT_component": IFT_COMPONENT,
        ### Simulation parameters
        "ode_solver": ODE_SOLVER,
        "rwa_sl": RWA_SL,
        "keep_track": "basis",
        # times
        "t_coh": args.t_coh,  # dummy value, will be updated
        "t_wait": args.t_wait,
        "t_det_max": args.t_det_max,
        "dt": args.dt,
        # phase cycling
        "n_phases": N_PHASES,
        # inhomogeneous broadening
        "n_freqs": N_FREQS,
    }

    bath_config = {
        ### Bath parameters
        "bath_type": BATH_TYPE,
        "Temp": BATH_TEMP,  # zero temperature
        "cutoff_": BATH_CUTOFF,
        "gamma_0": BATH_GAMMA_0,
        "gamma_phi": BATH_GAMMA_PHI,
    }

    # Create the simulation class instance
    sim_oqs = create_simulation_module_from_configs(
        atom_config=atomic_config,
        laser_config=pulse_config,
        bath_config=bath_config,
        simulation_config=simulation_config_dict,
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
