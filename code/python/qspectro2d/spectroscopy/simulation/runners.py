"""
Simulation runners for qspectro2d.

This module provides high-level functions for running 1D and 2D
spectroscopy simulations with proper error handling and validation.
"""

# =============================
# IMPORTS
# =============================
import numpy as np

### Project-specific imports
from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.core.simulation_class import SimulationModuleOQS


# =============================
# SIMULATION RUNNER FUNCTIONS
# =============================
def run_1d_simulation(sim_oqs: SimulationModuleOQS) -> tuple:
    """
    Run 1D spectroscopy simulation with updated calculation structure.

    Parameters:
        sim_oqs (SimulationModuleOQS): Simulation class containing system and configuration.
    Returns:
        tuple: Detection time values and averaged data.
    """
    from qspectro2d.spectroscopy.calculations import (
        parallel_compute_1d_E_with_inhomogenity,
        check_the_solver,
    )

    ### Create time arrays
    t_max = sim_oqs.simulation_config.t_max

    ### Validate solver
    time_cut = -np.inf
    try:
        _, time_cut = check_the_solver(sim_oqs)
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
        data = parallel_compute_1d_E_with_inhomogenity(
            sim_oqs=sim_oqs,
            time_cut=time_cut,
        )
        print("✅ Parallel computation completed successfully!")
        return data
    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise


def run_2d_simulation(
    info_config: dict, system: AtomicSystem, max_workers: int
) -> tuple:
    """
    Run 2D spectroscopy simulation with updated calculation structure.

    Parameters:
        info_config: Dictionary containing simulation parameters.
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
        t_coh_vals, t_det_vals, data_2d = parallel_compute_2d_E_with_inhomogenity(
            n_freqs=info_config.get("n_freqs", 1),
            n_phases=info_config["n_phases"],
            t_wait=info_config["t_wait"],
            t_det_max=info_config["t_det_max"],
            apply_ift=info_config.get("apply_ift", True),
            system=system,
            max_workers=max_workers,
            **kwargs,
        )
        print("✅ Parallel computation completed successfully!")
        return t_coh_vals, t_det_vals, data_2d
    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise
