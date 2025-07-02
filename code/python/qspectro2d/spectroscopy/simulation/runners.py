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
from qspectro2d.spectroscopy.calculations import (
    parallel_compute_1d_E_with_inhomogenity,
    parallel_compute_2d_E_with_inhomogenity,
    check_the_solver,
)
from qspectro2d.core.system_parameters import SystemParameters


# =============================
# SIMULATION RUNNER FUNCTIONS
# =============================
def run_1d_simulation(
    data_config: dict, system: SystemParameters, max_workers: int
) -> tuple:
    """
    Run 1D spectroscopy simulation with updated calculation structure.

    Parameters:
        data_config: Dictionary containing simulation parameters.
        system: System parameters object.
        max_workers: Number of parallel workers.

    Returns:
        tuple: Detection time values and averaged data.
    """
    ### Create time arrays
    tau_coh = data_config["tau_coh"]
    T_wait = data_config["T_wait"]
    t_det_max = data_config["t_det_max"]
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
            n_freqs=data_config["n_freqs"],
            n_phases=data_config["n_phases"],
            tau_coh=tau_coh,
            T_wait=T_wait,
            t_det_max=t_det_max,
            system=system,
            max_workers=max_workers,
            time_cut=time_cut,
            apply_ift=data_config.get("apply_ift", True)
        )
        print("✅ Parallel computation completed successfully!")
        return t_det_vals, data
    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise


def run_2d_simulation(
    data_config: dict, system: SystemParameters, max_workers: int
) -> tuple:
    """
    Run 2D spectroscopy simulation with updated calculation structure.

    Parameters:
        data_config: Dictionary containing simulation parameters.
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
            n_freqs=data_config.get("n_freqs", 1),
            n_phases=data_config["n_phases"],
            T_wait=data_config["T_wait"],
            t_det_max=data_config["t_det_max"],
            apply_ift=data_config.get(
                "apply_ift", True
            ),  # TODO also do this to 1d case
            system=system,
            max_workers=max_workers,
            **kwargs,
        )
        print("✅ Parallel computation completed successfully!")
        return tau_coh_vals, t_det_vals, data_2d
    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise
