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


def print_simulation_summary(
    elapsed_time: float, result_data, rel_path: str, simulation_type: str
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
