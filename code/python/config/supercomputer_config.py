"""
Supercomputer-specific configuration for 2D Electronic Spectroscopy.

This configuration is optimized for HPC environments with:
- High CPU count
- Large memory availability
- Batch job scheduling
"""

from dataclasses import dataclass
from typing import Optional
import psutil
import os


@dataclass
class SupercomputerConfig:
    """Configuration optimized for supercomputer environments."""

    # HPC-optimized simulation parameters
    n_times_T: int = 10  # More T_wait values for comprehensive data
    n_phases: int = 4  # Full phase cycling
    n_freqs: int = 20  # Higher frequency sampling for better statistics

    # System parameters optimized for parallel execution
    N_atoms: int = 1
    ODE_Solver: str = "Paper_eqs"
    RWA_laser: bool = True
    t_max: float = 200.0  # Longer simulation time
    dt: float = 0.05  # Higher resolution

    # HPC-specific settings
    max_workers: Optional[int] = None  # Will auto-detect from SLURM
    use_all_cpus: bool = True
    memory_limit_gb: Optional[float] = None

    # Output configuration
    save_raw_data: bool = True
    data_subdir: str = "2d_spectroscopy/hpc_run"
    compress_output: bool = True  # Compress large datasets

    # Batch job settings
    batch_size: int = 100  # Process in batches to manage memory
    checkpoint_interval: int = 50  # Save intermediate results

    def __post_init__(self):
        """Configure HPC-specific parameters."""
        # Auto-detect CPU count from SLURM or system
        if self.max_workers is None:
            slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
            if slurm_cpus:
                self.max_workers = int(slurm_cpus)
            elif self.use_all_cpus:
                self.max_workers = psutil.cpu_count(logical=True)
            else:
                self.max_workers = min(16, psutil.cpu_count(logical=True))

        # Auto-detect memory limit from SLURM
        if self.memory_limit_gb is None:
            slurm_mem = os.getenv("SLURM_MEM_PER_NODE")
            if slurm_mem:
                # Convert from MB to GB
                self.memory_limit_gb = int(slurm_mem) / 1024

        # Calculate derived parameters
        self.Delta_cm = 200 if self.n_freqs > 1 else 0
        self.T_wait_max = self.t_max / 10
        self.total_combinations = (
            self.n_times_T * self.n_phases * self.n_phases * self.n_freqs
        )


def get_hpc_info():
    """Get information about the HPC environment."""
    info = {
        "hostname": os.getenv("HOSTNAME", "unknown"),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_cpus": os.getenv("SLURM_CPUS_PER_TASK"),
        "slurm_mem": os.getenv("SLURM_MEM_PER_NODE"),
        "slurm_nodes": os.getenv("SLURM_NNODES"),
        "total_cpus": psutil.cpu_count(logical=True),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
    }
    return info


# Predefined configurations for different scenarios
QUICK_TEST = SupercomputerConfig(
    n_times_T=2,
    n_phases=2,
    n_freqs=1,
    t_max=50.0,
    dt=0.1,
    data_subdir="2d_spectroscopy/quick_test",
)

PRODUCTION_RUN = SupercomputerConfig(
    n_times_T=20,
    n_phases=4,
    n_freqs=50,
    t_max=500.0,
    dt=0.02,
    data_subdir="2d_spectroscopy/production",
)

HIGH_RESOLUTION = SupercomputerConfig(
    n_times_T=50,
    n_phases=4,
    n_freqs=100,
    t_max=1000.0,
    dt=0.01,
    data_subdir="2d_spectroscopy/high_res",
)
