# TODO MAKE THIS WAY SIMPLER
#!/usr/bin/env python3
"""
HPC-Optimized 2D Electronic Spectroscopy Simulation Script

This script is specifically designed for supercomputer environments with:
- SLURM job scheduling integration
- Optimized memory usage
- Checkpoint/restart capability
- Comprehensive logging
"""

import os
import sys
import time
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.supercomputer_config import (
    SupercomputerConfig,
    get_hpc_info,
    QUICK_TEST,
    PRODUCTION_RUN,
)
from src.spectroscopy.post_processing import *
from src.core.system_parameters import SystemParameters
from src.spectroscopy.calculations import (
    parallel_compute_2d_polarization_with_inhomogenity,
    check_the_solver,
)
from src.spectroscopy.inhomogenity import sample_from_sigma
from config.paths import DATA_DIR, FIGURES_DIR

import numpy as np
import psutil
import copy


def setup_logging(config: SupercomputerConfig) -> logging.Logger:
    """Set up comprehensive logging for HPC environment."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create log filename with job info
    job_id = os.getenv("SLURM_JOB_ID", "local")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"2des_simulation_{job_id}_{timestamp}.log"

    # Configure logger
    logger = logging.getLogger("2des_hpc")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_hpc_environment(logger: logging.Logger) -> dict:
    """Log HPC environment information."""
    hpc_info = get_hpc_info()

    logger.info("🖥️  HPC ENVIRONMENT INFORMATION")
    logger.info("=" * 50)
    for key, value in hpc_info.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)

    return hpc_info


def create_checkpoint(data: dict, checkpoint_dir: Path, step: str) -> None:
    """Create checkpoint file for restart capability."""
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_dir / f"checkpoint_{step}.pkl"

    with open(checkpoint_file, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(checkpoint_dir: Path, step: str) -> Optional[dict]:
    """Load checkpoint data if available."""
    checkpoint_file = checkpoint_dir / f"checkpoint_{step}.pkl"

    if checkpoint_file.exists():
        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)
    return None


def validate_memory_usage(
    logger: logging.Logger, max_memory_gb: Optional[float] = None
) -> bool:
    """Monitor memory usage and warn if approaching limits."""
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent_used = memory.percent

    logger.info(
        f"💾 Memory usage: {used_gb:.1f}GB / {total_gb:.1f}GB ({percent_used:.1f}%)"
    )

    if max_memory_gb and used_gb > max_memory_gb * 0.9:
        logger.warning(f"⚠️  Memory usage approaching limit ({max_memory_gb}GB)")
        return False

    if percent_used > 85:
        logger.warning(f"⚠️  High memory usage: {percent_used:.1f}%")
        return False

    return True


def main_hpc(
    config: Optional[SupercomputerConfig] = None, config_name: str = "default"
):
    """
    Main function optimized for HPC environments.

    Parameters
    ----------
    config : SupercomputerConfig, optional
        Simulation configuration
    config_name : str
        Name of predefined configuration to use
    """

    # Set up configuration
    if config is None:
        if config_name == "test":
            config = QUICK_TEST
        elif config_name == "production":
            config = PRODUCTION_RUN
        else:
            config = SupercomputerConfig()

    # Set up logging
    logger = setup_logging(config)
    logger.info(
        f"🚀 Starting 2D Electronic Spectroscopy simulation (config: {config_name})"
    )

    # Log environment
    hpc_info = log_hpc_environment(logger)

    # Create output directories
    output_dir = DATA_DIR / "raw" / config.data_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    start_time = time.time()

    try:
        # =============================
        # SYSTEM SETUP
        # =============================
        logger.info("⚙️  Setting up system parameters...")

        system = SystemParameters(
            N_atoms=config.N_atoms,
            ODE_Solver=config.ODE_Solver,
            RWA_laser=config.RWA_laser,
            t_max=config.t_max,
            dt=config.dt,
            Delta_cm=config.Delta_cm,
        )

        logger.info(f"System: {system.summary()}")

        # Generate time arrays
        FWHMs = system.FWHMs
        times = np.arange(-FWHMs[0], system.t_max, system.dt)
        times_T = np.linspace(0, config.T_wait_max, config.n_times_T)

        logger.info(f"Time points: {len(times)}, T_wait points: {len(times_T)}")

        # =============================
        # SOLVER VALIDATION
        # =============================
        logger.info("🔍 Validating solver...")

        checkpoint_data = load_checkpoint(checkpoint_dir, "solver_validation")
        if checkpoint_data:
            logger.info("📁 Loading solver validation from checkpoint")
            time_cut = checkpoint_data["time_cut"]
        else:
            test_system = copy.deepcopy(system)
            test_system.t_max = 10 * system.t_max
            test_system.dt = 10 * system.dt
            times_test = np.arange(-FWHMs[0], test_system.t_max, test_system.dt)

            try:
                _, time_cut = check_the_solver(times_test, test_system)
                logger.info(f"✅ Solver validation successful (time_cut: {time_cut})")

                # Save checkpoint
                create_checkpoint(
                    {"time_cut": time_cut}, checkpoint_dir, "solver_validation"
                )

            except Exception as e:
                logger.warning(f"⚠️  Solver validation failed: {e}")
                time_cut = 0

        # =============================
        # FREQUENCY SAMPLING
        # =============================
        logger.info("🎵 Setting up frequency sampling...")

        omega_ats = sample_from_sigma(
            config.n_freqs, FWHM=system.Delta_cm, mu=system.omega_A_cm
        )

        logger.info(f"Frequency samples: {len(omega_ats)}")

        # Generate phases
        all_phases = [k * np.pi / 2 for k in range(4)]
        phases = np.random.choice(
            all_phases, size=config.n_phases, replace=False
        ).tolist()

        logger.info(f"Phases: {config.n_phases} ({phases})")
        logger.info(f"Total combinations: {config.total_combinations}")

        # =============================
        # MEMORY CHECK
        # =============================
        if not validate_memory_usage(logger, config.memory_limit_gb):
            logger.error("❌ Insufficient memory for simulation")
            return None

        # =============================
        # MAIN SIMULATION
        # =============================
        logger.info("🎯 Starting main simulation...")

        kwargs = {"plot_example": False, "time_cut": time_cut}

        simulation_start = time.time()

        two_d_datas = parallel_compute_2d_polarization_with_inhomogenity(
            omega_ats=omega_ats,
            phases=phases,
            times_T=times_T,
            times=times,
            system=system,
            max_workers=config.max_workers,
            **kwargs,
        )

        simulation_time = time.time() - simulation_start
        logger.info(f"✅ Simulation completed in {simulation_time:.1f}s")

        # =============================
        # SAVE RESULTS
        # =============================
        logger.info("💾 Saving results...")

        # Generate filename
        job_id = os.getenv("SLURM_JOB_ID", "local")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"2d_data_{job_id}_{timestamp}.pkl"
        save_path = output_dir / filename

        # Prepare data
        results = {
            "times": times,
            "times_T": times_T,
            "system": system,
            "config": config,
            "two_d_datas": two_d_datas,
            "hpc_info": hpc_info,
            "execution_time": simulation_time,
            "total_time": time.time() - start_time,
        }

        # Save with compression if requested
        if config.compress_output:
            import gzip

            with gzip.open(str(save_path) + ".gz", "wb") as f:
                pickle.dump(results, f)
            save_path = Path(str(save_path) + ".gz")
        else:
            with open(save_path, "wb") as f:
                pickle.dump(results, f)

        # =============================
        # FINAL SUMMARY
        # =============================
        total_time = time.time() - start_time
        file_size_mb = save_path.stat().st_size / (1024**2)

        logger.info("🎉 SIMULATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"📊 Results: {len(two_d_datas)} datasets")
        logger.info(f"⏱️  Total time: {total_time:.1f}s")
        logger.info(f"⏱️  Simulation time: {simulation_time:.1f}s")
        logger.info(f"💾 File size: {file_size_mb:.2f} MB")
        logger.info(f"📁 Saved to: {save_path}")
        logger.info("=" * 50)

        return {
            "save_path": save_path,
            "config": config,
            "system": system,
            "execution_time": total_time,
            "data_shape": [
                data.shape if data is not None else None for data in two_d_datas
            ],
        }

    except Exception as e:
        logger.error(f"❌ SIMULATION FAILED: {e}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HPC 2D Electronic Spectroscopy Simulation"
    )
    parser.add_argument(
        "--config",
        default="default",
        choices=["default", "test", "production"],
        help="Configuration preset to use",
    )
    parser.add_argument(
        "--max-workers", type=int, help="Maximum number of worker processes"
    )

    args = parser.parse_args()

    # Override max_workers if specified
    if args.config == "test":
        config = QUICK_TEST
    elif args.config == "production":
        config = PRODUCTION_RUN
    else:
        config = SupercomputerConfig()

    if args.max_workers:
        config.max_workers = args.max_workers

    main_hpc(config, args.config)
