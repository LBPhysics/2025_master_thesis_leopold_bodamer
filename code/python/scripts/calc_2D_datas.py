"""
Modernized 2D Electronic Spectroscopy Simulation Script

This script computes 2D electronic spectroscopy data using parallel processing
and saves results in pickle format for simple data storage.
"""

# Import the outsourced settings / functions
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
import pickle
import copy
import time
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SimulationConfig:
    """Configuration class for 2D spectroscopy simulation."""

    # Simulation parameters
    n_times_T: int = 1  # Number of T_wait values
    n_phases: int = 2  # Number of phases for phase cycling
    n_freqs: int = 2  # Number of frequencies for inhomogeneous broadening

    # System parameters
    N_atoms: int = 1
    ODE_Solver: str = "Paper_eqs"
    RWA_laser: bool = True
    t_max: float = 100.0  # determines Δω
    dt: float = 0.1  # determines ωₘₐₓ

    # Output configuration
    save_raw_data: bool = True
    data_subdir: str = "2d_spectroscopy"
    T_wait_max: float = t_max / 10

    def __post_init__(self):
        """Calculate derived parameters."""
        self.Delta_cm = 200 if self.n_freqs > 1 else 0
        self.total_combinations = (
            self.n_times_T * self.n_phases * self.n_phases * self.n_freqs
        )


def create_output_directory(config: SimulationConfig) -> Path:
    """Create and return the output directory path."""
    if config.save_raw_data:
        output_dir = DATA_DIR / "raw" / config.data_subdir
    else:
        output_dir = DATA_DIR / "processed" / config.data_subdir

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_unique_filename(
    output_dir: Path, config: SimulationConfig, system: SystemParameters
) -> Path:
    """Generate a unique filename for the output data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = (
        f"2d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
        f"T{config.n_times_T}_ph{config.n_phases}_"
        f"freq{config.n_freqs}_{timestamp}.pkl"
    )

    save_path = output_dir / base_filename

    # Ensure unique filename
    counter = 1
    while save_path.exists():
        name_with_counter = (
            f"2d_data_tmax_{system.t_max:.0f}_dt_{system.dt:.1f}_"
            f"T{config.n_times_T}_ph{config.n_phases}_"
            f"freq{config.n_freqs}_{timestamp}_{counter}.pkl"
        )
        save_path = output_dir / name_with_counter
        counter += 1

    return save_path


def print_simulation_summary(
    config: SimulationConfig,
    system: SystemParameters,
    two_d_datas: List,
    times: np.ndarray,
    elapsed_time: float,
    save_path: Path,
) -> None:
    """Print simple simulation summary."""
    # Calculate basic stats
    total_data_points = sum(
        data.size if data is not None else 0 for data in two_d_datas
    )
    file_size_mb = save_path.stat().st_size / (1024**2) if save_path.exists() else 0

    print(f"✅ Simulation completed in {elapsed_time:.1f}s")
    print(system.summary())
    print(f"Configuration: {config}")
    print(f"📊 Data: {len(two_d_datas)} datasets, {total_data_points:,} points total")
    print(f"💾 Saved: {save_path.name} ({file_size_mb:.2f} MB)")
    print(f"📁 Location: {save_path.parent}")


def main(config: Optional[SimulationConfig] = None):
    """
    Main function to run the modernized 2D spectroscopy simulation.

    Parameters
    ----------
    config : SimulationConfig, optional
        Simulation configuration. If None, uses default configuration.
    """
    if config is None:
        config = SimulationConfig()

    start_time = time.time()

    # Set up output directory
    output_dir = create_output_directory(config)

    # Generate random subset of phases from the full set
    all_phases = [k * np.pi / 2 for k in range(4)]  # [0, π/2, π, 3π/2]
    phases = np.random.choice(all_phases, size=config.n_phases, replace=False).tolist()
    max_workers = psutil.cpu_count(logical=True)

    # =============================
    # SYSTEM PARAMETERS
    # =============================
    system = SystemParameters(
        N_atoms=config.N_atoms,
        ODE_Solver=config.ODE_Solver,
        RWA_laser=config.RWA_laser,
        t_max=config.t_max,
        dt=config.dt,
        Delta_cm=config.Delta_cm,
    )

    # Create time arrays
    FWHMs = system.FWHMs
    times = np.arange(-FWHMs[0], system.t_max, system.dt)

    times_T = np.linspace(0, config.T_wait_max, config.n_times_T)

    # =============================
    # SOLVER VALIDATION
    # =============================
    test_system = copy.deepcopy(system)
    test_system.t_max = 10 * system.t_max
    test_system.dt = 10 * system.dt
    times_test = np.arange(-FWHMs[0], test_system.t_max, test_system.dt)

    try:
        _, time_cut = check_the_solver(times_test, test_system)
    except Exception as e:
        print(f"⚠️  WARNING: Solver validation failed: {e}")
        time_cut = 0

    # =============================
    # FREQUENCY SAMPLING
    # =============================
    omega_ats = sample_from_sigma(
        config.n_freqs, FWHM=system.Delta_cm, mu=system.omega_A_cm
    )

    # =============================
    # RUN SIMULATION
    # =============================
    kwargs = {"plot_example": False, "time_cut": time_cut}

    try:
        two_d_datas = parallel_compute_2d_polarization_with_inhomogenity(
            omega_ats=omega_ats,
            phases=phases,
            times_T=times_T,
            times=times,
            system=system,
            max_workers=max_workers,
            **kwargs,
        )

        if two_d_datas and two_d_datas[0] is not None:
            pass  # Data shape logging removed for simplicity

    except Exception as e:
        print(f"❌ ERROR: Simulation failed: {e}")
        raise

    # =============================
    # SAVE RESULTS
    # =============================
    try:
        save_path = generate_unique_filename(output_dir, config, system)

        # Use the utility function for saving
        data = {
            "times": times,
            "times_T": times_T,
            "system": system,
            "two_d_datas": two_d_datas,
        }

        # Save as pickle
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    except Exception as e:
        print(f"❌ ERROR: Failed to save data: {e}")
        raise

    # =============================
    # SIMULATION SUMMARY
    # =============================
    elapsed_time = time.time() - start_time
    print_simulation_summary(
        config, system, two_d_datas, times, elapsed_time, save_path
    )

    return {
        "save_path": save_path,
        "config": config,
        "system": system,
        "execution_time": elapsed_time,
        "data_shape": [
            data.shape if data is not None else None for data in two_d_datas
        ],
    }


def run_with_custom_config():
    """Run simulation with custom configuration for testing different parameters."""

    # Example: High-resolution simulation
    high_res_config = SimulationConfig(
        n_times_T=3,
        n_phases=2,
        n_freqs=5,
        t_max=20.0,
        dt=0.1,
        data_subdir="2d_spectroscopy/high_resolution",
    )

    # Example: Quick test simulation
    test_config = SimulationConfig(
        n_times_T=1,
        n_phases=1,
        n_freqs=1,
        t_max=5.0,
        dt=0.5,
        data_subdir="2d_spectroscopy/test",
    )

    # Choose configuration
    config = high_res_config  # Change this to high_res_config for detailed simulation

    print(f"Running simulation with configuration: {config}")
    result = main(config)

    if result:
        print(f"\nSimulation completed successfully!")
        print(f"Results saved to: {result['save_path']}")
    else:
        print("Simulation failed!")


if __name__ == "__main__":
    # Check command line arguments for configuration
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Quick test run
            config = SimulationConfig(
                n_times_T=1,
                n_phases=1,
                n_freqs=1,
                t_max=5.0,
                dt=0.5,
                data_subdir="2d_spectroscopy/test",
            )
            main(config)
        elif sys.argv[1] == "custom":
            # Run with custom configurations
            run_with_custom_config()
        else:
            # Default run
            main()
    else:
        # Default run
        main()
