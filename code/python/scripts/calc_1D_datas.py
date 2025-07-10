"""
1D Electronic Spectroscopy Simulation Script – Flexible Execution Mode

This script Type of pulse_seq:s 1D spectroscopy data for a given set of simulation parameters.
It supports two modes of execution:

# Determine whether to use batch mode or single t_coh mode based on the provided arguments.
    --simulation_type <type>: Type of simulation (default: "1d")
        -> "1d" Run the simulation for one specific coherence time
        -> "2d" Run the simulation for a range of coherence times, splitted into batches

# other arguments:
    --batch_idx <index>: Batch index for the current job (0 to n_batches-1, default: 0)
    --n_batches <total>: Total number of batches (default: 1)
    --t_det_max <fs>   : Maximum detection time (default: 600.0 fs)
    --t_wait <fs>      : Waiting time between pump and probe pulses (default: 0.0 fs)
    --t_coh <value>    : Coherence time between 2 pump pulses (default: 0.0 fs)
    --dt <fs>          : Spacing between t_coh and t_det values (default: 10.0 fs)

This script is designed for both local development and HPC batch execution.
Results are saved automatically using the qspectro2d I/O framework.
"""

# ==============
# ANALYSIS: for f in batch_*.slurm; do sbatch "$f"; done
# ==============
# FOR N_ATOMS = 1, ODE_SOLVER=BR, RWA=TRUE
# for t_det_max=100, dt=0.1, one 1d calculation takes about 2-3 seconds, so
# -> 1000 t_coh_vals: -> split into 20 batches=20jobs of 50 each, each job will take 3s * 50 = 150s = 2.5 minutes
# -> 1000 t_coh_vals: -> split into 10 batches=10jobs of 100 each, each job will take 3s * 100 = 300s = 5 minutes
# one 1d calculation (t_det_max=600, dt=0.1) takes about 3-4 seconds, so
# -> 6000 t_coh_vals: -> split into 20 batches=20jobs of 300 each, each job will take 5s * 300 = 1500s = 25 minutes
# -> 6000 t_coh_vals: -> split into 10 batches=10jobs of 600 each, each job will take 5s * 600 = 3000s = 50 minutes

# FOR N_ATOMS = 2, ODE_SOLVER=BR, RWA=TRUE
# for t_det_max=100, dt=0.1, one 1d calculation takes about 5-6 second, so
# -> 1000 t_coh_vals: -> split into 20 batches=20jobs of 50 each, each job will take 6s * 50 = 300s = 5 minutes
# -> 1000 t_coh_vals: -> split into 10 batches=10jobs of 100 each, each job will take 6s * 100 = 600s = 10 minutes
# one 1d calculation (t_det_max=600, dt=0.1) takes about 18-19 seconds, so
# -> 6000 t_coh_vals: -> split into 20 batches=20jobs of 300 each, each job will take 20s * 300 = 6000s = 100 minutes < 2 hours
# -> 6000 t_coh_vals: -> split into 10 batches=10jobs of 600 each, each job will take 20s * 600 = 12000s = 200 minutes < 4 hours, actually on proteus it ran for 6,... h
# Estimation for RAM: 6000 tau vals * 10 batches < 40,000,000.0 complex64 numbers < 40 MB << 1 GB

from encodings.punycode import T
import time
import argparse
import numpy as np
from pathlib import Path

from qspectro2d.core.bath_system.bath_class import BathClass
from qspectro2d.config import DATA_DIR
from qspectro2d.spectroscopy import (
    run_1d_simulation,
    get_max_workers,
    print_simulation_summary,
)
from qspectro2d.data import (
    save_data_file,
    save_info_file,
    generate_unique_data_filename,
)
from qspectro2d.core.simulation_class import (
    AtomicSystem,
    LaserPulseSequence,
    SimulationConfigClass,
    SimClassOQS,
)

N_ATOMS = 1  # Number of atoms in the system, can be changed to 1 or 2


def create_simulation_module_from_configs(
    atom_config: dict,
    laser_config: dict,
    bath_config: dict,
    simulation_config: dict,
) -> SimClassOQS:
    """
    Create a simulation module from the provided configuration dictionaries.

    Parameters:
        atom_config (dict): Atomic system configuration.
        laser_config (dict): Laser pulse sequence configuration.
        bath_config (dict): Bath parameters configuration.
        simulation_config (dict): Simulation parameters configuration.

    Returns:
        SimClassOQS: Configured simulation class instance.
    """
    system = AtomicSystem.from_dict(atom_config)
    laser = LaserPulseSequence.from_delays(**laser_config)
    bath = BathClass.from_dict(bath_config)

    return SimClassOQS(
        simulation_config=SimulationConfigClass(**simulation_config),
        system=system,
        laser=laser,
        bath=bath,
    )


def run_single_tau(
    t_coh: float, t_det_max: float, dt: float, t_wait: float, save_info: bool = False
) -> Path:
    print(f"\n=== Starting t_coh = {t_coh:.2f} fs ===")

    atomic_config = {
        # solver parameters
        "N_atoms": N_ATOMS,
        "freqs_cm": [16000],  # Frequency of atom A [cm⁻¹]
        "dip_moments": [1.0] * N_ATOMS,  # Dipole moments for each atom
        "Delta_cm": 0.0,  # inhomogeneous broadening [cm⁻¹]
    }
    if N_ATOMS >= 2:
        atomic_config["J_cm"] = 300.0

    pulse_config = {
        "pulse_fwhm": 15.0 if N_ATOMS == 1 else 5.0,
        "base_amplitude": 0.005,  # Rename "E0" to match function signature
        "envelope_type": "gaussian",  # fix typo: "envelope_types" → "envelope_type"
        "carrier_freq_cm": atomic_config["freqs_cm"][
            0
        ],  # Carrier frequency of the pulse
        "delays": [0.0, t_coh, t_coh + t_wait],
    }

    max_workers = get_max_workers()
    simulation_config_dict = {
        "simulation_type": "1d",  # Add simulation type to config
        "max_workers": max_workers,
        "apply_ift": True,  # Apply inverse Fourier transform to get the photon echo signal
        ### Simulation parameters
        # solver parameters
        "ODE_Solver": "Paper_eqs",
        "RWA_SL": True,
        "keep_track": "basis",  # or "basis" to track basis states
        # times
        "t_coh": t_coh,
        "t_wait": 0.0,
        "t_det_max": t_det_max,
        "dt": dt,
        # phase cycling
        "n_phases": 4,  # -> 4*4 = 16 parallel jobs!
        # inhomogeneous broadening
        "n_freqs": 1,  # increases parallel jobs!!
    }
    bath_config = {
        ### Bath parameters
        "bath_type": "paper",
        # Temperature / cutoff of the bath
        "Temp": 1e-5,  # zero temperature
        "cutoff_": 1e2,  # later * omega_A
        # decay  rates
        "gamma_0": 1 / 300.0,  # default value 1/300
        "gamma_phi": 1 / 100.0,  # default value 1e-2
    }

    # Create the simulation class instance
    sim_oqs = create_simulation_module_from_configs(
        atom_config=atomic_config,
        laser_config=pulse_config,
        bath_config=bath_config,
        simulation_config=simulation_config_dict,
    )

    start_time = time.time()
    print(sim_oqs.simulation_config)

    # NOW DO THE SIMULATION
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
        data = run_1d_simulation(sim_oqs)

    abs_path = Path(generate_unique_data_filename(system, simulation_config_dict))
    data_path = Path(f"{abs_path}_data.npz")
    print(f"\nSaving data to: {data_path}")
    save_data_file(data_path, data, sim_oqs.times_det)

    rel_path = abs_path.relative_to(DATA_DIR)

    if save_info:
        all_infos_as_dict = sim_oqs.to_dict()
        print(f"\nthe simulation info dict: {all_infos_as_dict}")
        info_path = Path(f"{abs_path}_info.pkl")
        save_info_file(
            info_path,
            system,
            bath=bath,
            laser=laser,
            info_config=simulation_config_dict,
        )

        elapsed = time.time() - start_time

        print_simulation_summary(
            elapsed, data, rel_path, "1d"
        )  # Print the paths for feed-forward to plotting script

        print(f"{'='*60}")
        print(f"\n🎯 To plot this data, run:")
        print(f'python plot_datas.py --rel_path "{rel_path}"')
        print(f"{'='*60}")

        # For shell scripts, we need absolute paths for file existence checks

    return rel_path.parent


def create_configs(args):
    """
    Create configuration dictionaries for the simulation based on command line arguments.

    Parameters:
        args: Parsed command line arguments.

    Returns:
        tuple: Configuration dictionaries for atomic system, bath, laser, and simulation.
    """
    # Atomic system configuration
    atomic_config = {
        "N_atoms": N_ATOMS,
        "freqs_cm": [16000],  # Frequency of atom A [cm⁻¹]
        "dip_moments": [1.0] * N_ATOMS,  # Dipole moments for each atom
        "Delta_cm": 0.0,  # Inhomogeneous broadening [cm⁻¹]
    }
    if N_ATOMS >= 2:
        atomic_config["J_cm"] = 300.0

    # Laser pulse sequence configuration
    pulse_config = {
        "pulse_fwhm": 15.0 if N_ATOMS == 1 else 5.0,
        "base_amplitude": 0.005,  # Rename "E0" to match function signature
        "envelope_type": "gaussian",  # fix typo: "envelope_types" → "envelope_type"
        "delays": [0.0, args.t_coh, args.t_coh + args.t_wait],
    }

    max_workers = get_max_workers()

    # Simulation configuration
    simulation_config = {
        "simulation_type": "1d",  # Add simulation type to config
        "max_workers": max_workers,
        "apply_ift": True,  # Apply inverse Fourier transform to get the photon echo signal
        ### Simulation parameters
        # solver parameters
        "ODE_Solver": "Paper_eqs",
        "RWA_SL": True,
        "keep_track": "basis",  # or "basis" to track basis states
        # times
        "t_coh": args.t_coh,
        "t_wait": args.t_wait,
        "t_det_max": args.t_det_max,
        "dt": args.dt,
        # phase cycling
        "n_phases": 4,  # -> 4*4 = 16 parallel jobs!
        # inhomogeneous broadening
        "n_freqs": 1,  # increases parallel jobs!!
    }

    bath_config = {
        ### Bath parameters
        "bath_type": "paper",
    }

    return atomic_config, bath_config, pulse_config, simulation_config


def main():
    parser = argparse.ArgumentParser(description="Run 1D spectroscopy.")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--t_coh", type=float, default=0.0, help="Single t_coh value (fs)"
    )
    group.add_argument(
        "--batch_idx", type=int, default=0, help="Batch index for t_coh sweep"
    )

    parser.add_argument("--n_batches", type=int, default=1, help="Number of batches")
    parser.add_argument(
        "--t_det_max", type=float, default=600.0, help="Detection time window (fs)"
    )
    parser.add_argument("--dt", type=float, default=10.0, help="t_coh spacing (fs)")
    parser.add_argument(
        "--t_wait",
        type=float,
        default=0.0,
        help="Waiting time between 2 pump and probe pulse (fs)",
    )
    args = parser.parse_args()

    # =============================
    # Robust argument handling
    # =============================
    if args.t_coh is not None:
        from qspectro2d.spectroscopy.calculations import (
            parallel_compute_1d_E_with_inhomogenity,
            check_the_solver,
        )

        atomic_dict, bath_dict, laser_dict, simulation_dict = create_configs(args)
        ### Create time arrays
        t_max = sim_oqs.simulation_config.t_max

        ### Validate solver (TODO just do it once, not for each t_coh, just for the last one)
        time_cut = -np.inf
        if args.t_coh == 0.0:
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
        rel_dir = run_single_tau(
            args.t_coh, args.t_det_max, args.dt, t_wait=args.t_wait, save_info=True
        )

    elif args.batch_idx is not None:
        t_coh_vals = np.linspace(
            0, args.t_det_max, int((args.t_det_max - 0) / args.dt) + 1
        )
        subarrays = np.array_split(t_coh_vals, args.n_batches)
        tau_subarray = subarrays[args.batch_idx]

        print(
            f"🎯 Running batch {args.batch_idx + 1}/{args.n_batches} with {len(tau_subarray)} t_coh values..."
        )
        for t_coh in tau_subarray:
            save_info = t_coh == 0
            rel_dir = run_single_tau(
                t_coh,
                t_wait=args.t_wait,
                t_det_max=args.t_det_max,
                dt=args.dt,
                save_info=save_info,
            )

        print(f"\n🎯 To stack this data, run:")
        print(f'python stack_t_coh_to_2d.py --rel_path "{rel_dir}"')

    else:
        # Default: run a standard single t_coh simulation
        print(
            "No arguments specified. Running default single t_coh simulation with t_coh=0.0 fs."
        )
        rel_dir = run_single_tau(
            t_coh=0.0,
            t_det_max=args.t_det_max,
            dt=args.dt,
            t_wait=args.t_wait,
            save_info=True,
        )


if __name__ == "__main__":
    main()
