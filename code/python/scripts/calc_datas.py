"""
1D Electronic Spectroscopy Simulation Script – Flexible Execution Mode

This script runs 1D spectroscopy datas for a given set of simulation parameters.
It supports two modes of execution:

# Determine whether to use batch mode or single t_coh mode based on the provided arguments.
    --simulation_type <type>: Type of simulation (default: "1d")
        -> "1d" Run the simulation for one specific coherence time
    -> "2d" Run the simulation for a range of coherence times, split into n_batches

# other arguments:
    --n_batches <total>: Total number of batches (default: 1)
    --batch_idx <index>: Batch index for the current job (0 to n_batches-1, default: 0)

This script is designed for both local development and HPC batch execution.
Results are saved automatically using the qspectro2d I/O framework.

# usage
python calc_datas.py --simulation_type 1d
python calc_datas.py --simulation_type 2d
"""

import time
import argparse
import warnings
import numpy as np
from pathlib import Path

from project_config.paths import SCRIPTS_DIR

from qspectro2d.spectroscopy.one_d_field import parallel_compute_1d_e_comps
from qspectro2d.spectroscopy.inhomogenity import sample_from_gaussian
from qspectro2d.utils import (
    save_data_file,
    save_info_file,
    generate_unique_data_filename,
)
from qspectro2d.config.create_sim_obj import (
    create_base_sim_oqs,
)
from qspectro2d.core.simulation import SimulationModuleOQS

# Suppress noisy but harmless warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
# Silence QuTiP FutureWarning about keyword-only args in brmesolve (qutip >=5.3)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*c_ops, e_ops, args and options will be keyword only from qutip 5\.3.*",
    module=r"qutip\.solver\.brmesolve",
)


# Inhomogeneity batching removed previously; here we generate frequency samples and
# distribute work equally across (t_coh, frequency) pairs using n_batches/batch_idx.
def _generate_freq_samples(sim_oqs: SimulationModuleOQS) -> np.ndarray:
    """Generate inhomogeneous frequency samples for this system.

    Returns array of shape (n_inhomogen, n_atoms) with samples in cm^-1.
    """
    n_samples = int(sim_oqs.simulation_config.n_inhomogen)
    base_freqs = np.asarray(sim_oqs.system.frequencies_cm, dtype=float)
    fwhm_cm = float(sim_oqs.system.delta_inhomogen_cm)
    if n_samples <= 0:
        return np.empty((0, base_freqs.size), dtype=float)
    if np.isclose(fwhm_cm, 0.0):
        return np.tile(base_freqs, (n_samples, 1))
    samples = sample_from_gaussian(n_samples, fwhm_cm, base_freqs)
    # Ensure 2D shape even for degenerate cases
    samples = np.atleast_2d(samples)
    return samples


def _avg_E_over_freqs_for_tcoh(
    sim_oqs: SimulationModuleOQS,
    t_coh_val: float,
    freq_samples: np.ndarray,
    freq_indices: np.ndarray,
    *,
    time_cut: float,
) -> tuple[list[np.ndarray], int]:
    """Compute average E over a subset of frequency samples at a fixed t_coh.

    Returns (avg_E_components, count). If count==0, returns empty list and 0.
    """
    # Update coherence-time-dependent settings once (times and laser delays)
    sim_cfg = sim_oqs.simulation_config
    sim_cfg.t_coh = float(t_coh_val)
    t_wait = sim_cfg.t_wait
    t_det_max = sim_cfg.t_det_max
    sim_cfg.t_max = t_wait + 2 * t_det_max
    if hasattr(sim_oqs, "_times_global"):
        delattr(sim_oqs, "_times_global")
    sim_oqs.reset_times_local()
    sim_oqs.laser.update_delays([sim_cfg.t_coh, t_wait])

    if freq_indices.size == 0:
        return [], 0

    sum_components: list[np.ndarray] | None = None
    contribs = 0
    for idx in freq_indices:
        freqs_cm = freq_samples[int(idx)].tolist()
        sim_oqs.system.update_frequencies_cm(freqs_cm)
        E_list = parallel_compute_1d_e_comps(sim_oqs=sim_oqs, time_cut=time_cut)
        if sum_components is None:
            sum_components = [E.copy() for E in E_list]
        else:
            for k in range(len(sum_components)):
                sum_components[k] += E_list[k]
        contribs += 1

    assert sum_components is not None
    avg_components = [S / float(contribs) for S in sum_components]
    return avg_components, contribs


def run_1d_mode(args):
    """Run single 1D simulation for a specific coherence time."""
    config_path = SCRIPTS_DIR / "config.yaml"

    # Build base simulation (applies CLI overrides inside)
    sim_oqs, time_cut = create_base_sim_oqs(config_path=config_path)

    t_coh_print = sim_oqs.simulation_config.t_coh
    print(f"🎯 Running 1D mode with t_coh = {t_coh_print:.2f} fs (from config)")

    # Generate frequency samples and choose exactly one frequency index for this batch
    freq_samples = _generate_freq_samples(sim_oqs)
    n_inhom = freq_samples.shape[0]
    n_batches = int(args.n_batches)
    batch_idx = int(args.batch_idx)
    if n_inhom <= 0:
        raise ValueError("No inhomogeneous samples available (n_inhomogen == 0)")
    if batch_idx < 0 or batch_idx >= max(1, n_batches):
        raise ValueError(f"batch_idx {batch_idx} out of range for n_batches {n_batches}")
    # Map batch -> a single freq index deterministically
    if n_batches != n_inhom:
        print(
            f"ℹ️  n_batches ({n_batches}) != n_inhomogen ({n_inhom}); using modulo mapping batch_idx % n_inhomogen."
        )
    freq_idx_for_batch = batch_idx % n_inhom
    freq_idx_subset = np.array([freq_idx_for_batch], dtype=int)
    avg_E, contribs = _avg_E_over_freqs_for_tcoh(
        sim_oqs,
        t_coh_print,
        freq_samples,
        freq_idx_subset,
        time_cut=time_cut,
    )

    # Save one averaged dataset for this t_coh
    sim_config_obj = sim_oqs.simulation_config
    sim_config_obj.t_coh = t_coh_print
    abs_path = generate_unique_data_filename(sim_oqs.system, sim_config_obj)
    abs_data_path = Path(f"{abs_path}_data.npz")
    metadata = {
        "n_batches": int(max(1, n_batches)),
        "batch_idx": int(batch_idx),
        "n_inhomogen_total": int(n_inhom),
        "n_inhomogen_in_batch": 1,
        "freq_idx": int(freq_idx_for_batch),
        "t_idx": 0,
        "t_coh_value": float(t_coh_print),
    }
    save_data_file(
        abs_data_path,
        avg_E,
        sim_oqs.times_det,
        signal_types=sim_config_obj.signal_types,
        metadata=metadata,
    )

    abs_info_path = Path(f"{abs_path}_info.pkl")
    save_info_file(
        abs_info_path,
        sim_oqs.system,
        bath=sim_oqs.bath,
        laser=sim_oqs.laser,
        sim_config=sim_config_obj,
    )

    print(
        f"Saved 1D result for t_coh={t_coh_print:.2f} fs with {contribs}/{n_inhom} inhom samples."
    )

    # Next-step hints: stacking (optional) and plotting
    print(f"\n🎯 To stack this datas into 2D (skips automatically if already stacked), run:")
    print(f'python stack_1dto2d.py --abs_path "{abs_data_path}" --skip_if_exists')
    print(f"\n🎯 To plot this datas, run:")
    print(f'python plot_datas.py --abs_path "{abs_data_path}"')


def run_2d_mode(args):
    """Run 2D mode with batch processing for multiple coherence times."""
    config_path = SCRIPTS_DIR / "config.yaml"
    n_batches = args.n_batches
    batch_idx = args.batch_idx

    # Build base simulation (applies CLI overrides inside)
    sim_oqs, time_cut = create_base_sim_oqs(config_path=config_path)

    print(f"🎯 Running 2D mode - batch {batch_idx + 1}/{n_batches}")

    # Generate t_coh values (reuse detection times array) and frequency samples
    t_coh_vals = sim_oqs.times_det
    freq_samples = _generate_freq_samples(sim_oqs)
    n_t = len(t_coh_vals)
    n_inhom = freq_samples.shape[0]

    # Choose exactly one frequency index for this batch
    if n_inhom <= 0:
        raise ValueError("No inhomogeneous samples available (n_inhomogen == 0)")
    if n_batches != n_inhom:
        print(
            f"ℹ️  n_batches ({n_batches}) != n_inhomogen ({n_inhom}); using modulo mapping batch_idx % n_inhomogen."
        )
    freq_idx_for_batch = batch_idx % n_inhom
    print(
        f"📊 Processing all t_coh values with single inhom sample freq_idx={freq_idx_for_batch} (batch {batch_idx+1}/{n_batches})"
    )

    abs_data_path = None
    start_time = time.time()
    for it in range(n_t):
        print(f"\n--- t_idx={it} / {n_t-1} ---")
        t_coh = float(t_coh_vals[it])
        freq_idx_subset = np.asarray([freq_idx_for_batch], dtype=int)
        avg_E, contribs = _avg_E_over_freqs_for_tcoh(
            sim_oqs,
            t_coh,
            freq_samples,
            freq_idx_subset,
            time_cut=time_cut,
        )

        # Save averaged dataset for this t_coh in this batch
        sim_oqs.simulation_config.t_coh = t_coh
        abs_path = generate_unique_data_filename(sim_oqs.system, sim_oqs.simulation_config)
        abs_data_path = Path(f"{abs_path}_data.npz")
        metadata = {
            "n_batches": int(max(1, n_batches)),
            "batch_idx": int(batch_idx),
            "n_inhomogen_total": int(n_inhom),
            "n_inhomogen_in_batch": 1,
            "freq_idx": int(freq_idx_for_batch),
            "t_idx": int(it),
            "t_coh_value": float(t_coh),
        }
        save_data_file(
            abs_data_path,
            avg_E,
            sim_oqs.times_det,
            signal_types=sim_oqs.simulation_config.signal_types,
            metadata=metadata,
        )
        print(
            f"Saved 2D partial for t_coh={t_coh:.2f} fs with {contribs}/{n_inhom} inhom samples in this batch."
        )
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    print(f"\n✅ Batch {batch_idx + 1}/{n_batches} completed!")
    if abs_data_path is not None:
        print(f"\n🎯 To stack this datas into 2D (skips automatically if already stacked), run:")
        print(f'python stack_1dto2d.py --abs_path "{abs_data_path}" --skip_if_exists')
        print(f"\n🎯 To plot this datas, run:")
        print(f'python plot_datas.py --abs_path "{abs_data_path}"')
    else:
        print("ℹ️ No files were saved in this batch; nothing to stack or plot.")


def main():
    """
    Main function with argument parsing and execution logic.
    """
    parser = argparse.ArgumentParser(
        description="1D Electronic Spectroscopy Simulation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""cd 
Examples:
  # Run single 1D simulation
  python calc_datas.py --simulation_type 1d

  # Run 2D batch mode
    python calc_datas.py --simulation_type 2d --batch_idx 0 --n_batches 10
        """,
    )

    parser.add_argument(
        "--simulation_type",
        type=str,
        default="1d",
        choices=["1d", "2d"],
        help="Type of simulation: '1d' for single t_coh, '2d' for batch processing (default: 1d)",
    )

    parser.add_argument(
        "--batch_idx",
        type=int,
        default=0,
        help="Batch index for the current job (0 to n_batches-1)",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=1,
        help="Total number of batches >= 1 (only for 2D mode)",
    )
    # Inhomogeneity batching removed
    args = parser.parse_args()

    print("=" * 80)
    print("1D ELECTRONIC SPECTROSCOPY SIMULATION")
    print(f"Simulation type: {args.simulation_type}")

    if args.simulation_type == "1d":
        run_1d_mode(args)
    elif args.simulation_type == "2d":
        run_2d_mode(args)

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETED")


if __name__ == "__main__":
    main()
