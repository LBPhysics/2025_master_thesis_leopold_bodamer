"""Minimal SLURM job generator and submitter for batched 1D/2D runs.

This script creates one SLURM script per batch index (0..n_batches-1)
and, by default, submits them with ``sbatch``.

Layout:
- Scripts are generated under ``scripts_dir/batch_jobs/{n_batches}batches``
    where ``scripts_dir`` is the directory containing this file.
- Logs are written to ``scripts_dir/batch_jobs/{n_batches}batches[/_i]/logs/``
    via Slurm's ``--output``/``--error`` options.

Each job runs (from the batching directory):
    python ../../calc_datas.py --simulation_type {sim_type} --n_batches {n_batches} --batch_idx {batch_idx}

Notes:
- Mail notifications are included ONLY for the first and last batch indices.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _slurm_script_text(
    *, job_name: str, n_batches: int, batch_idx: int, simulation_type: str
) -> str:
    """Render the SLURM script text for a single batch index.

    Mail directives are included only for first and last batches.
    """
    mail_lines = (
        (
            '#SBATCH --mail-type="END,FAIL"\n'
            "#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de\n"
        )
        if (batch_idx == 0 or batch_idx == n_batches - 1)
        else ""
    )

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --time=0-02:00:00
{mail_lines}

python ../../calc_datas.py --simulation_type {simulation_type} --n_batches {n_batches} --batch_idx {batch_idx}
"""


def _ensure_dirs(job_dir: Path) -> None:
    """Create the batching directory and its logs subdirectory if missing."""
    job_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)


def _submit_job(script_path: Path) -> str:
    """Submit a job with sbatch and return the scheduler response.

    If ``sbatch`` is not available, a helpful error is raised.
    """
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise RuntimeError("sbatch not found on PATH. Run this on your cluster login node.")

    # Submit from within the batching directory so relative log paths work.
    result = subprocess.run(
        [sbatch, script_path.name],
        cwd=str(script_path.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM jobs for batched 1D/2D spectroscopy runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        required=True,
        help="Total number of batches (creates one job per batch_idx 0..n_batches-1)",
    )
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Only generate the .slurm scripts without submitting via sbatch.",
    )
    parser.add_argument(
        "--simulation_type",
        type=str,
        default="2d",
        choices=["1d", "2d"],
        help="Execution mode (default: 2d)",
    )
    args = parser.parse_args()

    n_batches = int(args.n_batches)
    if n_batches <= 0:
        raise ValueError("--n_batches must be a positive integer")

    scripts_dir = Path(__file__).resolve().parent  # directory containing this script

    # Create a unique job directory under scripts_dir/batch_jobs
    job_root = scripts_dir / "batch_jobs"
    sim_type = str(args.simulation_type).lower()
    base_name = f"{sim_type}_{n_batches}batches"
    job_dir = job_root / base_name
    suffix = 0
    while job_dir.exists():
        suffix += 1
        job_dir = job_root / f"{base_name}_{suffix}"

    _ensure_dirs(job_dir)

    action_verb = "Generating" if args.generate_only else "Creating and submitting"
    print(f"{action_verb} {n_batches} SLURM jobs in {job_dir} ...")

    for batch_idx in range(n_batches):
        job_name = f"{sim_type}_b{batch_idx:03d}_of_{n_batches:03d}"
        script_name = f"{job_name}.slurm"
        script_path = job_dir / script_name

        content = _slurm_script_text(
            job_name=job_name,
            n_batches=n_batches,
            batch_idx=batch_idx,
            simulation_type=sim_type,
        )
        script_path.write_text(content, encoding="utf-8")

        if args.generate_only:
            print(f"  generated {script_name}")
        else:
            try:
                submit_msg = _submit_job(script_path)
            except Exception as exc:  # Fail fast with a clear message
                raise RuntimeError(f"Failed to submit {script_name}: {exc}") from exc

            print(f"  submitted {script_name}: {submit_msg}")

    print("All jobs generated." if args.generate_only else "All jobs submitted.")


if __name__ == "__main__":  # pragma: no cover
    main()
