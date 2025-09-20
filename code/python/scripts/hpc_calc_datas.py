"""Minimal SLURM job generator and submitter for batched 2D runs.

This script creates one ``.slurm`` script per batch index (0..n_batches-1)
and submits them with ``sbatch``. The only CLI argument is ``--n_batches``.

Each job runs:
    python calc_datas.py --simulation_type 2d --n_batches {n_batches} --batch_idx {batch_idx}

Notes:
- Mail notifications are included ONLY for the first and last batch indices.
- The working directory for the job is set to the ``scripts`` directory (this file's parent),
  so that ``python calc_datas.py`` can resolve correctly.
- Slurm scripts are written to ``scripts/batch_jobs/``.
- Log files are written to ``scripts/logs/<job_name>.out`` and ``scripts/logs/<job_name>.err``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _slurm_script_text(*, job_name: str, n_batches: int, batch_idx: int) -> str:
    """Render the SLURM script text for a single batch index.

    Mail directives are included only for first and last batches.
    """
    # Use a Linux/HPC-friendly working directory; avoids leaking Windows paths into jobs
    work_dir_str = "$HOME/Master_thesis/code/python/scripts"
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
#SBATCH --chdir={work_dir_str}
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --time=0-02:00:00
{mail_lines}

# Load conda (adjust to your cluster if needed)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/$USER/miniconda3/etc/profile.d/conda.sh"
fi
conda activate master_env || true

python -u calc_datas.py --simulation_type 2d --n_batches {n_batches} --batch_idx {batch_idx}
"""

def _ensure_dirs(*, work_dir: Path, slurm_dir: Path) -> None:
    """Create required directories: the slurm scripts dir and the logs dir under work_dir."""
    slurm_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)


def _submit_job(script_path: Path) -> str:
    """Submit a job with sbatch and return the scheduler response.

    If ``sbatch`` is not available, a helpful error is raised.
    """
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise RuntimeError("sbatch not found on PATH. Run this on your cluster login node.")

    result = subprocess.run([sbatch, str(script_path)], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM jobs for batched 2D spectroscopy runs.",
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
    args = parser.parse_args()

    n_batches = int(args.n_batches)
    if n_batches <= 0:
        raise ValueError("--n_batches must be a positive integer")

    work_dir = Path(__file__).resolve().parent  # local scripts directory containing calc_datas.py
    slurm_dir = work_dir / "batch_jobs"
    _ensure_dirs(work_dir=work_dir, slurm_dir=slurm_dir)

    action_verb = "Generating" if args.generate_only else "Creating and submitting"
    print(
    f"{action_verb} {n_batches} SLURM jobs; scripts in {slurm_dir}, chdir -> $HOME/Master_thesis/code/python/scripts ..."
    )

    for batch_idx in range(n_batches):
        job_name = f"2d_b{batch_idx:03d}_of_{n_batches:03d}"
        script_name = f"slurm_{job_name}.slurm"
        script_path = slurm_dir / script_name

        content = _slurm_script_text(
            job_name=job_name,
            n_batches=n_batches,
            batch_idx=batch_idx,
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
