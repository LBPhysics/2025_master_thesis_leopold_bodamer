"""
Generate and submit a single SLURM job for spectroscopy runs (no batching, no inhomogeneity).

This script creates one SLURM script that calls `calc_datas.py` with
`--simulation_type` (either "1d" or "2d"), then submits it via `sbatch`.

Usage (on the cluster login node):
    # 2D: iterate over all t_det as t_coh, saving one file per point
    python hpc_calc_datas.py --simulation_type 2d

    # 1D: compute one trace at the configured t_coh
    python hpc_calc_datas.py --simulation_type 1d

Add --no_submit to generate the script without submitting it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from subprocess import CalledProcessError, run


def _slurm_header(
    job_name: str,
    job_dir: Path,
    mail_type: str = "END,FAIL",
    cpus: int = 16,
    mem: str = "2G",
    time_limit: str = "0-02:00:00",
) -> str:
    """Return a simple SLURM header for a single-job script.

    The email is left as a TODO so users are forced to personalize it.
    """
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --chdir={job_dir}
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --mail-type={mail_type}
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de  # TODO set your email

# Load conda (adjust to your cluster if needed)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/$USER/miniconda3/etc/profile.d/conda.sh"
fi
conda activate master_env || true
"""


def _create_job_script(
    *,
    scripts_dir: Path,
    job_dir: Path,
    sim_type: str,
    cpus: int,
    mem: str,
    time_limit: str,
) -> Path:
    """Create a single SLURM script that runs calc_datas.py once for sim_type."""
    calc_path = (scripts_dir / "calc_datas.py").resolve()
    if not calc_path.exists():
        raise FileNotFoundError(f"calc_datas.py not found at {calc_path}")

    job_name = f"qs_{sim_type}"
    mail_type = "END,FAIL"
    header = _slurm_header(
        job_name=job_name,
        job_dir=job_dir,
        mail_type=mail_type,
        cpus=cpus,
        mem=mem,
        time_limit=time_limit,
    )

    body = f"python {calc_path} --simulation_type {sim_type}\n"

    content = header + "\n" + body
    script_path = job_dir / f"run_{sim_type}.slurm"
    # Ensure LF newlines for SLURM
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    script_path.write_text(text, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def _submit_all(job_dir: Path) -> None:
    """Submit all .slurm scripts in job_dir via sbatch."""
    sbatch_path = shutil.which("sbatch")
    if not sbatch_path:
        print("WARN: 'sbatch' not found in PATH; skipping automatic submission.")
        return
    for slurm_script in sorted(job_dir.glob("*.slurm")):
        try:
            run([sbatch_path, str(slurm_script)], check=True)
        except (FileNotFoundError, CalledProcessError) as exc:
            print(
                f"WARN: Failed submitting {slurm_script.name}: {exc}\n"
                f"Suggestion: inspect script or submit manually with 'sbatch {slurm_script.name}'"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Generate one SLURM script that calls calc_datas.py and submit it.")
    )
    parser.add_argument(
        "--simulation_type",
        type=str,
        default="2d",
        choices=["1d", "2d"],
        help="Simulation type to run in each batch (default: 2d)",
    )
    parser.add_argument("--cpus", type=int, default=16, help="CPUs per task (default: 16)")
    parser.add_argument("--mem", type=str, default="2G", help="Memory (default: 2G)")
    parser.add_argument(
        "--time_limit", type=str, default="0-02:00:00", help="Walltime e.g. 0-02:00:00"
    )
    parser.add_argument(
        "--no_submit", action="store_true", help="Only generate scripts, do not submit"
    )
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    job_root = scripts_dir / "batch_jobs"

    base_name = f"{args.simulation_type}"
    job_dir = job_root / base_name
    suffix = 0
    while job_dir.exists():
        suffix += 1
        job_dir = job_root / f"{base_name}_{suffix}"

    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)

    # Create a single script
    slurm_script = _create_job_script(
        scripts_dir=scripts_dir,
        job_dir=job_dir,
        sim_type=args.simulation_type,
        cpus=args.cpus,
        mem=args.mem,
        time_limit=args.time_limit,
    )

    print(f"Generated script: {slurm_script}")

    if not args.no_submit:
        _submit_all(job_dir)
    else:
        print("Submission skipped (use without --no_submit to sbatch).")


if __name__ == "__main__":
    main()
