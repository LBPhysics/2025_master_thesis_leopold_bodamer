"""
Generate and (optionally) submit SLURM jobs for spectroscopy runs.

Supports both 1D (single job) and 2D (batched jobs) simulations.
All simulation parameters (except batch window for 2D) are resolved in
`calc_datas.py` via the config in `SCRIPTS_DIR`.

Usage examples (on the cluster after cloning the repo):
  # 1D (single job)
  python hpc_calc_datas.py --simulation_type 1d

  # 2D (batched)
  python hpc_calc_datas.py --simulation_type 2d --coh_batches 16
  python hpc_calc_datas.py --simulation_type 2d --coh_batches 32 --no_submit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from subprocess import run, CalledProcessError

from project_config.paths import SCRIPTS_DIR


def _slurm_header(job_name: str, job_dir: Path, mail_type: str) -> str:
    """Return a standard SLURM header used by all jobs."""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --chdir={job_dir}
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=1G
#SBATCH --time=0-02:00:00
#SBATCH --mail-type={mail_type}
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de # TODO CHANGE THE MAIL HERE

# Load conda (adjust to your cluster if needed)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/$USER/miniconda3/etc/profile.d/conda.sh"
fi
conda activate master_env || true
"""


def create_1d_script(
    job_dir: Path, *, inhom_idx: int | None = None, inhom_batches: int | None = None
) -> Path:
    """Create SLURM script that runs a 1D job via calc_datas.py.

    If inhom batching is provided, pass through the batch indices and name the job accordingly.
    """
    job_suffix = ""
    cli_suffix = ""
    if inhom_idx is not None and inhom_batches is not None:
        job_suffix = f"_inh{inhom_idx}of{inhom_batches}"
        cli_suffix = f" --inhom_idx {inhom_idx} --inhom_batches {inhom_batches} --sum_inhom"
    header = _slurm_header(job_name=f"qs1d{job_suffix}", job_dir=job_dir, mail_type="END,FAIL")

    # Capture output and extract --abs_path from the hint printed by calc_datas.py
    body = rf"""
OUTPUT=$(python ../../calc_datas.py --simulation_type 1d{cli_suffix} 2>&1 | tee -a logs/qs1d{job_suffix}.out)
STATUS=${{PIPESTATUS[0]}}

if [ "$STATUS" -eq 0 ]; then
    DIR_PATH=$(printf "%s" "$OUTPUT" | awk -F'--abs_path "|"' '/python plot_datas\.py/ {{print $2}}' | tail -n 1)
    if [ -n "$DIR_PATH" ]; then
        LAST_FILE=$(ls -t "$DIR_PATH"/*_data.npz 2>/dev/null | head -n 1)
        if [ -n "$LAST_FILE" ]; then
            echo ""
            echo " to now plot the 1D data [in SCRIPTS_DIR] call:"
            echo "python hpc_plot_datas.py --abs_path \"$LAST_FILE\""
            echo ""
        else
            echo "WARN: No *_data.npz file found in $DIR_PATH"
        fi
    else
        echo "WARN: Could not detect 1D data path from calc_datas.py output."
    fi
else
    echo "calc_datas.py failed with exit code $STATUS"
fi
"""

    content = header + "\n" + body
    fname = f"single_1d{job_suffix}.slurm"
    path = Path(job_dir) / fname
    try:
        path.write_text(content, newline="\n")
    except TypeError:
        path.write_text(content.replace("\r\n", "\n").replace("\r", "\n"))
    path.chmod(0o755)
    return path


def create_2d_batch_script(
    coh_idx: int,
    total_batches: int,
    job_dir: Path,
    *,
    inhom_idx: int | None = None,
    inhom_batches: int | None = None,
) -> Path:
    """Create one SLURM script that runs a single 2D batch via calc_datas.py."""
    # Determine mail type based on batch index
    mail_type = "NONE"
    if coh_idx == 0:
        mail_type = "FAIL"
    if coh_idx == total_batches - 1:
        mail_type = "END,FAIL" if mail_type == "FAIL" else "END"

    job_suffix = ""
    cli_suffix = ""
    if inhom_idx is not None and inhom_batches is not None:
        job_suffix = f"_inh{inhom_idx}of{inhom_batches}"
        cli_suffix = f" --inhom_idx {inhom_idx} --inhom_batches {inhom_batches} --sum_inhom"
    header = _slurm_header(
        job_name=f"qs2d_b{coh_idx}{job_suffix}", job_dir=job_dir, mail_type=mail_type
    )

    # Body differs for last batch: capture output, extract directory, suggest plotting command
    if coh_idx == total_batches - 1:
        body = f"""
OUTPUT=$(python ../../calc_datas.py --simulation_type 2d --coh_idx {coh_idx} --coh_batches {total_batches}{cli_suffix} 2>&1 | tee -a logs/qs2d_b{coh_idx}{job_suffix}.out)
STATUS=${{PIPESTATUS[0]}}

if [ "$STATUS" -eq 0 ]; then
    DIR_PATH=$(printf "%s" "$OUTPUT" | awk -F'--abs_path "|"' '/python stack_1dto2d\.py/ {{print $2}}' | tail -n 1)
    if [ -n "$DIR_PATH" ]; then
        LAST_FILE=$(ls -t "$DIR_PATH"/*_data.npz 2>/dev/null | head -n 1)
        if [ -n "$LAST_FILE" ]; then
            echo ""
            echo " to now stack the data and create the plots [in SCRIPTS_DIR] call:"
            echo "python hpc_plot_datas.py --abs_path \"$LAST_FILE\""
            echo ""
        else
            echo "WARN: No *_data.npz file found in $DIR_PATH"
        fi
    else
        echo "WARN: Could not detect 1D data directory from calc_datas.py output."
    fi
else
    echo "calc_datas.py failed with exit code $STATUS"
fi
"""
    else:
        body = f"""
python ../../calc_datas.py --simulation_type 2d --coh_idx {coh_idx} --coh_batches {total_batches}{cli_suffix}
"""

    content = header + "\n" + body
    path = Path(job_dir) / f"batch_{coh_idx}{job_suffix}.slurm"
    try:
        path.write_text(content, newline="\n")
    except TypeError:
        path.write_text(content.replace("\r\n", "\n").replace("\r", "\n"))
    path.chmod(0o755)
    return path


def execute_slurm_scripts(job_dir: Path) -> None:
    """Submit all generated SLURM scripts in the job directory."""
    for slurm_script in sorted(Path(job_dir).glob("*.slurm")):
        try:
            run(["sbatch", str(slurm_script)], check=True)
        except (FileNotFoundError, CalledProcessError) as exc:
            print(f"Failed submitting {slurm_script.name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally submit SLURM jobs for spectroscopy runs (1D/2D).\n"
            "For 2D, coh_idx/coh_batches are varied per job; other parameters come from config in SCRIPTS_DIR."
        )
    )
    parser.add_argument(
        "--simulation_type",
        type=str,
        default="2d",
        choices=["1d", "2d"],
        help="Simulation type to run on HPC: '1d' (single job) or '2d' (batched)",
    )
    parser.add_argument(
        "--coh_batches",
        type=int,
        default=10,
        help="Total number of batches (2D only, default 10).",
    )
    parser.add_argument(
        "--inhom_batches",
        type=int,
        default=None,
        help="Split inhomogeneous sampling into this many batches; if omitted, process all samples in one run.",
    )
    parser.add_argument(
        "--inhom_idx",
        type=int,
        default=None,
        help="Index of the inhomogeneity batch for this job (0..inhom_batches-1).",
    )
    parser.add_argument(
        "--no_submit",
        action="store_true",
        help="Only generate job scripts, do not submit.",
    )
    args = parser.parse_args()

    job_root = SCRIPTS_DIR / "batch_jobs"

    if args.simulation_type == "1d":
        # Unique job directory under scripts/batch_jobs/1d
        base_name = "1d"
        job_dir = job_root / base_name
        suffix = 0
        while job_dir.exists():
            suffix += 1
            job_dir = job_root / f"{base_name}_{suffix}"

        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=False)

        if args.inhom_batches is None:
            create_1d_script(job_dir)
            print(f"Generated 1D script in: {job_dir}")
        else:
            if args.inhom_batches <= 0:
                raise ValueError("--inhom_batches must be positive for 1D mode if provided")
            for inh_idx in range(args.inhom_batches):
                create_1d_script(job_dir, inhom_idx=inh_idx, inhom_batches=args.inhom_batches)
            print(f"Generated 1D scripts (inhom batches={args.inhom_batches}) in: {job_dir}")

        if not args.no_submit:
            execute_slurm_scripts(job_dir)
        else:
            print("Submission skipped (use without --no_submit to sbatch).")

    else:  # "2d"
        coh_batches = args.coh_batches
        if coh_batches <= 0:
            raise ValueError("coh_batches must be a positive integer for 2D mode")

        # Unique job directory under scripts/batch_jobs
        base_name = f"2d_{coh_batches}b"
        job_dir = job_root / base_name
        suffix = 0
        while job_dir.exists():
            suffix += 1
            job_dir = job_root / f"{base_name}_{suffix}"

        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=False)

        if args.inhom_batches is None:
            for bidx in range(coh_batches):
                create_2d_batch_script(coh_idx=bidx, total_batches=coh_batches, job_dir=job_dir)
        else:
            if args.inhom_batches <= 0:
                raise ValueError("--inhom_batches must be positive for 2D mode if provided")
            for bidx in range(coh_batches):
                for inh_idx in range(args.inhom_batches):
                    create_2d_batch_script(
                        coh_idx=bidx,
                        total_batches=coh_batches,
                        job_dir=job_dir,
                        inhom_idx=inh_idx,
                        inhom_batches=args.inhom_batches,
                    )

        print(f"Generated {coh_batches} scripts in: {job_dir}")

        if not args.no_submit:
            execute_slurm_scripts(job_dir)
        else:
            print("Submission skipped (use without --no_submit to sbatch).")


if __name__ == "__main__":
    main()
