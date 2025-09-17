# TODO
"""
Generate and (optionally) submit SLURM jobs for 2D spectroscopy batches.
Only the batch window arguments differ per job (batch_idx and n_batches).
All other parameters are resolved inside calc_datas.py via the config in SCRIPTS_DIR.

Usage examples (on the cluster after cloning the repo):
  python gen_run_2d_batches.py --n_batches 16
  python gen_run_2d_batches.py --n_batches 32 --no_submit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from subprocess import run, CalledProcessError

from project_config.paths import SCRIPTS_DIR


def create_batch_script(
    batch_idx: int,
    total_batches: int,
    job_dir: Path,
) -> Path:
    """Create one SLURM script that runs a single batch via calc_datas.py.

    The script changes only the time-window related arguments per batch
    (batch_idx and n_batches).
    """
    # Determine mail type based on batch index
    mail_type = "NONE"
    if batch_idx == 0:
        mail_type = "FAIL"
    if batch_idx == total_batches - 1:
        if mail_type == "FAIL":
            mail_type = "END,FAIL"
        else:
            mail_type = "END"

    # Base header shared by all batches
    header = f"""#!/bin/bash
#SBATCH --job-name=qs2d_b{batch_idx}
#SBATCH --chdir={job_dir}
#SBATCH --output=logs/batch_{batch_idx}.out
#SBATCH --error=logs/batch_{batch_idx}.err
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

    # Body differs for last batch: capture output, extract directory, suggest plotting command
    if batch_idx == total_batches - 1:
        body = f"""
# Execute calc_datas.py from scripts/ (two levels up from job_dir) and capture output
OUTPUT=$(python ../../calc_datas.py --simulation_type 2d --batch_idx {batch_idx} --n_batches {total_batches} 2>&1 | tee -a logs/batch_{batch_idx}.out)
STATUS=${{PIPESTATUS[0]}}

if [ "$STATUS" -eq 0 ]; then
    # Try to extract the 1D data directory path printed by calc_datas.py
    DIR_PATH=$(printf "%s" "$OUTPUT" | awk -F'--abs_path "|"' '/python stack_1dto2d\.py/ {{print $2}}' | tail -n 1)

    if [ -n "$DIR_PATH" ]; then
        # Pick the most recent *_data.npz file from that directory
        LAST_FILE=$(ls -t "$DIR_PATH"/*_data.npz 2>/dev/null | head -n 1)
        if [ -n "$LAST_FILE" ]; then
            echo ""
            echo " to now stack the data and create the plots [in SCRIPTS_DIR ]call:"
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
# Execute calc_datas.py from scripts/ (two levels up from job_dir)
python ../../calc_datas.py --simulation_type 2d --batch_idx {batch_idx} --n_batches {total_batches}
"""

    content = header + "\n" + body

    path = Path(job_dir) / f"batch_{batch_idx}.slurm"
    # Write with Unix line endings so SLURM doesn't complain on Linux clusters
    try:
        path.write_text(content, newline="\n")
    except TypeError:
        path.write_text(content.replace("\r\n", "\n").replace("\r", "\n"))
    path.chmod(0o755)
    return path


def execute_slurm_scripts(job_dir: Path) -> None:
    """Submit all generated SLURM scripts in the job directory."""
    for slurm_script in sorted(Path(job_dir).glob("batch_*.slurm")):
        try:
            run(["sbatch", str(slurm_script)], check=True)
        except (FileNotFoundError, CalledProcessError) as exc:
            print(f"Failed submitting {slurm_script.name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally submit SLURM jobs for 2D spectroscopy batches.\n"
            "Only batch_idx/n_batches are varied per job; other parameters come from config in SCRIPTS_DIR."
        )
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=10,
        help="Total number of batches (default 10).",
    )
    parser.add_argument(
        "--no_submit",
        action="store_true",
        help="Only generate job scripts, do not submit.",
    )
    args = parser.parse_args()

    n_batches = args.n_batches
    if n_batches <= 0:
        raise ValueError("n_batches must be a positive integer")

    # Create a unique job directory under scripts/batch_jobs
    base_name = f"2d_{n_batches}b"
    job_root = SCRIPTS_DIR / "batch_jobs"
    job_dir = job_root / base_name
    suffix = 0
    while job_dir.exists():
        suffix += 1
        job_dir = job_root / f"{base_name}_{suffix}"

    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)

    # Write scripts
    for bidx in range(n_batches):
        create_batch_script(
            batch_idx=bidx,
            total_batches=n_batches,
            job_dir=job_dir,
        )

    print(f"Generated {n_batches} scripts in: {job_dir}")

    # Optionally submit
    if not args.no_submit:
        execute_slurm_scripts(job_dir)
    else:
        print("Submission skipped (use without --no_submit to sbatch).")


if __name__ == "__main__":
    main()
