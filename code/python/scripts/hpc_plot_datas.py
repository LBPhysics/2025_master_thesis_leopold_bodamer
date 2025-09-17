"""
Generate and (optionally) submit a SLURM job to plot data.

This script calls `stack_1dto2d.py --skip_if_exists` first, which will stack 1D → 2D
if needed or skip when an existing 2D dataset is present, then we create/submit a
plotting job using the path printed by the stacking script.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from subprocess import run, CalledProcessError

from project_config.paths import SCRIPTS_DIR


def run_stacking_script(abs_path: str, skip_if_exists: bool = True) -> str:
    """Call stack_1dto2d.py and return the plotting abs_path printed by it.

    Always delegates existence checks to stack_1dto2d.py using --skip_if_exists when requested.
    """
    data_dir = Path(abs_path)
    cmd = ["python", "stack_1dto2d.py", "--abs_path", str(data_dir)]
    if skip_if_exists:
        cmd.append("--skip_if_exists")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPTS_DIR)

    if result.returncode != 0:
        raise RuntimeError(f"Stacking script failed: {result.stderr}")

    # Parse the output to find the plotting command
    output = result.stdout
    match = re.search(r'python plot_datas\.py --abs_path "([^"]+)"', output)
    if not match:
        raise RuntimeError("Could not find the plotting abs_path in stacking output")

    return match.group(1)


def create_plotting_script(
    abs_path: str,
    job_dir: Path,
) -> Path:
    """Create a SLURM script that runs plot_datas.py with the given abs_path."""
    content = f"""#!/bin/bash
#SBATCH --job-name=plot_data
#SBATCH --chdir={job_dir}
#SBATCH --output=logs/plotting.out
#SBATCH --error=logs/plotting.err
#SBATCH --partition=GPGPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de  # TODO CHANGE THE MAIL HERE

# Load conda (adjust to your cluster if needed)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/$USER/miniconda3/etc/profile.d/conda.sh"
fi
conda activate master_env || true

# Execute plot_datas.py from scripts/ (two levels up from job_dir)
python ../../plot_datas.py --abs_path "{abs_path}"
"""
    path = job_dir / "plotting.slurm"
    # Write with Unix line endings so SLURM doesn't complain on Linux clusters
    try:
        path.write_text(content, newline="\n")
    except TypeError:
        path.write_text(content.replace("\r\n", "\n").replace("\r", "\n"))
    path.chmod(0o755)
    return path


def execute_slurm_script(job_dir: Path) -> None:
    """Submit the generated SLURM script."""
    slurm_script = job_dir / "plotting.slurm"
    try:
        run(["sbatch", str(slurm_script)], check=True)
        print(f"Submitted {slurm_script}")
    except (FileNotFoundError, CalledProcessError) as exc:
        print(f"Failed submitting {slurm_script}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and optionally submit a SLURM job for stacking 1D data to 2D and plotting. Skips stacking if 2D already exists."
    )
    parser.add_argument(
        "--abs_path",
        type=str,
        required=True,
        help="Absolute path to one of the 1D data files (e.g., t_coh_50.0_data.npz).",
    )
    parser.add_argument(
        "--no_submit",
        action="store_true",
        help="Only generate the job script, do not submit.",
    )
    args = parser.parse_args()

    print("🔄 Preparing 2D dataset (stacking if needed)...")
    try:
        plotting_abs_path = run_stacking_script(args.abs_path)
        print(f"✅ Dataset ready. Plot path base: {plotting_abs_path}")
    except RuntimeError as e:
        print(f"❌ Stacking failed: {e}")
        return

    # Create a unique job directory under scripts/batch_jobs
    base_name = "plotting"
    job_root = SCRIPTS_DIR / "batch_jobs"
    job_dir = job_root / base_name
    suffix = 0
    while job_dir.exists():
        suffix += 1
        job_dir = job_root / f"{base_name}_{suffix}"

    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)

    # Create the script
    create_plotting_script(
        abs_path=plotting_abs_path,
        job_dir=job_dir,
    )

    print(f"Generated plotting script in: {job_dir}")

    # Optionally submit
    if not args.no_submit:
        execute_slurm_script(job_dir)
    else:
        print("Submission skipped (use without --no_submit to sbatch).")


if __name__ == "__main__":
    main()
