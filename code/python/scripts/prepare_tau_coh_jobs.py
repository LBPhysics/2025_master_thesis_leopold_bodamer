from pathlib import Path

TOTAL_BATCHES = 10  # You can increase/decrease this
T_DET_MAX = 600.0  # Maximum detection time in fs
DT = 0.1  # Spacing between tau_coh, and of also t_det values in fs

def create_batch_script(batch_idx, total_batches, job_dir, t_det_max=600, dt=10):
    """Create a batch script for SLURM to run a specific batch of calculations."""

    content = f"""#!/bin/bash
#SBATCH --job-name=batch_{batch_idx}
#SBATCH --output=logs/batch_{batch_idx}.out
#SBATCH --error=logs/batch_{batch_idx}.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de

#SBATCH --cpus-per-task=16
#SBATCH --mem=10G # this should be more than enough
#SBATCH --time=0-5 # 5 hours

source /home/lbodamer/miniconda3/etc/profile.d/conda.sh
conda activate master_env

cd /home/lbodamer/Master_thesis/code/python/scripts/

python3 calc_1D_datas.py --batch_idx {batch_idx} --n_batches {total_batches} --t_det_max {t_det_max} --dt {dt}
"""

    path = job_dir / f"batch_{batch_idx}.slurm"
    path.write_text(content)
    path.chmod(0o755)
    return path


def main():
    # =============================
    # Ensure job_dir is unique by appending a timestamp if it exists
    # =============================
    base_job_dir = Path("jobs_tau_batches")
    job_dir      = base_job_dir
    count        = 1
    while job_dir.exists():
        job_dir = Path(f"{base_job_dir}_{count}")
        count  += 1
    log_dir = job_dir / "logs"
    job_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)

    for idx in range(TOTAL_BATCHES):
        create_batch_script(idx, TOTAL_BATCHES, job_dir, t_det_max=T_DET_MAX, dt=DT)



if __name__ == "__main__":
    main()
