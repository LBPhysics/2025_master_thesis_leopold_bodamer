from pathlib import Path
# one 1d calculation (t_det_max=600, dt=0.1) takes about 10 seconds, so 
# -> 6000 tau_coh_vals: -> split into 10 batches=10jobs of 600 each, each job will take 10s * 600 = 6000s = 100 minutes
# -> 6000 tau_coh_vals: -> split into 20 batches=20jobs of 300 each, each job will take 10s * 300 = 3000s = 50 minutes
# for t_det_max=100, dt=0.1, one 1d calculation takes about 2 second, so
# -> 1000 tau_coh_vals: -> split into 10 batches=10jobs of 100 each, each job will take 2s * 100 = 200s = 3.3 minutes
# -> 1000 tau_coh_vals: -> split into 20 batches=20jobs of 50 each, each job will take 2s * 50 = 100s = 1.7 minutes


TOTAL_BATCHES = 10  # You can increase/decrease this
T_DET_MAX = 100.0  # Maximum detection time in fs
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
#SBATCH --mem=1G # this should be more than enough
#SBATCH --time=0-2 # 2 hours

source /home/lbodamer/miniconda3/etc/profile.d/conda.sh
conda activate master_env

cd /home/lbodamer/Master_thesis/code/python/scripts/

python calc_1d_datas.py --batch_idx {batch_idx} --n_batches {total_batches} --t_det_max {t_det_max} --dt {dt}
"""

    path = job_dir / f"batch_{batch_idx}.slurm"
    path.write_text(content)
    path.chmod(0o755)
    return path


def main():
    job_dir = Path("jobs_tau_batches")
    log_dir = Path("logs")
    job_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    for idx in range(TOTAL_BATCHES):
        create_batch_script(idx, TOTAL_BATCHES, job_dir, t_det_max=T_DET_MAX, dt=DT)


if __name__ == "__main__":
    main()
