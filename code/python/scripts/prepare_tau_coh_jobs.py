from pathlib import Path


def create_batch_script(batch_idx, total_batches, job_dir):
    content = f"""#!/bin/bash
#SBATCH --job-name=batch_{batch_idx}
#SBATCH --output=logs/batch_{batch_idx}.out
#SBATCH --error=logs/batch_{batch_idx}.err
#SBATCH --time=0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de
#SBATCH --partition=GPGPU
#SBATCH --nodelist=orion00
#SBATCH --mem=1G
#SBATCH -c 16

source /home/lbodamer/miniconda3/etc/profile.d/conda.sh
conda activate master_env

cd /home/lbodamer/Master_thesis/code/python/scripts/

python calc_1d_datas.py --batch-idx {batch_idx} --n-batches {total_batches}
"""

    path = job_dir / f"batch_{batch_idx}.sh"
    path.write_text(content)
    path.chmod(0o755)
    return path


def main():
    total_batches = 10  # You can increase/decrease this
    job_dir = Path("jobs_tau_batches")
    log_dir = Path("logs")
    job_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    for idx in range(total_batches):
        create_batch_script(idx, total_batches, job_dir)


if __name__ == "__main__":
    main()
