#!/bin/bash
#SBATCH --job-name=2d_b000_of_001
#SBATCH --chdir=$HOME/Master_thesis/code/python/scripts
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --time=0-02:00:00
#SBATCH --mail-type="END,FAIL"
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de


# Load conda (adjust to your cluster if needed)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/$USER/miniconda3/etc/profile.d/conda.sh"
fi
conda activate master_env || true

python -u calc_datas.py --simulation_type 2d --n_batches 1 --batch_idx 0
