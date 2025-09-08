#!/bin/bash
# High-memory plotting job for 2D spectroscopy data

#SBATCH --job-name=plot_2d
# NOTE: Set this to your current batch job directory to store logs there
#SBATCH --chdir=/home/lbodamer/Master_thesis/code/python/scripts/batch_jobs/
#SBATCH --output=logs/plot_%j.out
#SBATCH --error=logs/plot_%j.err
#SBATCH --partition=GPGPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=0-00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de

# Ensure logs directory exists inside the batch job directory
mkdir -p logs

# Load conda (adjust to your cluster if needed)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/$USER/miniconda3/etc/profile.d/conda.sh"
fi
conda activate master_env || true

#############################################
# Run the plotting command (edit the path)  #
#############################################
# We are in batch_jobs/<current_job>/; plot_datas.py is two levels up in scripts/
python ../../plot_datas.py --abs_path \
"/home/lbodamer/Master_thesis/data/2d_spectroscopy/N2/BR/noRWA/t_dm100.0/t_wait_0.0/dt_0.1/Coupled/[15750.0, 16300.0]_data.npz"
