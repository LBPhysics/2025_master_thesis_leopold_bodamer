#!/bin/bash
#SBATCH --job-name=plot2d
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --chdir=$HOME/Master_thesis/code/python/scripts
#SBATCH --partition=GPGPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G
#SBATCH --time=00:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de

# Load Python or activate your environment if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate master_env

# Run from the scripts directory so plot_datas.py resolves correctly
srun python plot_datas.py --abs_path "$HOME/Master_thesis/data/2d_spectroscopy/N2/f15640-16360cm_mu1.0-1.0/Paper_eqs/RWA/t_dm40.0_t_wait_0.0_dt_1.0/2d_data.npz"