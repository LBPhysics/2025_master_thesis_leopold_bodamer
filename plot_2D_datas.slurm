#!/bin/bash
#SBATCH --job-name=plot_2d_data
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de

#SBATCH --time=0-00:30:00
#SBATCH --mem=16G
#SBATCH -c 1
    
# =============================
# SLURM JOB SCRIPT FOR 2D SPECTROSCOPY PLOTTING
# =============================

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate conda environment (no need for modules since conda provides Python)
source ~/.bashrc
conda activate master_env

# Navigate to scripts directory
cd code/python/scripts/

# Run the plotting script
echo "Starting 2D spectroscopy data plotting..."
python3 plot_2D_datas.py

echo "Job completed at: $(date)"
