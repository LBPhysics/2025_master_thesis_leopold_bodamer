#!/bin/bash
#SBATCH --job-name=calc_2d_data
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de

#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH -c 8

# =============================
# SLURM JOB SCRIPT FOR 2D SPECTROSCOPY
# =============================

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Initialize conda
source /home/lbodamer/miniconda3/etc/profile.d/conda.sh

# Activate conda environment (no need for modules since conda provides Python)
conda activate master_env

# Navigate to scripts directory
cd /home/lbodamer/Master_thesis/code/python/scripts/

# Run the HPC-optimized simulation
echo "Starting 2D spectroscopy simulation..."
python3 calc_2D_datas.py

echo "Job completed at: $(date)"
