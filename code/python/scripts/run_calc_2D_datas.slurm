#!/bin/bash
#SBATCH --job-name=2D_unified
#SBATCH --output=logs/2D_unified_%j.out
#SBATCH --error=logs/2D_unified_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de

#SBATCH --time=3-00:00:00
#SBATCH --mem=30G
#SBATCH -c 16

# =============================
# UNIFIED 2D SPECTROSCOPY SIMULATION
# =============================

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Configuration: Unified script with dynamic paths"

# Initialize conda
source /home/lbodamer/miniconda3/etc/profile.d/conda.sh

# Activate conda environment (no need for modules since conda provides Python)
conda activate master_env

# Navigate to scripts directory
cd /home/lbodamer/Master_thesis/code/python/scripts/

# Run the unified simulation
echo "Starting unified 2D spectroscopy simulation..."
echo "Note: Modify N_atoms in calc_2D_datas.py to change configuration"
python3 calc_2D_datas.py

echo "Job completed at: $(date)"
