#!/bin/bash
#SBATCH --job-name=2d_spectroscopy
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de
#SBATCH --partition=metis,hermesv2,hermesv1,kratos48G,kratos96G,calypso,hermesBigMem,metisBigMem

#SBATCH --time=24:00:00
#SBATCH --mem=14G
#SBATCH -c 8

# =============================
# SLURM JOB SCRIPT FOR 2D SPECTROSCOPY
# =============================

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Activate conda environment (no need for modules since conda provides Python)
source ~/.bashrc
conda activate master_env

# Create logs directory
mkdir -p logs

# Navigate to scripts directory
cd code/python/scripts/

# Run the HPC-optimized simulation
echo "Starting 2D spectroscopy simulation..."
python3 calc_2D_datas_hpc.py --config test # TODO change this to the actual config file

echo "Job completed at: $(date)"
