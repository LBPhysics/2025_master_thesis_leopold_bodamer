#!/bin/bash
#SBATCH --job-name=2D_calc_plot
#SBATCH --output=logs/2D_calc_plot_%j.out
#SBATCH --error=logs/2D_calc_plot_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de

#SBATCH --time=24:30:00
#SBATCH --mem=12G
#SBATCH -c 8

# =============================
# UNIFIED 2D SPECTROSCOPY: CALCULATION + PLOTTING
# =============================

echo "=========================================="
echo "2D Spectroscopy Calculation and Plotting"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Configuration: Calculation followed by plotting"

# Initialize conda
source /home/lbodamer/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate master_env

# Navigate to scripts directory
cd /home/lbodamer/Master_thesis/code/python/scripts/

# =============================
# STEP 1: RUN 2D DATA CALCULATION
# =============================
echo ""
echo "=========================================="
echo "STEP 1: Starting 2D spectroscopy calculation..."
echo "=========================================="
echo "Start time: $(date)"

python3 calc_2D_datas.py

# Check if calculation completed successfully
if [ $? -eq 0 ]; then
    echo "✅ 2D data calculation completed successfully at: $(date)"
else
    echo "❌ 2D data calculation failed at: $(date)"
    echo "Exiting without plotting..."
    exit 1
fi

# =============================
# STEP 2: GENERATE PLOTS
# =============================
echo ""
echo "=========================================="
echo "STEP 2: Starting 2D data plotting..."
echo "=========================================="
echo "Start time: $(date)"

python3 plot_2D_datas.py

# Check if plotting completed successfully
if [ $? -eq 0 ]; then
    echo "✅ 2D data plotting completed successfully at: $(date)"
else
    echo "❌ 2D data plotting failed at: $(date)"
    exit 1
fi

# =============================
# JOB COMPLETION
# =============================
echo ""
echo "=========================================="
echo "🎉 COMPLETE: Both calculation and plotting finished successfully!"
echo "Job completed at: $(date)"
echo "=========================================="

# Optional: Show memory usage summary
echo ""
echo "Memory usage summary:"
echo "Peak memory used: $(sacct -j $SLURM_JOB_ID --format=MaxRSS --noheader --units=M | head -1)"
