#!/bin/bash
#SBATCH --job-name=1D_calc_plot
#SBATCH --output=logs/1D_calc_plot_%j.out
#SBATCH --error=logs/1D_calc_plot_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de

#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH -c 4

# =============================
# UNIFIED 1D SPECTROSCOPY: CALCULATION + PLOTTING
# =============================

echo "=========================================="
echo "1D Spectroscopy Calculation and Plotting"
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
# STEP 1: RUN 1D DATA CALCULATION
# =============================
echo ""
echo "=========================================="
echo "STEP 1: Starting 1D spectroscopy calculation..."
echo "=========================================="
echo "Start time: $(date)"

# Run calculation and capture the filepath
calc_output=$(python3 calc_1D_datas.py)
echo "$calc_output"

# Extract the saved filepath from the output
saved_filepath=$(echo "$calc_output" | grep "SAVED_FILEPATH:" | cut -d':' -f2)

# Check if calculation completed successfully
if [ $? -eq 0 ] && [ ! -z "$saved_filepath" ]; then
    echo "✅ 1D data calculation completed successfully at: $(date)"
    echo "📁 Data saved to: $saved_filepath"
else
    echo "❌ 1D data calculation failed at: $(date)"
    echo "Exiting without plotting..."
    exit 1
fi

# =============================
# STEP 2: GENERATE PLOTS
# =============================
echo ""
echo "=========================================="
echo "STEP 2: Starting 1D data plotting..."
echo "=========================================="
echo "Start time: $(date)"
echo "📊 Plotting data from: $saved_filepath"

python3 plot_1D_datas.py "$saved_filepath"

# Check if plotting completed successfully
if [ $? -eq 0 ]; then
    echo "✅ 1D data plotting completed successfully at: $(date)"
else
    echo "❌ 1D data plotting failed at: $(date)"
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
