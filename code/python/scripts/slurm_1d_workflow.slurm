#!/bin/bash
#SBATCH --job-name=1D_calc_plot
#SBATCH --output=logs/1D_calc_plot_%j.out
#SBATCH --error=logs/1D_calc_plot_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de
#SBATCH --partition=GPGPU
#SBATCH --nodelist=orion00

#SBATCH --time=0-01:00:00
#SBATCH --mem=1G
#SBATCH -c 16

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

# Run calculation and capture output
calc_output=$(python3 calc_1D_datas.py 2>&1)
calc_exit_code=$?
echo "$calc_output"

# Check if calculation completed successfully
if [ $calc_exit_code -eq 0 ]; then
    echo "✅ 1D data calculation completed successfully at: $(date)"
    
    # Extract plotting command from output (workflow method)
    plot_cmd=$(echo "$calc_output" | grep "python plot_1D_datas.py" | tail -1)
    
    if [ -n "$plot_cmd" ]; then
        echo "📁 Found plotting command: $plot_cmd"
    else
        echo "❌ Could not extract plotting command from calculation output"
        echo "Last few lines of calculation output:"
        echo "$calc_output" | tail -10
        exit 1
    fi
else
    echo "❌ 1D data calculation failed at: $(date)"
    echo "Calculation output:"
    echo "$calc_output"
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
echo "📊 Executing: $plot_cmd"

# Execute the plotting command
eval $plot_cmd
plot_exit_code=$?

# Check if plotting completed successfully
if [ $plot_exit_code -eq 0 ]; then
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
echo ""
echo "📁 Generated files:"
echo "$calc_output" | grep "Data file:" | tail -1
echo "$calc_output" | grep "Info file:" | tail -1
echo ""
echo "📊 Plots saved to: figures/1d_spectroscopy/plots/"
echo "=========================================="

# Optional: Show memory usage summary
echo ""
echo "Memory usage summary:"
echo "Peak memory used: $(sacct -j $SLURM_JOB_ID --format=MaxRSS --noheader --units=M | head -1)"