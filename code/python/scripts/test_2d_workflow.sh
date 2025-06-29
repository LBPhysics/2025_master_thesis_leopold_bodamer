#!/bin/bash
# =============================================================================
# 2D Spectroscopy Workflow Test Script
# =============================================================================
# This script runs the complete 2D spectroscopy workflow:
# 1. Generate data with calc_2D_datas.py
# 2. Automatically plot the results with plot_2D_datas.py
#
# Usage: ./test_2d_workflow.sh
# =============================================================================

set -e  # Exit on any error

echo "🚀 Starting 2D Spectroscopy Workflow Test"
echo "=========================================="

# =============================
# STEP 1: RUN SIMULATION
# =============================
echo "📊 Step 1: Running 2D simulation..."
echo ""

# Run the calculation and capture the output
echo "Executing: python calc_2D_datas.py"
python calc_2D_datas.py > calc_output.log 2>&1

# Check if simulation completed successfully
if [ $? -eq 0 ]; then
echo "✅ Simulation completed successfully!"
else
echo "❌ Simulation failed! Check calc_output.log for details."
exit 1
fi

# =============================
# STEP 2: EXTRACT PATHS FROM OUTPUT
# =============================
echo ""
echo "📂 Step 2: Extracting file paths..."

# Extract the plotting command from the output
PLOT_CMD=$(grep "python plot_2D_datas.py" calc_output.log | tail -1)

if [ -z "$PLOT_CMD" ]; then
echo "❌ Could not find plotting command in output!"
echo "Last few lines of calc_output.log:"
tail -10 calc_output.log
exit 1
fi

echo "Found plotting command: $PLOT_CMD"

# =============================
# STEP 3: RUN PLOTTING
# =============================
echo ""
echo "🎨 Step 3: Running plotting script..."
echo "Executing: $PLOT_CMD"

# Execute the plotting command
eval $PLOT_CMD

# Check if plotting completed successfully
if [ $? -eq 0 ]; then
echo ""
echo "✅ Plotting completed successfully!"
else
echo "❌ Plotting failed!"
exit 1
fi

# =============================
# WORKFLOW COMPLETE
# =============================
echo ""
echo "🎉 2D Spectroscopy Workflow Completed Successfully!"
echo "================================================="
echo ""
echo "📁 Generated files:"
grep "Data file:" calc_output.log | tail -1
grep "Info file:" calc_output.log | tail -1
echo ""
echo "📊 Plots saved to: figures/2d_spectroscopy/plots/"
echo ""
echo "🔍 Full simulation log available in: calc_output.log"
echo "================================================="

# Clean up
rm -f calc_output.log