#!/bin/bash
# Test script for 1D calculation and plotting workflow
#
# This script tests the new feed-forward workflow where:
# 1. calc_1D_datas.py outputs both data_path and info_path
# 2. plot_1D_datas.py accepts these paths as --data-path and --info-path arguments
# 3. Figures are automatically saved in matching directory structure
#
# New workflow benefits:
# - Direct feed-forward compatibility between calculation and plotting
# - Consistent directory structure between data and figures
# - Supports both data and info file paths for complete data loading
#
# Usage: bash test_1d_workflow.sh

echo "=========================================="
echo "TESTING 1D CALCULATION AND PLOTTING WORKFLOW"
echo "=========================================="

# Configuration - Get paths from Python configuration
echo "Loading Python path configuration..."

# Get paths from Python paths.py module
PYTHON_PATHS=$(python -c "
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('$0')), '..'))
from config.paths import PROJECT_ROOT, DATA_DIR, FIGURES_DIR, SCRIPTS_DIR
print(f'PROJECT_ROOT={PROJECT_ROOT}')
print(f'SCRIPTS_DIR={SCRIPTS_DIR}')
print(f'DATA_DIR={DATA_DIR}')
print(f'FIGURES_DIR={FIGURES_DIR}')
")

# Parse the Python output and set shell variables
eval "$PYTHON_PATHS"

echo "✅ Paths loaded from Python configuration:"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  SCRIPTS_DIR: $SCRIPTS_DIR"
echo "  DATA_DIR: $DATA_DIR"
echo "  FIGURES_DIR: $FIGURES_DIR"

# Navigate to scripts directory
cd "$SCRIPTS_DIR"

# Check if we're already in a virtual environment
echo "Checking Python environment..."
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
elif [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    echo "✅ Conda environment detected: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  No virtual environment detected, using system Python"
fi

# Check if we can run python
echo "Python version: $(python --version 2>/dev/null || echo 'Python not found')"
echo "Current working directory: $(pwd)"

# Quick package check
echo "Checking required packages..."
# Ensure we're using a non-interactive backend for matplotlib
# This prevents the "invalid command name \".!canvas\"" error when running on servers
# or when the terminal session might be interrupted
export MPLBACKEND="Agg"

echo ""
echo "=== STEP 1: Testing calculation phase ==="
echo "Running: python calc_1D_datas.py"

# Run calc_1D_datas.py and capture both data and info paths
CALC_OUTPUT=$(python calc_1D_datas.py)
CALC_EXIT_CODE=$?

if [[ $CALC_EXIT_CODE -ne 0 ]]; then
    echo "❌ Error: calc_1D_datas.py failed with exit code $CALC_EXIT_CODE"
    exit 1
fi

# Extract data and info paths from the output
DATA_PATH=$(echo "$CALC_OUTPUT" | grep "Data file:" | cut -d':' -f2- | xargs)
INFO_PATH=$(echo "$CALC_OUTPUT" | grep "Info file:" | cut -d':' -f2- | xargs)

# Validate that both paths were captured
if [[ -z "$DATA_PATH" ]]; then
    echo "❌ Error: No data file path returned from calc_1D_datas.py"
    echo "Full output:"
    echo "$CALC_OUTPUT"
    exit 1
fi

if [[ -z "$INFO_PATH" ]]; then
    echo "❌ Error: No info file path returned from calc_1D_datas.py"
    echo "Full output:"
    echo "$CALC_OUTPUT"
    exit 1
fi

echo "✅ DATA_PATH: $DATA_PATH"
echo "✅ INFO_PATH: $INFO_PATH"

# Verify files exist
if [[ ! -f "$DATA_PATH" ]]; then
    echo "❌ Error: Data file does not exist: $DATA_PATH"
    exit 1
fi

if [[ ! -f "$INFO_PATH" ]]; then
    echo "❌ Error: Info file does not exist: $INFO_PATH"
    exit 1
fi

echo "✅ Both data files verified to exist"

echo ""
echo "=== STEP 2: Testing plotting phase ==="
echo "Running: python plot_1D_datas.py --data-path \"$DATA_PATH\" --info-path \"$INFO_PATH\""

# Run plot_1D_datas.py with the captured file paths
python plot_1D_datas.py --data-path "$DATA_PATH" --info-path "$INFO_PATH"

if [[ $? -ne 0 ]]; then
    echo "❌ Error: plot_1D_datas.py failed"
    exit 1
fi

echo "✅ Plotting completed successfully"

echo ""
echo "=== WORKFLOW TEST COMPLETED ==="
echo "✅ Successfully tested the complete calculation → plotting workflow"
echo "✅ Data files: $DATA_PATH"
echo "✅ Info files: $INFO_PATH"
echo "✅ Plots should be saved in the figures directory"
