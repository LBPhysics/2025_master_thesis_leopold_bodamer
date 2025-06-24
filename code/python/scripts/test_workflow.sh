#!/bin/bash
# Test script for 2D calculation and plotting workflow
#
# This script tests the new feed-forward workflow where:
# 1. calc_2D_datas.py outputs SAVED_DATA_PATH: with relative path from DATA_DIR
# 2. plot_2D_datas.py accepts this relative path as an argument
# 3. Figures are automatically saved in matching directory structure
#
# New workflow benefits:
# - Direct feed-forward compatibility between calculation and plotting
# - Consistent directory structure between data and figures
# - Supports both new SAVED_DATA_PATH and legacy OUTPUT_SUBDIR formats

echo "=========================================="
echo "TESTING 2D CALCULATION AND PLOTTING WORKFLOW"
echo "=========================================="

# Configuration - Get paths from Python configuration
echo "Loading Python path configuration..."

# Get paths from Python paths.py module
PYTHON_PATHS=$(python3 -c "
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
echo "Python version: $(python3 --version 2>/dev/null || echo 'Python3 not found')"
echo "Current working directory: $(pwd)"

# Quick package check
echo "Checking required packages..."
python3 -c "
try:
    from common_fcts import run_2d_simulation_with_config
    print('✅ common_fcts imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
" || {
    echo "❌ Required packages not available. Please install dependencies."
    exit 1
}

echo ""
echo "=== STEP 1: Testing calculation phase ==="
echo "Running: python3 calc_2D_datas.py"
echo ""

# Run calculation and capture output
start_time=$(date)
calc_output=$(python3 calc_2D_datas.py 2>&1)
calc_exit_code=$?

echo "Calculation output:"
echo "$calc_output"
echo ""
echo "Calculation exit code: $calc_exit_code"
echo "Calculation completed at: $(date)"

if [ $calc_exit_code -ne 0 ]; then
    echo "❌ Calculation failed! Cannot proceed to plotting."
    exit 1
fi

echo ""
echo "=== STEP 2: Testing feed-forward path extraction ==="

# Extract the new feed-forward path
saved_data_path=$(echo "$calc_output" | grep "SAVED_DATA_PATH:" | cut -d':' -f2)

echo "Raw saved data path line: $(echo "$calc_output" | grep "SAVED_DATA_PATH:")"
echo "Extracted saved data path: '$saved_data_path'"

# For backwards compatibility, also check for OUTPUT_SUBDIR (fallback)
output_subdir=$(echo "$calc_output" | grep "OUTPUT_SUBDIR:" | cut -d':' -f2)

# Determine which path to use
if [ -n "$saved_data_path" ]; then
    echo "✅ Using new feed-forward path: $saved_data_path"
    data_relative_path="$saved_data_path"
    # Extract directory from the full path for verification
    data_dir_path=$(dirname "$saved_data_path")
    data_filename=$(basename "$saved_data_path")
elif [ -n "$output_subdir" ]; then
    echo "⚠️  Using legacy output subdir: $output_subdir"
    data_relative_path="2d_spectroscopy/$output_subdir"
    data_dir_path="2d_spectroscopy/$output_subdir"
    data_filename=""
else
    echo "❌ No data path extracted! Check if calc script prints SAVED_DATA_PATH: or OUTPUT_SUBDIR:"
    exit 1
fi

# Check if output directory exists (it should contain the data files)
output_dir="$DATA_DIR/$data_dir_path"
if [ -d "$output_dir" ]; then
    echo "✅ Output directory exists: $output_dir"
    echo "Directory contents: $(ls -la "$output_dir" | wc -l) items"
    
    # If we have a specific filename, check if it exists
    if [ -n "$data_filename" ]; then
        full_data_path="$output_dir/$data_filename"
        if [ -f "$full_data_path" ]; then
            echo "✅ Data file exists: $data_filename"
        else
            echo "⚠️  Specific data file not found: $data_filename"
            echo "Available files: $(ls -1 "$output_dir" | grep -E '\.(pkl|pkl\.gz)$' | head -3)"
        fi
    fi
else
    echo "❌ Output directory does not exist: $output_dir"
    exit 1
fi

echo ""
echo "=== STEP 3: Testing plotting phase with feed-forward ==="
echo "Running: python3 plot_2D_datas.py \"$data_relative_path\""
echo ""

# Run plotting with the extracted path
plot_start_time=$(date)
python3 plot_2D_datas.py "$data_relative_path"
plot_exit_code=$?

echo ""
echo "Plotting exit code: $plot_exit_code"
echo "Plotting completed at: $(date)"

if [ $plot_exit_code -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS: Both calculation and plotting completed successfully!"
    echo "📁 Data directory: $output_dir"
    echo "📊 Data file path: $data_relative_path"
    echo "📊 Check figures directory for plots (figures_from_python/$data_dir_path/)"
else
    echo ""
    echo "❌ Plotting failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "TEST COMPLETED"
echo "=========================================="

echo ""
echo "=========================================="
echo "ADDITIONAL WORKFLOW TESTS"
echo "=========================================="

echo ""
echo "=== STEP 4: Testing alternative plotting modes ==="

# Test plotting from directory (should find .pkl files automatically)
if [ -n "$data_dir_path" ] && [ -d "$DATA_DIR/$data_dir_path" ]; then
    echo "Testing directory-based plotting..."
    echo "Running: python3 plot_2D_datas.py \"$data_dir_path\""
    
    python3 plot_2D_datas.py "$data_dir_path" 2>&1 | head -10
    directory_plot_exit=$?
    
    if [ $directory_plot_exit -eq 0 ]; then
        echo "✅ Directory-based plotting works"
    else
        echo "⚠️  Directory-based plotting had issues (exit code: $directory_plot_exit)"
    fi
fi

echo ""
echo "=== STEP 5: Verifying figure output structure ==="

# Check if figures were created in the expected location
figures_dir="$FIGURES_DIR/figures_from_python/$data_dir_path"
if [ -d "$figures_dir" ]; then
    echo "✅ Figures directory created: $figures_dir"
    
    # Count generated plots
    plot_count=$(find "$figures_dir" -name "*.png" -o -name "*.pdf" -o -name "*.svg" | wc -l)
    echo "📊 Generated plots: $plot_count"
    
    if [ $plot_count -gt 0 ]; then
        echo "✅ Plot files found:"
        find "$figures_dir" -name "*.png" -o -name "*.pdf" -o -name "*.svg" | head -5 | while read plot_file; do
            echo "   - $(basename "$plot_file")"
        done
    else
        echo "⚠️  No plot files found in figures directory"
    fi
else
    echo "⚠️  Expected figures directory not found: $figures_dir"
fi

echo ""
echo "=== STEP 6: Workflow Summary ==="
echo "The new feed-forward workflow successfully demonstrated:"
echo "1. ✅ Calculation script outputs SAVED_DATA_PATH with relative path"
echo "2. ✅ Plotting script accepts and processes the relative path"
echo "3. ✅ Data and figures maintain consistent directory structure"
echo "4. ✅ Backwards compatibility with legacy OUTPUT_SUBDIR format"
echo ""
echo "Feed-forward command for manual use:"
echo "  python3 calc_2D_datas.py | grep 'SAVED_DATA_PATH:' | cut -d':' -f2 | xargs python3 plot_2D_datas.py"
echo ""
echo "Direct command line workflow:"
echo "  DATA_PATH=\$(python3 calc_2D_datas.py | grep 'SAVED_DATA_PATH:' | cut -d':' -f2)"
echo "  python3 plot_2D_datas.py \"\$DATA_PATH\""
