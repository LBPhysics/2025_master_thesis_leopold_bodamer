#!/bin/bash
# Test script for 2D calculation and plotting workflow

echo "=========================================="
echo "TESTING 2D CALCULATION AND PLOTTING WORKFLOW"
echo "=========================================="

# Navigate to scripts directory
cd /home/leopold/Projects/Master_thesis/code/python/scripts/

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
echo "=== STEP 2: Testing output subdirectory extraction ==="

# Extract output subdirectory
output_subdir=$(echo "$calc_output" | grep "OUTPUT_SUBDIR:" | cut -d':' -f2)

echo "Raw output subdir line: $(echo "$calc_output" | grep "OUTPUT_SUBDIR:")"
echo "Extracted output subdir: '$output_subdir'"

if [ -z "$output_subdir" ]; then
    echo "❌ No output subdirectory extracted! Check if calc script prints OUTPUT_SUBDIR:"
    exit 1
fi

# Check if output directory exists (it should contain the data files)
output_dir="/home/leopold/Projects/Master_thesis/code/python/data/$output_subdir"
if [ -d "$output_dir" ]; then
    echo "✅ Output directory exists: $output_dir"
    echo "Directory contents: $(ls -la "$output_dir" | wc -l) items"
else
    echo "❌ Output directory does not exist: $output_dir"
    exit 1
fi

echo ""
echo "=== STEP 3: Testing plotting phase ==="
echo "Running: python3 plot_2D_datas.py \"$output_subdir\""
echo ""

# Run plotting
plot_start_time=$(date)
python3 plot_2D_datas.py "$output_subdir"
plot_exit_code=$?

echo ""
echo "Plotting exit code: $plot_exit_code"
echo "Plotting completed at: $(date)"

if [ $plot_exit_code -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS: Both calculation and plotting completed successfully!"
    echo "📁 Data directory: $output_dir"
    echo "📊 Check figures directory for plots"
else
    echo ""
    echo "❌ Plotting failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "TEST COMPLETED"
echo "=========================================="
