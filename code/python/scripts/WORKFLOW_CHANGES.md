# Feed-Forward Workflow Implementation Summary

## Changes Made

### 1. Updated Data Saving Path Structure
- **Before**: `DATA_DIR/simulation_type/output_subdir/filename.pkl`
- **After**: `DATA_DIR/{simulation_type}d_spectroscopy/output_subdir/filename.pkl`

### 2. Modified Return Values
- Calculation scripts now return relative paths from `DATA_DIR` instead of absolute paths
- Format: `{simulation_type}d_spectroscopy/output_subdir/filename.pkl`

### 3. Added Feed-Forward Functions
- `plot_2d_from_relative_path(relative_path_str, config=None)`
- `plot_1d_from_relative_path(relative_path_str, config=None)`
- Automatically create figure directories matching data structure

### 4. Enhanced Directory Structure Consistency
- Data: `DATA_DIR/2d_spectroscopy/special_dir/unique_filename.pkl`
- Figures: `FIGURES_DIR/figures_from_python/2d_spectroscopy/special_dir/`

## Usage Examples

### Method 1: Manual Feed-Forward
```python
# Run calculation
from calc_2D_datas import main as calc_main
relative_path = calc_main()  # Returns: "2d_spectroscopy/N_2/br/t_max_4fs/2d_data_..."

# Plot using the returned path
from common_fcts import plot_2d_from_relative_path
plot_2d_from_relative_path(relative_path)
```

### Method 2: Command Line Feed-Forward
```bash
# Run calculation (prints relative path)
python calc_2D_datas.py
# Output: SAVED_DATA_PATH:2d_spectroscopy/N_2/br/t_max_4fs/2d_data_tmax_4_dt_2.0_T1_ph4_freq1_20250624_123456.pkl

# Plot using the relative path
python plot_2D_datas.py "2d_spectroscopy/N_2/br/t_max_4fs/2d_data_tmax_4_dt_2.0_T1_ph4_freq1_20250624_123456.pkl"
```

### Method 3: Script Integration
```python
# Use the test workflow script
python test_workflow.py
```

## Directory Structure Example

### After running a 2D calculation:
```
DATA_DIR/
├── 2d_spectroscopy/
│   └── N_2/
│       └── br/
│           └── t_max_4fs/
│               └── 2d_data_tmax_4_dt_2.0_T1_ph4_freq1_20250624_123456.pkl

FIGURES_DIR/
├── figures_from_python/
│   └── 2d_spectroscopy/
│       └── N_2/
│           └── br/
│               └── t_max_4fs/
│                   ├── 2d_real_component.png
│                   ├── 2d_imag_component.png
│                   ├── 2d_abs_component.png
│                   └── 2d_phase_component.png
```

## Key Benefits

1. **Consistent Structure**: Data and figure directories mirror each other
2. **Feed-Forward Compatible**: Easy to pass results between scripts
3. **Organized**: Simulation type is part of the path structure
4. **Flexible**: Supports both relative and absolute path inputs
5. **Backwards Compatible**: Original functionality still works

## Modified Files

1. `common_fcts.py`: 
   - Updated `save_spectroscopy_data()` to include simulation type in path
   - Added `plot_2d_from_relative_path()` and `plot_1d_from_relative_path()`
   - Modified `create_output_directory_from_data_path()` for consistent structure
   - Updated return value to be relative path

2. `calc_2D_datas.py`:
   - Modified to print `SAVED_DATA_PATH:` with relative path
   - Kept backwards compatibility with `OUTPUT_SUBDIR:`

3. `plot_2D_datas.py`:
   - Added support for relative path arguments
   - Auto-detects feed-forward mode vs. file path mode
   - Imports new plotting functions

4. `test_workflow.py`: 
   - New script demonstrating the complete workflow
