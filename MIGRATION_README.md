"""
MIGRATION NOTICE: common_fcts.py has been reorganized

The functionality from this file has been moved to the proper qspectro2d package structure:

OLD LOCATION: scripts/common_fcts.py
NEW LOCATIONS:
- Data I/O functions → qspectro2d.data.io
- File utilities → qspectro2d.data.files  
- Simulation runners → qspectro2d.simulation.runners
- Simulation utilities → qspectro2d.simulation.utils
- Plotting functions → qspectro2d.visualization.data_plots

UPDATED IMPORTS:
Instead of: from common_fcts import load_data_from_rel_path, save_simulation_data
Use: from qspectro2d.data import load_data_from_rel_path, save_simulation_data

Instead of: from common_fcts import run_1d_simulation, create_system_parameters
Use: from qspectro2d.simulation import run_1d_simulation, create_system_parameters

Instead of: from common_fcts import plot_1d_data, plot_2d_data
Use: from qspectro2d.visualization import plot_1d_data, plot_2d_data

BENEFITS:
✅ No more sys.path.append() hacks
✅ Clean package structure
✅ Better code organization
✅ Easier imports from notebooks
✅ Follows Python conventions

The old file has been backed up as common_fcts.py.backup
"""
