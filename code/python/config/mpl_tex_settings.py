"""
Matplotlib settings for LaTeX-compatible scientific plotting.

⚠️  CURRENT CONFIGURATION: HPC CLUSTER SIMULATION MODE
This file is currently configured to simulate a headless HPC environment by:
- Forcing the 'Agg' backend (no interactive display)
- Using PNG format with white backgrounds
- Ensuring all plots are saved to files

To restore normal desktop behavior, modify the BACKEND SELECTION section.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Matplotlib settings according to LaTeX caption formatting
plt.rcParams.update(
    {
        "text.usetex": True,  # Enable LaTeX for text rendering
        "font.family": "serif",  # Use a serif font family
        "font.serif": "Palatino",  # or [], Set Palatino or standard latex font
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 20,  # Font size for general text
        "axes.titlesize": 20,  # Font size for axis titles
        "axes.labelsize": 20,  # Font size for axis labels
        "xtick.labelsize": 20,  # Font size for x-axis tick labels
        "ytick.labelsize": 20,  # Font size for y-axis tick labels
        "legend.fontsize": 20,  # Font size for legends
        "figure.figsize": [8, 6],  # Size of the plot (width x height)
        "figure.autolayout": True,  # Automatic layout adjustment
        "savefig.format": "svg",  # Changed to PNG for HPC compatibility
        "savefig.dpi": 300,  # High DPI for quality output
        "savefig.bbox": "tight",  # Ensure tight bounding box
        "figure.facecolor": "white",  # White background for PNG files
        "axes.facecolor": "white",  # White axes background for PNG files
        "savefig.transparent": True,  # Disable transparency for PNG compatibility
        "savefig.facecolor": "white",  # Ensure white background when saving
    }
)

# =============================
# BACKEND SELECTION - FORCED AGG FOR HPC SIMULATION
# =============================
import os
import sys

# FORCE AGG BACKEND TO SIMULATE HPC CLUSTER ENVIRONMENT
print("🖥️  Forcing Agg backend to simulate headless HPC environment")
mpl.use("Agg")
"""
# Original backend selection logic (commented out for HPC simulation):
# Check if running in Jupyter notebook first (highest priority)
try:
    if "ipykernel" in sys.modules:
        mpl.use("module://matplotlib_inline.backend_inline")
    elif "DISPLAY" not in os.environ or os.environ.get("SLURM_JOB_ID"):
        # Use non-interactive backend for HPC/SLURM environments
        mpl.use("Agg")
    else:
        # Use TkAgg for interactive environments
        mpl.use("TkAgg")
except (ImportError, ValueError):
    # Fallback to Agg if inline backend is not available
    mpl.use("Agg")

print(f"✅ Matplotlib backend set to: {mpl.get_backend()}")
print("📊 All plots will be saved to files (no interactive display)")
"""
