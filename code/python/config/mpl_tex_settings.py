"""
Matplotlib settings for LaTeX-compatible scientific plotting.

This file contains consistent matplotlib settings for the Master's thesis project.
Import this module at the beginning of any script to apply these settings automatically.

Example:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.mpl_tex_settings import *  # Apply matplotlib settings
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys
import shutil
import subprocess
from matplotlib.font_manager import findfont, FontProperties

# =============================
# SYSTEM CHECKS FOR FAILSAFE
# =============================


def check_latex_available():
    """
    Check if LaTeX is installed and available in the system path.

    Returns:
    --------
    bool
        True if LaTeX is available, False otherwise
    """
    try:
        # Try to run pdflatex with version flag
        result = subprocess.run(
            ["pdflatex", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            # Alternative: check if latex is in PATH
            return shutil.which("latex") is not None
        except:
            return False


def check_font_available(font_name):
    """
    Check if a specified font is available in matplotlib.

    Parameters:
    -----------
    font_name : str
        Name of the font to check

    Returns:
    --------
    bool
        True if font is available, False otherwise
    """
    try:
        font_path = findfont(FontProperties(family=font_name))
        # If the default font is returned instead of the requested one,
        # it means the font is not available
        return (
            os.path.basename(font_path).lower().startswith(font_name.lower())
            or font_name.lower() in os.path.basename(font_path).lower()
        )
    except:
        return False


# Check if LaTeX is available
latex_available = check_latex_available()

# Define preferred and fallback fonts
preferred_font = "Palatino"
fallback_fonts = ["Times", "DejaVu Serif", "Computer Modern Roman", "serif"]

# Find first available font
font_to_use = preferred_font  # Default to preferred
if not check_font_available(preferred_font):
    for font in fallback_fonts:
        if check_font_available(font):
            font_to_use = font
            break
    # If none of the specified fonts are available, use the system's default serif font
    if (
        font_to_use == preferred_font
    ):  # Still on preferred, means none of fallbacks found
        font_to_use = "serif"

# =============================
# MATPLOTLIB LaTeX SETTINGS
# =============================

# Base settings common to both LaTeX and non-LaTeX modes
base_settings = {
    "font.family": "serif",  # Use a serif font family
    "font.serif": [font_to_use],  # Use the available font
    "font.size": 18,  # Font size for general text
    "axes.titlesize": 20,  # Font size for axis titles
    "axes.labelsize": 18,  # Font size for axis labels
    "xtick.labelsize": 16,  # Font size for x-axis tick labels
    "ytick.labelsize": 16,  # Font size for y-axis tick labels
    "legend.fontsize": 16,  # Font size for legends
    "figure.figsize": [10, 8],  # Size of the plot (width x height)
    "figure.autolayout": True,  # Automatic layout adjustment
    "savefig.format": "svg",  # SVG for vector graphics in LaTeX
    "savefig.dpi": 300,  # High DPI for quality output
    "savefig.bbox": "tight",  # Ensure tight bounding box
    "figure.facecolor": "white",  # White background
    "axes.facecolor": "white",  # White axes background
    "savefig.transparent": False,  # Disable transparency for consistency
    "savefig.facecolor": "white",  # Ensure white background when saving
    "axes.grid": False,  # No grid lines by default
    "axes.axisbelow": True,  # Place grid lines below plots
    "legend.frameon": True,  # Show legend frame
    "legend.fancybox": True,  # Rounded corners on legend
    "legend.framealpha": 0.8,  # Partial transparency for legend background
}

# LaTeX-specific settings
if latex_available:
    latex_settings = {
        "text.usetex": True,  # Enable LaTeX for text rendering
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{physics}",  # Add physics package for better notation
    }
    # Update base settings with LaTeX settings
    base_settings.update(latex_settings)
else:
    # Non-LaTeX fallback settings
    non_latex_settings = {
        "text.usetex": False,
        "mathtext.default": "regular",  # Use regular mathtext if LaTeX is not available
    }
    # Update base settings with non-LaTeX settings
    base_settings.update(non_latex_settings)

# Apply all settings
plt.rcParams.update(base_settings)

# =============================
# PLOTTING SETTINGS
# =============================

# Default figure settings
DEFAULT_FIGSIZE = (10, 8)
DEFAULT_DPI = 300
DEFAULT_FONT_SIZE = 18

# Color palette for consistent plotting
COLORS = {
    "C0": "#1f77b4",  # blue
    "C1": "#ff7f0e",  # orange
    "C2": "#2ca02c",  # green
    "C3": "#d62728",  # red
    "C4": "#9467bd",  # purple
    "C5": "#8c564b",  # brown
    "C6": "#e377c2",  # pink
    "C7": "#7f7f7f",  # gray
    "C8": "#bcbd22",  # olive
    "C9": "#17becf",  # cyan
}

# Line styles for distinguishing multiple curves
LINE_STYLES = ["solid", "dashed", "dashdot", "dotted", (0, (3, 1, 1, 1)), (0, (5, 1))]

# Marker styles
MARKERS = ["o", "s", "^", "v", "D", "p", "*", "X", "+", "x"]

# =============================
# HELPER FUNCTIONS
# =============================


def set_size(width_pt=510, fraction=1, subplots=(1, 1), height_ratio=None):
    """
    Set figure dimensions to match LaTeX document dimensions.

    Parameters:
    -----------
    width_pt : float
        Document width in points (510 is for a standard LaTeX article)
    fraction : float
        Fraction of the width to use (default: 1)
    subplots : tuple
        Number of rows and columns of subplots
    height_ratio : float
        Aspect ratio (height/width), if None, golden ratio is used

    Returns:
    --------
    fig_dim : tuple
        Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    if height_ratio is None:
        height_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    # Figure height in inches
    fig_height_in = fig_width_in * height_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def format_sci_notation(x, decimals=1, include_dollar=True):
    """
    Format number in scientific notation for plot labels.
    Works with both LaTeX and non-LaTeX rendering.

    Parameters:
    -----------
    x : float
        Number to format
    decimals : int
        Number of decimal places
    include_dollar : bool
        Whether to include dollar signs in the output.
        Set to False when embedding in a larger LaTeX expression.

    Returns:
    --------
    formatted_string : str
        LaTeX-formatted string
    """
    if x == 0:
        return r"$0$" if include_dollar else r"0"

    exp = int(np.floor(np.log10(abs(x))))
    coef = round(x / 10**exp, decimals)

    # Handle the coefficient and exponent formatting
    if coef == 1:
        result = r"10^{" + str(exp) + r"}"
    else:
        # Choose the appropriate multiplication symbol
        if latex_available:
            mult_symbol = r" \times "
        else:
            mult_symbol = r" \cdot "
        result = str(coef) + mult_symbol + r"10^{" + str(exp) + r"}"

    # Add dollar signs if requested
    if include_dollar:
        return r"$" + result + r"$"
    else:
        return result


def save_fig(
    fig, filename, formats=["svg", "png"], dpi=300, transparent=False, category=None
):
    """
    Save figure in multiple formats for different uses.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Base filename without extension
    formats : list
        List of formats to save as
    dpi : int
        Resolution for raster formats
    transparent : bool
        Whether to use transparency
    category : str
        Optional category to override automatic determination
    """
    try:
        # Try to import paths from config
        from config.paths import (
            FIGURES_PYTHON_DIR,
            FIGURES_1D_DIR,
            FIGURES_2D_DIR,
            FIGURES_BATH_DIR,
            FIGURES_PULSES_DIR,
            FIGURES_TESTS_DIR,
        )

        using_paths_module = True
    except ImportError:
        # Fallback to old method if paths module is not available
        using_paths_module = False
        # Get the base directory structure using relative paths
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "figures",
            "figures_from_python",
        )

    # Create directory if it doesn't exist
    # First check if filename has a directory component
    if filename and os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        full_path = filename
    else:
        # If filename is just a base name without path, add the default path
        if category:
            # Use the provided category
            category_name = category
        else:
            # Get script name to determine category
            calling_script = os.path.basename(sys.argv[0]).replace(".py", "")
            category_name = get_figure_category(calling_script)

        # Determine save directory based on category
        if using_paths_module:
            # Use the paths from the paths module
            if category_name == "tests" or "test" in calling_script.lower():
                save_dir = FIGURES_TESTS_DIR
            elif category_name == "1d_spectroscopy":
                save_dir = FIGURES_1D_DIR
            elif category_name == "2d_spectroscopy":
                save_dir = FIGURES_2D_DIR
            elif category_name == "bath_correlator":
                save_dir = FIGURES_BATH_DIR
            elif category_name == "pulses":
                save_dir = FIGURES_PULSES_DIR
            else:
                save_dir = FIGURES_PYTHON_DIR / category_name
        else:
            # Use the old method
            save_dir = os.path.join(base_dir, category_name)

        # Create the directory and construct the full path
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)

    # Save in all requested formats
    for fmt in formats:
        fig.savefig(
            f"{full_path}.{fmt}",
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )

    print(f"Figure saved as: {full_path}")


def get_figure_category(script_name):
    """
    Determine appropriate figure category based on script name.

    Parameters:
    -----------
    script_name : str
        Name of the calling script

    Returns:
    --------
    category : str
        Figure category folder name
    """
    if "test" in script_name.lower():
        return "tests"
    elif "1d_" in script_name or "_1d" in script_name:
        return "1d_spectroscopy"
    elif "2d_" in script_name or "_2d" in script_name:
        return "2d_spectroscopy"
    elif "bath" in script_name or "correlation" in script_name:
        return "bath_correlator"
    elif "pulse" in script_name:
        return "pulses"
    elif "dani" in script_name:
        return "replicate_danis_work"
    elif "paper" in script_name or "br" in script_name or "BR" in script_name:
        return "replicate_paper_with_BR"
    else:
        return "misc"  # Default category


# =============================
# BACKEND SELECTION - AUTOMATIC BASED ON ENVIRONMENT
# =============================


def setup_backend():
    """
    Set the appropriate matplotlib backend based on execution environment.
    - Jupyter notebook: inline backend for integrated display
    - HPC/SLURM: Agg backend for headless environments
    - Regular terminal: TkAgg for interactive display
    """
    # Check if backend has already been set explicitly
    if mpl.get_backend() not in ["", "agg"]:
        return

    # Check if running in Jupyter notebook first (highest priority)
    try:
        if "ipykernel" in sys.modules:
            mpl.use("module://matplotlib_inline.backend_inline")
            print("✅ Matplotlib backend: Jupyter inline")
        elif "DISPLAY" not in os.environ or os.environ.get("SLURM_JOB_ID"):
            # Use non-interactive backend for HPC/SLURM environments
            mpl.use("Agg")
            print("✅ Matplotlib backend: Agg (headless/HPC environment)")
        else:
            # Use TkAgg for interactive environments
            mpl.use("TkAgg")
            print("✅ Matplotlib backend: TkAgg (interactive)")
    except (ImportError, ValueError) as e:
        # Fallback to Agg if other backends are not available
        mpl.use("Agg")
        print(f"ℹ️ Matplotlib backend set to Agg (fallback): {e}")


# Set the backend based on environment
setup_backend()

# =============================
# EXPORT FUNCTIONS AND SETTINGS
# =============================

__all__ = [
    "DEFAULT_FIGSIZE",
    "DEFAULT_DPI",
    "DEFAULT_FONT_SIZE",
    "COLORS",
    "LINE_STYLES",
    "MARKERS",
    "set_size",
    "format_sci_notation",
    "save_fig",
    "check_latex_available",
    "check_font_available",
    "latex_available",
    "font_to_use",
]

# Print confirmation of successful import
print(f"📊 Matplotlib settings loaded")
print(
    f"   - LaTeX rendering: {'Enabled' if latex_available else 'Disabled (fallback to mathtext)'}"
)
if font_to_use == preferred_font:
    print(f"   - Font: {font_to_use}")
else:
    print(f"   - Font: {font_to_use} (fallback from {preferred_font})")
print(f"   - Backend: {mpl.get_backend()}")
print(f"   - Default figure size: {DEFAULT_FIGSIZE}")
