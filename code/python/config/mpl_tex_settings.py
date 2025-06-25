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
from matplotlib import font_manager as fm
from matplotlib import rcParams

# =============================
# PLOTTING SETTINGS
# =============================

DEFAULT_FIGSIZE = [10, 8]
DEFAULT_DPI = 10  # 100 is very high, 10 is good for notebooks
DEFAULT_FONT_SIZE = 16
DEFAULT_FIG_FORMAT = "svg"  # pdf, png, svg
COLORS = {
    "C0": "#1f77b4",
    "C1": "#ff7f0e",
    "C2": "#2ca02c",
    "C3": "#d62728",
    "C4": "#9467bd",
    "C5": "#8c564b",
    "C6": "#e377c2",
    "C7": "#7f7f7f",
    "C8": "#bcbd22",
    "C9": "#17becf",
}

LINE_STYLES = ["solid", "dashed", "dashdot", "dotted", (0, (3, 1, 1, 1)), (0, (5, 1))]
MARKERS = ["o", "s", "^", "v", "D", "p", "*", "X", "+", "x"]


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
        result = subprocess.run(
            ["pdflatex", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            return shutil.which("latex") is not None
        except:
            return False


def get_available_fonts(keywords):
    """Return a list of available fonts that match given keywords."""
    available = set()
    for f in fm.fontManager.ttflist:
        for kw in keywords:
            if kw in f.name.lower():
                available.add(f.name)
    return sorted(available)


def set_best_serif_font():
    preferred_order = [
        "palatino linotype",
        "palatino",
        "cmu serif",  # latex imitate
        "times new roman",
    ]
    available = get_available_fonts(preferred_order)
    for pref in preferred_order:
        match = next((f for f in available if pref == f.lower()), None)
        if match:
            rcParams["font.family"] = match
            global font_to_use
            font_to_use = match
            print(f"✔️ Using font: {match}")
            return
    print("⚠️ No preferred serif fonts found.")
    font_to_use = "serif"


# Initialize font_to_use
font_to_use = "serif"  # fall back
set_best_serif_font()  # update font to use
latex_available = (
    check_latex_available()
)  # what happens with FONT_TO_USE if latex is not available??

# =============================
# MATPLOTLIB LaTeX SETTINGS
# =============================
base_settings = {
    # font
    "font.family": "serif",
    "font.serif": [
        font_to_use,
        "cmu serif",
        "times new roman",
        "serif",
    ],
    # font sizes
    "font.size": DEFAULT_FONT_SIZE,
    "axes.titlesize": DEFAULT_FONT_SIZE + 4,
    "axes.labelsize": DEFAULT_FONT_SIZE + 2,
    "xtick.labelsize": DEFAULT_FONT_SIZE,
    "ytick.labelsize": DEFAULT_FONT_SIZE,
    "legend.fontsize": DEFAULT_FONT_SIZE,
    # layout
    "figure.figsize": DEFAULT_FIGSIZE,
    "figure.autolayout": True,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.framealpha": 0.8,
    "savefig.transparent": True,
    # "figure.facecolor": "white", # Uncomment if you want white background, but only with transparent=False
    #    "axes.facecolor": "white",
    #    "savefig.facecolor": "white",
    # Quality of the plot:
    "savefig.format": DEFAULT_FIG_FORMAT,
    "savefig.dpi": DEFAULT_DPI,
}

if latex_available:
    latex_settings = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{physics}\usepackage{mathpazo}",
    }
    base_settings.update(latex_settings)
else:
    non_latex_settings = {
        "text.usetex": False,
        "mathtext.default": "regular",
    }
    base_settings.update(non_latex_settings)

plt.rcParams.update(base_settings)

# =============================
# HELPER FUNCTIONS
# =============================


def set_size(width_pt=510, fraction=1, subplots=(1, 1), height_ratio=None):
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    if height_ratio is None:
        height_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * height_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def format_sci_notation(x, decimals=1, include_dollar=True):
    if x == 0:
        return r"$0$" if include_dollar else r"0"
    exp = int(np.floor(np.log10(abs(x))))
    coef = round(x / 10**exp, decimals)
    mult_symbol = r" \times " if latex_available else r" \cdot "
    if coef == 1:
        result = r"10^{" + str(exp) + r"}"
    else:
        result = str(coef) + mult_symbol + r"10^{" + str(exp) + r"}"
    return r"$" + result + r"$" if include_dollar else result


def save_fig(
    fig,
    filename,
    formats=["svg", "png", "pdf"],
    dpi=DEFAULT_DPI,
    transparent=False,
    category=None,
    output_dir=None,
):
    try:
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
        using_paths_module = False
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "figures",
            "figures_from_python",
        )

    if filename and os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        full_path = filename
    elif output_dir:
        # Use provided output_dir directly
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
    else:
        if category:
            category_name = category
        else:
            calling_script = os.path.basename(sys.argv[0]).replace(".py", "")
            category_name = get_figure_category(calling_script)

        if using_paths_module:
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
            save_dir = os.path.join(base_dir, category_name)

        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)

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
        return "misc"


# =============================
# BACKEND SELECTION - AUTOMATIC BASED ON ENVIRONMENT
# =============================


def setup_backend():
    if mpl.get_backend() not in ["", "agg"]:
        return
    try:
        if "ipykernel" in sys.modules:
            mpl.use("module://matplotlib_inline.backend_inline")
            print("✅ Matplotlib backend: Jupyter inline")
        elif "DISPLAY" not in os.environ or os.environ.get("SLURM_JOB_ID"):
            mpl.use("Agg")
            print("✅ Matplotlib backend: Agg (headless/HPC environment)")
        else:
            mpl.use("TkAgg")
            print("✅ Matplotlib backend: TkAgg (interactive)")
    except (ImportError, ValueError) as e:
        mpl.use("Agg")
        print(f"ℹ️ Matplotlib backend set to Agg (fallback): {e}")


setup_backend()

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
    "latex_available",
    "font_to_use",
]

print(f"📊 Matplotlib settings loaded")
print(
    f"   - LaTeX rendering: {'Enabled' if latex_available else 'Disabled (fallback to mathtext)'}"
)
print(f"   - Font: {font_to_use}")
print(f"   - Backend: {mpl.get_backend()}")
print(f"   - Default figure size: {DEFAULT_FIGSIZE}")
