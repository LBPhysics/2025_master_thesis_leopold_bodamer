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

import os
from pathlib import Path
from typing import Optional, Union, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys
import shutil
import subprocess
from matplotlib import font_manager as fm
from matplotlib import rcParams
from config.paths import FIGURES_DIR

LATEX_DOC_WIDTH = 441.01775  # 510  # Default LaTeX document width in points (pt); FIND OUT WITH "The width of this document is: \the\textwidth"
LATEX_FONT_SIZE = 11  # Default LaTeX font size in points (pt)


# =============================
# HELPER FUNCTIONS
# =============================


def set_size(
    width_pt: float = 51,
    fraction: float = 0.5,
    subplots: tuple = (1, 1),
    height_ratio: Optional[float] = None,
) -> tuple:
    """
    Calculate figure dimensions for optimal display in LaTeX documents.

    This function calculates the optimal figure dimensions based on the text width of your
    LaTeX document. It ensures that figures will perfectly fit within the document without
    requiring scaling, which can distort text and make it inconsistent with the document.

    Matplotlib uses inches as its default unit for figure sizes, which is why the calculation
    converts from LaTeX points to inches. This is historical: matplotlib was originally
    designed with US-based plotting conventions.

    Parameters:
    -----------
    width_pt : float, optional
        Width of the LaTeX document's text in points (pt). For a standard LaTeX article
        or thesis, this is typically around 510pt for a single column. You can determine
        this value by adding `\\the\\textwidth` in your LaTeX document. Default is 510pt.

    fraction : float, optional
        Fraction of the text width that the figure should occupy. For example, 0.5 for
        half-width figures, 1.0 for full-width figures. Default is 1.0.

    subplots : tuple, optional
        Grid dimensions for subplots as (rows, columns). This affects the height calculation
        to maintain proper proportions when using multiple subplots. Default is (1, 1).

    height_ratio : float, optional
        Ratio of height to width. If None, the golden ratio (≈0.618) is used, which is
        considered aesthetically pleasing. Default is None.

    Returns:
    --------
    tuple
        Figure dimensions as (width, height) in inches, ready to use with plt.figure(figsize=...).

    Notes:
    ------
    To calculate a matching font size for your LaTeX document, you can use the following formula:

    ```python
    # For typical 11pt LaTeX document:
    latex_font_pt = 11
    # Convert LaTeX pt to matplotlib pt (1.0 LaTeX pt ≈ 1.13636 matplotlib pt):
    mpl_font_size = latex_font_pt * 1.13636
    ```

    Examples:
    ---------
    # Figure with default settings (full text width, golden ratio)
    plt.figure(figsize=set_size())

    # Figure with half text width
    plt.figure(figsize=set_size(fraction=0.5))

    # Figure with 2x2 subplots
    plt.figure(figsize=set_size(subplots=(2, 2)))

    # Figure with custom height ratio
    plt.figure(figsize=set_size(height_ratio=0.75))
    """
    # Calculate the figure width in points
    fig_width_pt = width_pt * fraction

    # Convert from points to inches (standard conversion)
    inches_per_pt = 1 / 72.27  # There are 72.27 points in an inch

    # If no height ratio provided, use the golden ratio
    if height_ratio is None:
        height_ratio = (5**0.5 - 1) / 2  # Golden ratio ≈ 0.618

    # Calculate width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    # Calculate height in inches, adjusted for subplot configuration
    # For more rows than columns, this makes the figure taller
    fig_height_in = fig_width_in * height_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def format_sci_notation(
    x: float, decimals: int = 1, include_dollar: bool = True
) -> str:
    """
    Format a number in scientific notation using LaTeX formatting.

    This function converts a number to scientific notation (e.g., 1.2 × 10^3 instead of 1200)
    with LaTeX formatting suitable for use in matplotlib figures. The formatting follows
    scientific conventions and can be customized to include or exclude LaTeX math delimiters.

    Parameters:
    -----------
    x : float
        The number to format in scientific notation.

    decimals : int, optional
        Number of decimal places to show in the coefficient part. Default is 1.

    include_dollar : bool, optional
        Whether to wrap the result in LaTeX math delimiters ($...$).
        Set to False when using within an environment that already
        processes math mode. Default is True.

    Returns:
    --------
    str
        LaTeX-formatted string representing the number in scientific notation.

    Examples:
    ---------
    >>> format_sci_notation(1234)
    '$1.2 \\times 10^{3}$'

    >>> format_sci_notation(0.00456, decimals=2)
    '$4.56 \\times 10^{-3}$'

    >>> format_sci_notation(1000, include_dollar=False)
    '10^{3}'  # Note: Simplifies when coefficient is 1

    >>> format_sci_notation(0)
    '$0$'
    """
    # Special case: handle zero separately since log10(0) is undefined
    if x == 0:
        return r"$0$" if include_dollar else r"0"

    # Calculate the exponent (power of 10)
    exp = int(np.floor(np.log10(abs(x))))

    # Calculate the coefficient and round to specified decimal places
    coef = round(x / 10**exp, decimals)

    # Choose multiplication symbol based on LaTeX availability
    mult_symbol = r" \times " if latex_available else r" \cdot "

    # Format the result, simplifying when coefficient is 1
    if coef == 1:
        result = r"10^{" + str(exp) + r"}"
    else:
        result = str(coef) + mult_symbol + r"10^{" + str(exp) + r"}"

    # Add LaTeX math delimiters if requested
    return r"$" + result + r"$" if include_dollar else result


def _calculate_matching_font_size(latex_font_pt: float = 11) -> float:
    """
    Calculate a matplotlib font size that matches a LaTeX document font size.

    In LaTeX documents, fonts are typically specified in points (pt). However,
    matplotlib's point size doesn't exactly match LaTeX's point size. This function
    converts from LaTeX points to matplotlib points to ensure consistent text
    appearance between your document and figures.

    Parameters:
    -----------
    latex_font_pt : float, optional
        The font size in points used in your LaTeX document.
        Common values: 10pt, 11pt, 12pt. Default is 11pt.

    Returns:
    --------
    float
        The equivalent font size to use in matplotlib settings.

    Notes:
    ------
    The conversion factor (1.13636) accounts for differences between LaTeX and
    matplotlib's point size definitions. This helps ensure that when your figure
    is included in the LaTeX document, the text in the figure will appear visually
    similar in size to the document text.

    Examples:
    ---------
    # For a LaTeX document using 12pt font:
    plt.rcParams.update({'font.size': _calculate_matching_font_size(12)})
    """
    # Conversion factor: 1 LaTeX point ≈ 1.13636 matplotlib points
    mpl_to_latex_ratio = 1.13636

    # Calculate the equivalent matplotlib font size
    mpl_font_size = latex_font_pt * mpl_to_latex_ratio

    return mpl_font_size


# SYSTEM CHECKS FOR FAILSAFE
def _check_latex_available():
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


def _get_available_fonts(keywords):
    """Return a list of available fonts that match given keywords."""
    available = set()
    for f in fm.fontManager.ttflist:
        for kw in keywords:
            if kw in f.name.lower():
                available.add(f.name)
    return sorted(available)


def _set_best_serif_font():
    preferred_order = [
        "palatino linotype",
        "palatino",
        "cmu serif",  # latex imitate
        "times new roman",
    ]
    available = _get_available_fonts(preferred_order)
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


# BACKEND SELECTION - AUTOMATIC BASED ON ENVIRONMENT
def _setup_backend():
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


# =============================
# PLOTTING SETTINGS
# =============================
DEFAULT_FIG_PATH = FIGURES_DIR / "tests"
DEFAULT_FIG_FORMAT = "svg"  # pdf, png, svg
DEFAULT_DPI = 100  # 100 is very high, 10 is good for notebooks
# Default font size calculation based on standard 11pt LaTeX document
# Uncomment and adjust as needed:
DEFAULT_FONT_SIZE = _calculate_matching_font_size(LATEX_FONT_SIZE)  # For 11pt LaTeX
DEFAULT_FIGSIZE = (
    8,
    6,
)  # set_size(width_pt=LATEX_DOC_WIDTH, fraction=0.5, subplots=(1, 1))
DEFAULT_TRANSPARENCY = True  # True for transparent background, False for white
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


# Initialize font_to_use
font_to_use = "serif"  # fall back
_set_best_serif_font()  # update font to use
latex_available = _check_latex_available()

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
    "axes.titlesize": DEFAULT_FONT_SIZE + 2,
    "axes.labelsize": DEFAULT_FONT_SIZE + 2,
    "xtick.labelsize": DEFAULT_FONT_SIZE,
    "ytick.labelsize": DEFAULT_FONT_SIZE,
    "legend.fontsize": DEFAULT_FONT_SIZE,
    # layout
    "figure.figsize": DEFAULT_FIGSIZE,
    "figure.autolayout": True,
    "axes.grid": False,
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.framealpha": 0.8,
    "savefig.bbox": "tight",
    "savefig.transparent": DEFAULT_TRANSPARENCY,
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

_setup_backend()


def save_fig(
    fig: plt.Figure,
    filename: str,
    formats: list = [DEFAULT_FIG_FORMAT],
    dpi: int = DEFAULT_DPI,
    transparent: bool = DEFAULT_TRANSPARENCY,
    figsize: tuple = None,
) -> None:
    """
    Save a matplotlib figure in multiple formats.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Full path including directory without extension
    formats : list, optional
        List of file formats to save
    dpi : int, optional
        Resolution for raster formats
    transparent : bool, optional
        Whether to save with transparent background
    figsize : tuple, optional
        Figure dimensions (width, height) in inches. If provided, the figure size
        will be updated before saving.
    """
    if figsize is not None:
        fig.set_size_inches(figsize)

    # Save in all requested formats
    for fmt in formats:
        fig.savefig(
            f"{filename}.{fmt}",
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )

    print(
        f"Figure saved as: {filename} ({', '.join(formats)})",
        flush=True,
    )


__all__ = [
    # constants
    "LATEX_DOC_WIDTH",
    "LATEX_FONT_SIZE",
    "DEFAULT_FIGSIZE",
    "DEFAULT_DPI",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_FIG_FORMAT",
    "DEFAULT_TRANSPARENCY",
    "COLORS",
    "LINE_STYLES",
    "MARKERS",
    # settings
    "latex_available",
    "font_to_use",
    # functions
    "set_size",
    "format_sci_notation",
    "save_fig",
]


def main():
    print(f"📊 Matplotlib settings loaded")
    print(
        f"   - LaTeX rendering: {'Enabled' if latex_available else 'Disabled (fallback to mathtext)'}"
    )
    print(f"   - Font: {font_to_use}")
    print(f"   - Backend: {mpl.get_backend()}")
    print(f"   - Default figure size: {DEFAULT_FIGSIZE}")


if __name__ == "__main__":
    main()
