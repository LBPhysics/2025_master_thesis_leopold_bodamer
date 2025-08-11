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
from typing import Optional, Callable, Union
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys
import shutil
import subprocess
from qspectro2d.config.paths import FIGURES_PYTHON_DIR, FIGURES_TESTS_DIR

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
FIG_PATH = FIGURES_TESTS_DIR
FIG_FORMAT = "svg"  # pdf, png, svg
DPI = 100  # 100 is very high, 10 is good for notebooks
# Default font size calculation based on standard 11pt LaTeX document
# Uncomment and adjust as needed:
FONT_SIZE = _calculate_matching_font_size(LATEX_FONT_SIZE)  # For 11pt LaTeX
FIGSIZE = (
    8,
    6,
)  # set_size(width_pt=LATEX_DOC_WIDTH, fraction=0.5, subplots=(1, 1))
TRANSPARENCY = True  # True for transparent background, False for white
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

# AUTOMATIC LaTeX DETECTION (no env opt-in required)
latex_available = False  # TODO _check_latex_available()

# =============================
# MATPLOTLIB LaTeX SETTINGS
# =============================
base_settings = {
    # font
    "font.family": "serif",
    "font.serif": [
        "cmu serif",
        "times new roman",
        "serif",
    ],
    # font sizes
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    # layout
    "figure.figsize": FIGSIZE,
    "figure.autolayout": True,
    "axes.grid": False,
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.framealpha": 0.8,
    "savefig.bbox": "tight",
    "savefig.transparent": TRANSPARENCY,
    # Quality of the plot:
    "savefig.format": FIG_FORMAT,
    "savefig.dpi": DPI,
}

if latex_available:
    latex_settings = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{mathpazo}",
    }
    base_settings.update(latex_settings)
else:
    base_settings.update({"text.usetex": False, "mathtext.default": "regular"})

plt.rcParams.update(base_settings)
if latex_available:
    print("LaTeX: enabled (auto-detected)")
else:
    print("LaTeX: not available -> using mathtext")

_setup_backend()

# =============================
# LATEX FALLBACK HELPERS (shared with plotting)
# =============================
import re as _re

_LATEX_REGEX_REPLACEMENTS: list[
    tuple[_re.Pattern, Union[str, Callable[["_re.Match"], str]]]
] = [
    (_re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
    (_re.compile(r"\\text\{([^}]*)\}"), r"\1"),
    (_re.compile(r"\\left"), ""),
    (_re.compile(r"\\right"), ""),
    (
        _re.compile(r"\\(langle|rangle)"),
        lambda m: "<" if m.group(1) == "langle" else ">",
    ),
    (_re.compile(r"\\omega"), "w"),
    (_re.compile(r"\\mu"), "mu"),
    (_re.compile(r"\\Delta"), "Δ"),
    (_re.compile(r"\\propto"), "∝"),
]


def strip_latex(text: str) -> str:
    """Convert a LaTeX/math string into a plain-text approximation.

    Removes math delimiters and common formatting commands while keeping
    semantic meaning where possible.
    """
    if not text:
        return text
    out = text.replace("$", "")
    for pat, repl in _LATEX_REGEX_REPLACEMENTS:
        out = pat.sub(repl, out)
    out = _re.sub(r"\s+", " ", out).strip()
    return out


def maybe_latex(label_latex: str, label_plain: str | None = None) -> str:
    """Return LaTeX label if LaTeX is available else a plain fallback.

    If label_plain is omitted it's inferred via strip_latex.
    """
    if latex_available:
        return label_latex
    return label_plain if label_plain is not None else strip_latex(label_latex)


def simplify_figure_text(fig, force: bool = False) -> None:
    """Downgrade all axis titles/labels/legend texts to plain text if LaTeX unavailable.

    Set force=True to apply even when LaTeX is available (debug/testing).
    """
    if latex_available and not force:
        return
    try:
        for ax in fig.get_axes():
            ax.set_title(strip_latex(ax.get_title()))
            ax.set_xlabel(strip_latex(ax.get_xlabel()))
            ax.set_ylabel(strip_latex(ax.get_ylabel()))
            leg = ax.get_legend()
            if leg:
                for txt in leg.get_texts():
                    txt.set_text(strip_latex(txt.get_text()))
    except Exception as e:  # non-fatal
        print(f"Warning (simplify_figure_text): {e}")


def save_fig(
    fig: plt.Figure,
    filename: str,
    formats: list = [FIG_FORMAT],
    dpi: int = DPI,
    transparent: bool = TRANSPARENCY,
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
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory:  # Only create if directory is not empty string
        os.makedirs(directory, exist_ok=True)

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

    # Print the absolute path
    try:
        abs_path = os.path.abspath(filename)
    except Exception:
        abs_path = filename
    print(
        f"Figure saved as: {abs_path}.{', '.join(formats)}",
        flush=True,
    )

    # Display figure if running in an interactive GUI environment
    if mpl.is_interactive():
        plt.draw()
        plt.pause(0.001)
        plt.close(fig)
    else:
        backend = mpl.get_backend().lower()
        if not any(x in backend for x in ("agg", "pdf", "svg")):
            try:
                plt.show(block=True)
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not display figure interactively: {e}")
                plt.close(fig)
        else:
            # Non-interactive/headless: just close
            plt.close(fig)


__all__ = [
    # constants
    "LATEX_DOC_WIDTH",
    "LATEX_FONT_SIZE",
    "FIGSIZE",
    "DPI",
    "FONT_SIZE",
    "FIG_FORMAT",
    "TRANSPARENCY",
    "COLORS",
    "LINE_STYLES",
    "MARKERS",
    # settings
    "latex_available",
    # functions
    "set_size",
    "format_sci_notation",
    "save_fig",
    # helpers
    "strip_latex",
    "maybe_latex",
    "simplify_figure_text",
]

"""
from matplotlib import font_manager
# List all available fonts
available_fonts = [f.name for f in font_manager.fontManager.ttflist]
print("Available fonts:")
print("\n".join(sorted(set(available_fonts))))
"""


def main():
    try:
        print(f"📊 Matplotlib settings loaded")
        print(
            f"   - LaTeX rendering: {'Enabled' if latex_available else 'Disabled (fallback to mathtext)'}"
        )
        print(f"   - Backend: {mpl.get_backend()}")
        print(f"   - Default figure size: {FIGSIZE}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
