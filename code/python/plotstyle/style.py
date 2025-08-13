"""Matplotlib styling (standalone implementation).

Import from ``plotstyle``::

        from plotstyle import init_style, save_fig, set_size

Design goals:
        * Idempotent style initialization (safe in parallel workers)
        * Zero side effects until ``init_style()`` is called
        * Minimal, readable set of defaults suitable for thesis figures
"""

from __future__ import annotations
import os, sys
from typing import Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------
# Public constants / defaults
# -----------------------------
LATEX_DOC_WIDTH = 441.01775  # pt width of LaTeX document text block
LATEX_FONT_SIZE = 11
DPI = 100
FIG_FORMAT = "svg"
TRANSPARENCY = True
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
latex_available = False  # placeholder; extend later if LaTeX detection desired
_INITIALIZED = False


def _calculate_matching_font_size(latex_font_pt: float = LATEX_FONT_SIZE) -> float:
    """Return matplotlib 'font.size' approximating LaTeX appearance."""
    return latex_font_pt * 1.13636


FONT_SIZE = _calculate_matching_font_size(LATEX_FONT_SIZE)


def set_size(
    width_pt: float = LATEX_DOC_WIDTH,
    fraction: float = 0.5,
    subplots=(1, 1),
    height_ratio: Optional[float] = None,
):
    """Compute (width, height) in inches for a figure scaled to LaTeX width.

    Parameters
    ----------
    width_pt : float
            Full width of the LaTeX text block in points.
    fraction : float
            Fraction of the width to occupy (0 < fraction <= 1).
    subplots : tuple
            (n_rows, n_cols) to scale height for multi-panel figures.
    height_ratio : float | None
            Optional manual golden-ratio-like modification; default uses golden ratio.
    """
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    if height_ratio is None:
        height_ratio = (5**0.5 - 1) / 2  # golden ratio approximation
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * height_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def format_sci_notation(
    x: float, decimals: int = 1, include_dollar: bool = True
) -> str:
    """Format a number into scientific notation suitable for LaTeX labels."""
    if x == 0:
        return r"$0$" if include_dollar else "0"
    exp = int(np.floor(np.log10(abs(x))))
    coef = round(x / 10**exp, decimals)
    mult_symbol = r" \times " if latex_available else r" \cdot "
    if coef == 1:
        result = f"10^{{{exp}}}"
    else:
        result = f"{coef}{mult_symbol}10^{{{exp}}}"
    return f"${result}$" if include_dollar else result


def _setup_backend():
    """Select a safe backend (inline in notebooks, Agg in headless, else TkAgg)."""
    try:
        current = mpl.get_backend().lower()
        if "ipykernel" in sys.modules:  # Jupyter / IPython inline context
            inline_name = "module://matplotlib_inline.backend_inline"
            if current != inline_name:
                mpl.use(inline_name)
            return
        headless = ("DISPLAY" not in os.environ) or bool(os.environ.get("SLURM_JOB_ID"))
        if headless:
            if current != "agg":
                mpl.use("Agg")
            return
        if not any(k in current for k in ("tk", "qt", "wx", "macosx")):
            try:
                mpl.use("TkAgg")
            except Exception:
                mpl.use("Agg")
    except Exception:  # fall back robustly
        mpl.use("Agg")


def init_style(force: bool = False, quiet: bool = True):
    """Initialize matplotlib rcParams once (idempotent) for thesis-quality plots."""
    global _INITIALIZED
    if _INITIALIZED and not force:
        return
    base_settings = {
        "font.family": "serif",
        "font.serif": ["cmu serif", "times new roman", "serif"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 2,
        "axes.labelsize": FONT_SIZE + 2,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.figsize": (8, 6),
        "figure.autolayout": True,
        "axes.grid": False,
        "axes.axisbelow": True,
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.8,
        "savefig.bbox": "tight",
        "savefig.transparent": TRANSPARENCY,
        "savefig.format": FIG_FORMAT,
        "savefig.dpi": DPI,
        "text.usetex": False,
        "mathtext.default": "regular",
    }
    plt.rcParams.update(base_settings)
    _setup_backend()
    _INITIALIZED = True
    if not quiet:
        print("[plotstyle.init_style] matplotlib style initialized")


def save_fig(
    fig: mpl.figure.Figure,
    filename: str,
    formats=None,
    dpi: int = DPI,
    transparent: bool = TRANSPARENCY,
    figsize=None,
):
    """Save figure to one or multiple formats, creating directories as needed."""
    if formats is None:
        formats = [FIG_FORMAT]
    import os as _os

    directory = _os.path.dirname(filename)
    if directory:
        _os.makedirs(directory, exist_ok=True)
    if figsize is not None:
        fig.set_size_inches(figsize)
    for fmt in formats:
        fig.savefig(
            f"{filename}.{fmt}",
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )
    if mpl.is_interactive():
        plt.draw()
        plt.pause(0.001)
        plt.close(fig)
    else:
        plt.close(fig)


__all__ = [
    "init_style",
    "save_fig",
    "set_size",
    "format_sci_notation",
    "COLORS",
    "LINE_STYLES",
    "MARKERS",
    "latex_available",
]
