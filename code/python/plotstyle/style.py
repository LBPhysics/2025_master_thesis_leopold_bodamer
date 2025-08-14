"""Matplotlib styling (standalone implementation).

Import from ``plotstyle``::

        from plotstyle import init_style, save_fig, set_size

Design goals:
        * Idempotent style initialization (safe in parallel workers)
        * Zero side effects until ``init_style()`` is called
        * Minimal, readable set of defaults suitable for thesis figures
"""

from __future__ import annotations
import os, sys, subprocess, shutil
from typing import Optional, Iterable
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------
# Public constants / defaults
# -----------------------------
LATEX_DOC_WIDTH = 441.01775  # pt width of LaTeX document text block
LATEX_FONT_SIZE = 11
DPI = 100
FIG_FORMAT = "png"  # NOTE this can be changed to svg for later
TRANSPARENCY = True

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
LINE_STYLES = ["solid", "dashed", "dashdot", "dotted", (0, (3, 1, 1, 1)), (0, (5, 1))]
MARKERS = ["o", "s", "^", "v", "D", "p", "*", "X", "+", "x"]
latex_available = True  # Will be detected on first use
_LATEX_PROBE_DONE = False
FONT_SIZE = 11
FIG_SIZE = (8, 6)


def _calculate_matching_font_size(latex_font_pt: float = LATEX_FONT_SIZE) -> float:
    """Return matplotlib 'font.size' approximating LaTeX appearance."""
    return latex_font_pt * 1.13636


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


def init_style(quiet: bool = True):
    """Initialize matplotlib rcParams once (idempotent) for thesis-quality plots."""
    base_settings = {
        "font.family": "serif",
        "font.serif": ["cmu serif", "times new roman", "serif"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 2,
        "axes.labelsize": FONT_SIZE + 2,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.figsize": FIG_SIZE,
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
        "mathtext.default": "regular",
    }

    _ensure_latex_probe()
    if latex_available:
        latex_settings = {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{mathpazo}",
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
    if not quiet:
        print("[plotstyle.init_style] matplotlib style initialized")
        print("latex", latex_available, "backend", mpl.get_backend())


def _check_latex_available():
    """
    Check if LaTeX (pdflatex or latex) is installed and available in the system path.

    Returns
    -------
    bool
        True if LaTeX is available, False otherwise.
    """
    latex_commands = ["pdflatex", "latex", "latexmk"]

    for cmd in latex_commands:
        # Quick check if command exists in PATH
        if shutil.which(cmd) is not None:
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=2,
                )
                # If it ran without FileNotFoundError, LaTeX is present
                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

    return False


def _ensure_latex_probe() -> None:
    """Probe for LaTeX availability once and cache the result."""
    global latex_available, _LATEX_PROBE_DONE
    if _LATEX_PROBE_DONE:
        return
    latex_available = _check_latex_available()
    _LATEX_PROBE_DONE = True


def _strip_latex(s: str) -> str:
    """Convert a LaTeX/mathtext label to a plain-text safe fallback.

    Notes
    -----
    This is a best-effort sanitizer aiming to avoid LaTeX-specific commands that
    would error in non-TeX environments. It preserves basic intent (subscripts
    via underscores, powers via caret) and removes formatting commands.
    """
    if not s:
        return s
    # Remove math delimiters
    out = s.replace("$", "")
    # Common spacing/formatting
    for tok in ("\,", "\ ", "\;", "\:", "\!", "\quad", "\qquad"):
        out = out.replace(tok, " ")
    for tok in ("\left", "\right"):
        out = out.replace(tok, "")
    # Text/roman wrappers
    import re as _re

    def _unbrace(cmd: str, text: str) -> str:
        # Match LaTeX-like wrappers such as \text{...}, \mathrm{...}, etc.,
        # without using f-strings to avoid brace-escaping issues.
        pattern = _re.compile(r"\\" + cmd + r"\{([^}]*)\}")
        return _re.sub(pattern, r"\1", text)

    for cmd in ("text", "mathrm", "mathbf", "mathit", "mathcal", "mathbb", "mathfrak"):
        out = _unbrace(cmd, out)
    # Greek symbols and a few common macros
    replacements = {
        r"\omega": "omega",
        r"\Omega": "Omega",
        r"\phi": "phi",
        r"\varphi": "phi",
        r"\theta": "theta",
        r"\Theta": "Theta",
        r"\mu": "mu",
        r"\varepsilon": "eps",
        r"\epsilon": "eps",
        r"\pi": "pi",
        r"\cdot": "*",
        r"\times": "x",
        r"\propto": "~",
        r"\infty": "inf",
        r"\langle": "<",
        r"\rangle": ">",
        r"\vec": "",  # drop arrow accent
        r"\hat": "",  # drop hat accent
        r"\bar": "",  # drop bar accent
    }
    for k, v in replacements.items():
        out = out.replace(k, v)
    # Remove braces but keep structure like E_{out} -> E_out; 10^{4} -> 10^4
    out = out.replace("{", "").replace("}", "")
    # Collapse multiple spaces
    out = _re.sub(r"\s+", " ", out).strip()
    return out


def simplify_figure_text(fig: mpl.figure.Figure) -> mpl.figure.Figure:
    """Sanitize all text in a figure if LaTeX isn't available.

    Iterates over titles, axis labels, tick labels, and legend text to ensure
    strings won't trigger LaTeX rendering errors in environments without TeX.
    """
    _ensure_latex_probe()
    if latex_available:
        return fig

    def _sanitize_textobjs(objs: Iterable[mpl.text.Text]):
        for t in objs:
            try:
                s = t.get_text()
                if s:
                    t.set_text(_strip_latex(s))
            except Exception:
                continue

    try:
        # Suptitle
        if getattr(fig, "_suptitle", None) is not None:
            st = fig._suptitle
            st.set_text(_strip_latex(st.get_text()))
        # Axes content
        for ax in fig.get_axes():
            _sanitize_textobjs([ax.title, ax.xaxis.label, ax.yaxis.label])
            _sanitize_textobjs(ax.get_xticklabels())
            _sanitize_textobjs(ax.get_yticklabels())
            leg = ax.get_legend()
            if leg is not None:
                _sanitize_textobjs(leg.get_texts())
    except Exception:
        pass
    return fig


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
    # Always sanitize figure text for LaTeX fallback
    simplify_figure_text(fig)
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
    "_check_latex_available",
    "simplify_figure_text",
]
