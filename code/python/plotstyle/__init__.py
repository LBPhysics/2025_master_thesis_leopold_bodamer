"""Standalone plotting style package (decoupled from qspectro2d).

Usage:
    from plotstyle import init_style, save_fig, set_size, COLORS
"""

from .style import (
    init_style,
    save_fig,
    set_size,
    format_sci_notation,
    COLORS,
    LINE_STYLES,
    MARKERS,
    latex_available,
)

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
