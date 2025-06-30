"""
Configuration package for the Master's thesis project.

This package contains all configuration files for the project.
"""

# Make modules importable
from .mpl_tex_settings import (
    # Constants
    LATEX_DOC_WIDTH,
    LATEX_FONT_SIZE,
    DEFAULT_FIGSIZE,
    DEFAULT_DPI,
    DEFAULT_FONT_SIZE,
    DEFAULT_FIG_FORMAT,
    DEFAULT_TRANSPARENCY,
    COLORS,
    LINE_STYLES,
    MARKERS,
    # Settings
    latex_available,
    # Functions
    set_size,
    format_sci_notation,
    save_fig,
)

from .paths import (
    # Paths
    DATA_DIR,
    FIGURES_DIR,
    FIGURES_PYTHON_DIR,
    SCRIPTS_DIR,
    FIGURES_BATH_DIR,
    FIGURES_PULSES_DIR,
    FIGURES_TESTS_DIR,
)

# Export all important symbols for import *
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
    # functions
    "set_size",
    "format_sci_notation",
    "save_fig",
    # paths
    "DATA_DIR",
    "FIGURES_DIR",
    "FIGURES_PYTHON_DIR",
    "SCRIPTS_DIR",
    "FIGURES_BATH_DIR",
    "FIGURES_PULSES_DIR",
    "FIGURES_TESTS_DIR",
]
