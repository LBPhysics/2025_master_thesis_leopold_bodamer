"""
File utilities for qspectro2d.

This module provides file and path related utility functions.
Currently this is just an alias to the data.files module to maintain
backwards compatibility and logical organization.
"""

# Import everything from data.files for backwards compatibility
from qspectro2d.data.files import (
    generate_unique_data_filename,
    generate_unique_plot_filename,
    _generate_base_filename,
    _generate_unique_filename,
    generate_base_sub_dir,
)

__all__ = [
    "generate_unique_data_filename",
    "generate_unique_plot_filename",
    "_generate_base_filename",
    "_generate_unique_filename",
    "generate_base_sub_dir",
]
