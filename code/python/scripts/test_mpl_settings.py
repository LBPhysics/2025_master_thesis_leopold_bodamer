#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for mpl_tex_settings with failsafe mechanisms.

This script demonstrates how the settings handle both:
1. Missing LaTeX installation
2. Missing preferred font
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the config module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our settings module - this will apply all settings
from config.mpl_tex_settings import *

# =============================
# TEST PLOT FUNCTION
# =============================


def create_test_plot():
    """Create a test plot to demonstrate failsafe settings."""
    # Create data
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

    # Create figure with custom size
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # First subplot - math expressions
    ax1 = axes[0]
    ax1.plot(x, np.sin(x), color="C0", linestyle="solid", label=r"$\sin(x)$")
    ax1.plot(x, np.cos(x), color="C1", linestyle="dashed", label=r"$\cos(x)$")
    ax1.set_title(
        r"Math Expressions (mode: "
        + ("LaTeX" if latex_available else "Mathtext")
        + r")"
    )
    ax1.set_xlabel(r"$x$ [rad]")
    ax1.set_ylabel(r"$f(x)$")
    ax1.legend()

    # Second subplot - scientific notation
    ax2 = axes[1]
    y = np.exp(x / 3) * 1e-5
    ax2.semilogy(x, y, color="C2", linestyle="solid")
    ax2.set_title(f"Scientific notation with {font_to_use} font")
    ax2.set_xlabel(r"$x$ [rad]")

    # Use include_dollar=False parameter for embedding in a larger expression
    sci_notation = format_sci_notation(1e-5, include_dollar=False)
    ax2.set_ylabel(rf"$y = {sci_notation} \cdot e^{{x/3}}$")

    # Format ticks with scientific notation
    yticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    ax2.set_yticks(yticks)
    # For tick labels, we want the dollar signs included
    ax2.set_yticklabels([format_sci_notation(y, include_dollar=True) for y in yticks])

    # Adjust layout
    plt.tight_layout()

    # Display information
    print("\nTest plot created:")
    print(f"- Font: {font_to_use}")
    print(f"- LaTeX: {'Enabled' if latex_available else 'Disabled (using mathtext)'}")

    return fig


if __name__ == "__main__":
    # Create and show the test plot
    fig = create_test_plot()

    # Save the plot
    save_fig(fig, "mpl_settings_test", category="tests")

    # Display the plot
    plt.show()
