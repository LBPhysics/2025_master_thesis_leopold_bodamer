#!/usr/bin/env python3
"""
Test script to verify Agg backend functionality for HPC simulation.

This script tests whether matplotlib works correctly with the Agg backend
and saves plots as PNG files without attempting to display them.
"""

import matplotlib

print(f"🖥️  Matplotlib backend: {matplotlib.get_backend()}")

# Import our custom settings which should force Agg backend
try:
    from config.mpl_tex_settings import *

    print(f"✅ Custom matplotlib settings loaded")
    print(f"🖥️  Backend after import: {matplotlib.get_backend()}")
except ImportError as e:
    print(f"❌ Failed to import custom settings: {e}")
    matplotlib.use("Agg")  # Fallback

import matplotlib.pyplot as plt
import numpy as np
from config.paths import FIGURES_DIR
from config import mpl_tex_settings

# Create a simple test plot
print("📊 Creating test plot...")
fig, ax = plt.subplots(figsize=(8, 6))

# Generate some test data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Plot the data
ax.plot(x, y1, label=r"$\sin(x)$", linewidth=2)
ax.plot(x, y2, label=r"$\cos(x)$", linewidth=2)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title(r"Test Plot for Agg Backend")
ax.legend()
ax.grid(True, alpha=0.3)

# Test saving the plot
output_dir = FIGURES_DIR / "test_agg"
output_dir.mkdir(parents=True, exist_ok=True)

save_path = output_dir / "test_agg_backend.png"

try:
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Plot saved successfully to: {save_path}")

    # Check if file was actually created
    if save_path.exists():
        file_size = save_path.stat().st_size
        print(f"📁 File size: {file_size / 1024:.1f} KB")
    else:
        print("❌ File was not created!")

except Exception as e:
    print(f"❌ Error saving plot: {e}")

# Test plt.show() - should not display anything with Agg backend
print("🔄 Testing plt.show() (should not display anything with Agg backend)...")
try:
    plt.show()
    print("✅ plt.show() executed without errors (no display expected)")
except Exception as e:
    print(f"❌ Error with plt.show(): {e}")

# Close the figure to free memory
plt.close(fig)

print("🎯 Agg backend test completed!")
print("📊 Check the saved PNG file to verify the plot was generated correctly.")
