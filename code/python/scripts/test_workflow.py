#!/usr/bin/env python3
"""
Example workflow demonstrating the feed-forward capability between 
calculation and plotting scripts.

This script shows how to:
1. Run a 2D calculation and get the relative path
2. Feed that path directly to the plotting script
"""

import subprocess
import sys
from pathlib import Path

# Add the scripts directory to Python path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from common_fcts import run_2d_simulation_with_config, plot_2d_from_relative_path


def test_workflow():
    """Test the complete workflow: calculate -> save -> plot."""
    
    print("🧪 TESTING FEED-FORWARD WORKFLOW")
    print("=" * 50)
    
    # =============================
    # STEP 1: RUN CALCULATION
    # =============================
    print("\n📊 Step 1: Running 2D calculation...")
    
    config = {
        "N_atoms": 1,
        "t_max": 4,
        "dt": 2,
        "ODE_Solver": "BR",
        "pulse_fwhm": 15.0,
        "RWA_laser": False,
        "T_wait_max": 2,
        "n_times_T": 1,
        "n_phases": 2,  # Reduced for quick test
        "n_freqs": 1,
        "Delta_cm": 0,
        "envelope_type": "gaussian",
        "E0": 0.005,
        "output_subdir": "test_workflow/quick_test",
    }
    
    try:
        relative_path = run_2d_simulation_with_config(config)
        print(f"✅ Calculation completed! Data saved at: {relative_path}")
    except Exception as e:
        print(f"❌ Calculation failed: {e}")
        return
    
    # =============================
    # STEP 2: PLOT FROM RELATIVE PATH
    # =============================
    print(f"\n🎨 Step 2: Plotting from relative path: {relative_path}")
    
    plot_config = {
        "spectral_components_to_plot": ["real", "imag"],  # Reduced for quick test
        "plot_time_domain": False,  # Skip time domain for speed
        "extend_for": (1, 2),
        "section": (1.4, 1.8, 1.4, 1.8),
    }
    
    try:
        plot_2d_from_relative_path(str(relative_path), plot_config)
        print("✅ Plotting completed!")
    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        return
    
    print("\n🎉 WORKFLOW TEST COMPLETED SUCCESSFULLY!")
    print(f"   Data: DATA_DIR/{relative_path}")
    print(f"   Figures: FIGURES_DIR/figures_from_python/{relative_path.parent}/")


if __name__ == "__main__":
    test_workflow()
