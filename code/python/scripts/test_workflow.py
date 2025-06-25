#!/usr/bin/env python3
"""
Comprehensive Test Workflow for Unified Electronic Spectroscopy Structure

This script tests the new unified structure implementation:
1. Tests 1D and 2D simulation workflows
2. Verifies standardized data structure
3. Tests feed-forward capability (calc -> plot)
4. Validates compressed storage (.pkl.gz)
5. Checks backward compatibility
"""

import sys
from pathlib import Path

# Add the necessary directories to Python path
scripts_dir = Path(__file__).parent
python_dir = scripts_dir.parent  # Go up to the python directory
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(python_dir))

# Test imports with better error handling
try:
    from common_fcts import (
        run_1d_simulation_with_config,
        run_2d_simulation_with_config,
        plot_1d_from_relative_path,
        plot_2d_from_relative_path,
        load_pickle_file,
    )

    print("✅ Successfully imported common_fcts functions")
except ImportError as e:
    print(f"❌ Failed to import common_fcts: {e}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Python sys.path includes: {sys.path[:3]}...")
    sys.exit(1)

from config.paths import DATA_DIR, FIGURES_DIR


def test_data_structure_validity(relative_path: str, expected_type: str):
    """Test if the saved data follows the standardized structure."""
    print(f"\n🔍 Testing data structure for {expected_type}...")

    full_path = DATA_DIR / relative_path
    if not full_path.exists():
        print(f"❌ File doesn't exist: {full_path}")
        return False

    # Load and check data structure
    data = load_pickle_file(full_path)
    if data is None:
        print(f"❌ Failed to load data from {full_path}")
        return False

    # Check required keys
    required_keys = ["data", "axes", "system", "config", "metadata"]
    for key in required_keys:
        if key not in data:
            print(f"❌ Missing required key: {key}")
            return False

    # Check axes structure
    axes = data["axes"]
    required_axes = ["t_det", "tau_coh", "T_wait"]
    for axis in required_axes:
        if axis not in axes:
            print(f"❌ Missing axis: {axis}")
            return False

    # Check metadata
    metadata = data["metadata"]
    if metadata.get("simulation_type") != expected_type:
        print(
            f"❌ Wrong simulation type: expected {expected_type}, got {metadata.get('simulation_type')}"
        )
        return False

    print(f"✅ Data structure is valid for {expected_type}!")
    print(
        f"   Data shape: {data['data'].shape if hasattr(data['data'], 'shape') else len(data['data'])}"
    )
    print(f"   File format: {'.pkl.gz' if str(full_path).endswith('.gz') else '.pkl'}")
    print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")

    return True


def test_1d_workflow():
    """Test the complete 1D workflow."""
    print("\n" + "=" * 60)
    print("🧪 TESTING 1D WORKFLOW")
    print("=" * 60)

    # Configuration for quick test
    config = {
        "N_atoms": 1,
        "tau_coh": 300.0,  # Short for quick test
        "T_wait": 100.0,  # Short for quick test
        "t_det_max": 60.0,  # Short for quick test
        "dt": 2.0,  # Larger timestep for speed
        "ODE_Solver": "BR",
        "pulse_fwhm": 15.0,
        "RWA_laser": False,
        "n_phases": 2,  # Reduced for speed
        "n_freqs": 1,
        "Delta_cm": 0,
        "envelope_type": "gaussian",
        "E0": 0.005,
        "output_subdir": "test_workflow/1d_quick_test",
    }

    # Run simulation
    print("\n📊 Step 1: Running 1D calculation...")
    try:
        relative_path = run_1d_simulation_with_config(config)
        print(f"✅ 1D Calculation completed! Data saved at: {relative_path}")
    except Exception as e:
        print(f"❌ 1D Calculation failed: {e}")
        return False

    # Test data structure
    if not test_data_structure_validity(relative_path, "1d"):
        return False

    # Test plotting
    print(f"\n🎨 Step 2: Plotting 1D data from relative path...")
    plot_config = {
        "spectral_components_to_plot": ["real", "imag"],
        "plot_time_domain": True,
        "plot_frequency_domain": True,
    }

    try:
        plot_1d_from_relative_path(relative_path, plot_config)
        print("✅ 1D Plotting completed!")
        return True
    except Exception as e:
        print(f"❌ 1D Plotting failed: {e}")
        return False


def test_2d_workflow():
    """Test the complete 2D workflow."""
    print("\n" + "=" * 60)
    print("🧪 TESTING 2D WORKFLOW")
    print("=" * 60)

    # Configuration for quick test
    config = {
        "N_atoms": 1,
        "t_max": 10,  # Very short for quick test
        "dt": 2,  # Large timestep for speed
        "ODE_Solver": "BR",
        "pulse_fwhm": 15.0,
        "RWA_laser": False,
        "T_wait_max": 2,  # Very short for quick test
        "n_times_T": 1,
        "n_phases": 2,  # Reduced for speed
        "n_freqs": 1,
        "Delta_cm": 0,
        "envelope_type": "gaussian",
        "E0": 0.005,
        "output_subdir": "test_workflow/2d_quick_test",
    }

    # Run simulation
    print("\n📊 Step 1: Running 2D calculation...")
    try:
        relative_path = run_2d_simulation_with_config(config)
        print(f"✅ 2D Calculation completed! Data saved at: {relative_path}")
    except Exception as e:
        print(f"❌ 2D Calculation failed: {e}")
        return False

    # Test data structure
    if not test_data_structure_validity(relative_path, "2d"):
        return False

    # Test plotting
    print(f"\n🎨 Step 2: Plotting 2D data from relative path...")
    plot_config = {
        "spectral_components_to_plot": ["real", "imag"],
        "plot_time_domain": True,
        "extend_for": (1, 6),
        "section": (0, 2, 0, 2),
        # "section": (1.4, 1.8, 1.4, 1.8),
    }

    try:
        plot_2d_from_relative_path(relative_path, plot_config)
        print("✅ 2D Plotting completed!")
        return True
    except Exception as e:
        print(f"❌ 2D Plotting failed: {e}")
        return False


def test_file_compression():
    """Test that new files are properly compressed."""
    print("\n" + "=" * 60)
    print("🗜️  TESTING FILE COMPRESSION")
    print("=" * 60)

    # Look for recently created test files
    test_dirs = [
        DATA_DIR / "1d_spectroscopy" / "test_workflow",
        DATA_DIR / "2d_spectroscopy" / "test_workflow",
    ]

    compressed_files = []
    uncompressed_files = []

    for test_dir in test_dirs:
        if test_dir.exists():
            compressed_files.extend(list(test_dir.glob("**/*.pkl.gz")))
            uncompressed_files.extend(list(test_dir.glob("**/*.pkl")))

    print(f"Found {len(compressed_files)} compressed files (.pkl.gz)")
    print(f"Found {len(uncompressed_files)} uncompressed files (.pkl)")

    if len(compressed_files) > 0:
        print("✅ New files are being compressed!")
        for f in compressed_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   📁 {f.name} ({size_mb:.2f} MB)")
    else:
        print("❌ No compressed files found - compression might not be working")

    return len(compressed_files) > 0


def main():
    """Run all workflow tests."""
    print("🧪 COMPREHENSIVE WORKFLOW TESTING")
    print("=" * 60)
    print("Testing the new unified electronic spectroscopy structure...")

    results = {
        "1d_workflow": False,
        "2d_workflow": False,
        "compression": False,
    }

    """
    # Test 1D workflow
    try:
        results["1d_workflow"] = test_1d_workflow()
    except Exception as e:
        print(f"❌ 1D workflow test crashed: {e}")
    """

    # Test 2D workflow
    try:
        results["2d_workflow"] = test_2d_workflow()
    except Exception as e:
        print(f"❌ 2D workflow test crashed: {e}")

    # Test compression
    try:
        results["compression"] = test_file_compression()
    except Exception as e:
        print(f"❌ Compression test crashed: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The unified structure is working correctly!")
        print(f"   Data saved in: {DATA_DIR}")
        print(f"   Figures saved in: {FIGURES_DIR}")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

    return all_passed


if __name__ == "__main__":
    main()
