"""
1D Electronic Spectroscopy Data Plotting Script

This script loads and plots 1D electronic spectroscopy data from numpy/pickle files
in various formats (real, imaginary, absolute, phase) for analysis and visualization.

Usage modes:
1. Direct file paths (feedforward from calculation script):
   python plot_1D_datas.py --data-path path/to/data.npz --info-path path/to/info.pkl

2. Relative directory search:
   python plot_1D_datas.py --relative-dir 1d_spectroscopy/N1_atoms/Paper_eqs/RWA

3. Auto-search mode (no arguments):
   python plot_1D_datas.py
"""

import sys
import argparse
from pathlib import Path
from common_fcts import load_latest_data, load_data_from_paths, plot_1d_data
from config.paths import DATA_DIR, FIGURES_1D_DIR


# =============================
# ARGUMENT PARSING
# =============================
def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Plot 1D electronic spectroscopy data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct file paths (feedforward from calculation script)
  python plot_1D_datas.py --data-path data/1d_spectroscopy/data.npz --info-path data/1d_spectroscopy/info.pkl
  
  # Relative directory search
  python plot_1D_datas.py --relative-dir 1d_spectroscopy/N1_atoms/Paper_eqs/RWA
  
  # Auto-search mode (finds latest data)
  python plot_1D_datas.py
        """,
    )

    # Mutually exclusive group for different input modes
    input_group = parser.add_mutually_exclusive_group()

    input_group.add_argument(
        "--data-path", type=Path, help="Direct path to the data file (.npz)"
    )

    parser.add_argument(
        "--info-path",
        type=Path,
        help="Direct path to the info file (.pkl) - required when using --data-path",
    )

    input_group.add_argument(
        "--relative-dir",
        type=Path,
        help="Relative directory path to search for latest data files",
    )

    # Optional plotting parameters
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Save plots to figures directory (default: True)",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        default=False,
        help="Display plots interactively (default: False)",
    )

    return parser


# =============================
# MAIN FUNCTION
# =============================
def main():
    """Main function to run the 1D spectroscopy plotting."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.data_path and not args.info_path:
        parser.error("--info-path is required when using --data-path")

    try:
        # =============================
        # LOAD DATA BASED ON INPUT MODE
        # =============================
        if args.data_path and args.info_path:
            ### Direct file paths mode (feedforward from calculation script)
            print(f"📁 Loading data from direct paths:")
            print(f"   Data: {args.data_path}")
            print(f"   Info: {args.info_path}")

            # Validate paths exist
            if not args.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {args.data_path}")
            if not args.info_path.exists():
                raise FileNotFoundError(f"Info file not found: {args.info_path}")

            data_dict = load_data_from_paths(args.data_path, args.info_path)

            # Debug: Print available keys
            print(f"🔍 Available data keys: {list(data_dict.keys())}")
            if "axes" in data_dict:
                print(f"🔍 Available axes keys: {list(data_dict['axes'].keys())}")

        elif args.relative_dir:
            ### Relative directory search mode
            print(f"📁 Searching for latest data in: {args.relative_dir}")
            data_dict = load_latest_data(args.relative_dir)

        else:
            ### Auto-search mode - find latest data in default directory
            print("🔍 Auto-search mode: Looking for latest 1D spectroscopy data...")
            default_search_dir = Path("1d_spectroscopy")
            data_dict = load_latest_data(
                default_search_dir
            )  # =============================
        # EXTRACT DATA FOR PLOTTING
        # =============================
        ### Extract standardized data structure
        t_det_vals = data_dict["axes"]["axs1"]  # Detection time axis
        E_field_data = data_dict["data"]  # Complex electric field data

        # Print data information
        print(f"✅ Data loaded successfully:")
        print(f"   Shape: {E_field_data.shape}")
        print(f"   Time points: {len(t_det_vals)}")
        print(f"   Time range: {t_det_vals[0]:.1f} to {t_det_vals[-1]:.1f} fs")

        # =============================
        # CONFIGURE PLOTTING
        # =============================
        ### Set up plotting configuration
        plot_config = {
            "save_plots": args.save_plots,
            "show_plots": args.show_plots,
            "formats": ["real", "imag", "abs", "phase"],  # Plot all formats
            "use_tex": True,  # Use LaTeX formatting
        }  # Determine output directory
        if args.relative_dir:
            output_dir = FIGURES_1D_DIR / args.relative_dir
        else:
            # Create output directory based on system parameters
            system = data_dict["system"]
            # Handle both dict and SystemParameters object formats
            if hasattr(system, "N_atoms"):
                N_atoms = system.N_atoms
            else:
                N_atoms = system.get("N_atoms", 1)  # Default to 1 if not found
            output_dir = FIGURES_1D_DIR / f"N{N_atoms}_atoms" / "latest_plots"

        output_dir.mkdir(parents=True, exist_ok=True)

        # =============================
        # GENERATE PLOTS
        # =============================
        print(f"📊 Generating 1D spectroscopy plots...")
        print(f"   Output directory: {output_dir}")

        plot_1d_data(
            ax1=t_det_vals,
            data=E_field_data,
            data_dict=data_dict,
            plot_config=plot_config,
            output_dir=output_dir,
        )

        print("✅ 1D spectroscopy plotting completed successfully!")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Missing required data key: {e}")
        print("   Check that the data file contains the expected structure")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
