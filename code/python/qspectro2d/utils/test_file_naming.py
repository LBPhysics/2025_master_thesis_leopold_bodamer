from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from qspectro2d.utils.file_naming import generate_unique_data_filename
from qspectro2d.core.simulation import SimulationConfig


def test_basic_functionality():
    """Test basic functionality with real temporary directory."""
    # Create temporary test directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Mock system object
        mock_system = MagicMock()
        mock_system.n_atoms = 1
        mock_system.coupling_cm = None

        # Simple configuration instance
        sim_cfg = SimulationConfig(
            simulation_type="1d",
            t_coh=10.5,
            ode_solver="ME",
            rwa_sl=True,
            t_det_max=100.0,
            t_wait=5.0,
            dt=0.1,
            n_inhomogen=1,
        )

        # Mock the helper functions to see what they're called with
        with (
            patch("qspectro2d.utils.file_naming.DATA_DIR", temp_dir),
            patch("qspectro2d.utils.file_naming.generate_base_sub_dir") as mock_subdir,
            patch("qspectro2d.utils.file_naming._generate_base_filename") as mock_base,
            patch("qspectro2d.utils.file_naming._generate_unique_filename") as mock_unique,
        ):

            # Set up return values
            mock_subdir.return_value = Path("1d_spectroscopy/N1/ME/RWA")
            mock_base.return_value = "sim_t10.5_ME_RWA"
            mock_unique.return_value = str(temp_dir / "1d_spectroscopy/N1/ME/RWA/sim_t10.5_ME_RWA")

            # Call the function
            result = generate_unique_data_filename(mock_system, sim_cfg)

            # Print detailed information
            print(f"\n=== TEST RESULTS ===")
            print(f"Result: {result}")
            print(f"Result type: {type(result)}")
            print(f"Result exists: {Path(result).exists()}")

            print(f"\n=== FUNCTION CALLS ===")
            print(f"generate_base_sub_dir called with:")
            print(f"  - sim_config: {mock_subdir.call_args[0][0]}")
            print(f"  - system: {mock_subdir.call_args[0][1]}")

            print(f"_generate_base_filename called with:")
            print(f"  - system: {mock_base.call_args[0][0]}")
            print(f"  - sim_config: {mock_base.call_args[0][1]}")

            print(f"_generate_unique_filename called with:")
            print(f"  - path: {mock_unique.call_args[0][0]}")
            print(f"  - base_name: {mock_unique.call_args[0][1]}")

            # Check directory creation
            expected_dir = temp_dir / "1d_spectroscopy/N1/ME/RWA"
            print(f"\n=== DIRECTORY CREATION ===")
            print(f"Expected directory: {expected_dir}")
            print(f"Directory exists: {expected_dir.exists()}")
            print(f"Directory is dir: {expected_dir.is_dir()}")

            # Basic assertions
            assert isinstance(result, str)
            assert len(result) > 0

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_with_different_configs():
    """Test with various configurations to see behavior."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        mock_system = MagicMock()
        mock_system.n_atoms = 2
        mock_system.coupling_cm = 15.0

        configs = [
            {
                "name": "1D simulation",
                "config": SimulationConfig(
                    simulation_type="1d",
                    t_coh=25.0,
                    ode_solver="BR",
                    rwa_sl=False,
                    t_det_max=200.0,
                    t_wait=10.0,
                    dt=0.05,
                    n_inhomogen=5,
                ),
            },
            {
                "name": "2D simulation",
                "config": SimulationConfig(
                    simulation_type="2d",
                    ode_solver="ME",
                    rwa_sl=True,
                    t_det_max=150.0,
                    t_wait=7.5,
                    dt=0.2,
                    n_inhomogen=1,
                ),
            },
        ]

        for test_case in configs:
            print(f"\n=== TESTING: {test_case['name']} ===")

            with (
                patch("qspectro2d.utils.file_naming.DATA_DIR", temp_dir),
                patch("qspectro2d.utils.file_naming.generate_base_sub_dir") as mock_subdir,
                patch("qspectro2d.utils.file_naming._generate_base_filename") as mock_base,
                patch("qspectro2d.utils.file_naming._generate_unique_filename") as mock_unique,
            ):

                # Set up return values based on config
                cfg = test_case["config"]
                sim_type = cfg.simulation_type
                solver = cfg.ode_solver
                rwa = "RWA" if cfg.rwa_sl else "noRWA"

                mock_subdir.return_value = Path(f"{sim_type}_spectroscopy/N2/{solver}/{rwa}")
                mock_base.return_value = f"sim_{sim_type}_{solver}_{rwa}"
                mock_unique.return_value = str(
                    temp_dir
                    / f"{sim_type}_spectroscopy/N2/{solver}/{rwa}/sim_{sim_type}_{solver}_{rwa}"
                )

                result = generate_unique_data_filename(mock_system, cfg)

                print(f"Config: {test_case['config']}")
                print(f"Result: {result}")
                print(f"Subdir called: {mock_subdir.called}")
                print(f"Base filename called: {mock_base.called}")
                print(f"Unique filename called: {mock_unique.called}")

                assert isinstance(result, str)

    finally:
        shutil.rmtree(temp_dir)


def test_error_scenarios():
    """Test error handling scenarios."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        mock_system = MagicMock()
        mock_system.n_atoms = 1

        sim_cfg = SimulationConfig(
            simulation_type="1d",
            t_coh=10.0,
            ode_solver="ME",
            rwa_sl=True,
            t_det_max=100.0,
            t_wait=5.0,
            dt=0.1,
        )

        # Test error in generate_base_sub_dir
        print(f"\n=== TESTING ERROR IN generate_base_sub_dir ===")
        with (
            patch("qspectro2d.utils.file_naming.DATA_DIR", temp_dir),
            patch("qspectro2d.utils.file_naming.generate_base_sub_dir") as mock_subdir,
        ):

            mock_subdir.side_effect = ValueError("Subdirectory generation failed")

            try:
                result = generate_unique_data_filename(mock_system, sim_cfg)
                print(f"Unexpected success: {result}")
            except ValueError as e:
                print(f"Expected error caught: {e}")
                assert "Subdirectory generation failed" in str(e)

        # Test error in _generate_base_filename
        print(f"\n=== TESTING ERROR IN _generate_base_filename ===")
        with (
            patch("qspectro2d.utils.file_naming.DATA_DIR", temp_dir),
            patch("qspectro2d.utils.file_naming.generate_base_sub_dir") as mock_subdir,
            patch("qspectro2d.utils.file_naming._generate_base_filename") as mock_base,
        ):

            mock_subdir.return_value = Path("test_subdir")
            mock_base.side_effect = RuntimeError("Base filename generation failed")

            try:
                result = generate_unique_data_filename(mock_system, sim_cfg)
                print(f"Unexpected success: {result}")
            except RuntimeError as e:
                print(f"Expected error caught: {e}")
                assert "Base filename generation failed" in str(e)

    finally:
        shutil.rmtree(temp_dir)


def test_directory_creation_details():
    """Test directory creation behavior in detail."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        mock_system = MagicMock()
        mock_system.n_atoms = 1

        sim_cfg = SimulationConfig(
            simulation_type="1d",
            t_coh=10.0,
            ode_solver="ME",
            rwa_sl=True,
            t_det_max=100.0,
            t_wait=5.0,
            dt=0.1,
        )

        # Create nested directory structure
        nested_path = "deep/nested/directory/structure"

        with (
            patch("qspectro2d.utils.file_naming.DATA_DIR", temp_dir),
            patch("qspectro2d.utils.file_naming.generate_base_sub_dir") as mock_subdir,
            patch("qspectro2d.utils.file_naming._generate_base_filename") as mock_base,
            patch("qspectro2d.utils.file_naming._generate_unique_filename") as mock_unique,
        ):

            mock_subdir.return_value = Path(nested_path)
            mock_base.return_value = "test_filename"
            mock_unique.return_value = str(temp_dir / nested_path / "test_filename")

            print(f"\n=== TESTING DIRECTORY CREATION ===")
            print(f"Temp dir: {temp_dir}")
            print(f"Nested path: {nested_path}")
            print(f"Full expected path: {temp_dir / nested_path}")
            print(f"Directory exists before: {(temp_dir / nested_path).exists()}")

            result = generate_unique_data_filename(mock_system, sim_cfg)

            print(f"Result: {result}")
            print(f"Directory exists after: {(temp_dir / nested_path).exists()}")
            print(f"Directory is dir: {(temp_dir / nested_path).is_dir()}")

            # Check all parent directories
            current_path = temp_dir
            for part in nested_path.split("/"):
                current_path = current_path / part
                print(
                    f"  {current_path}: exists={current_path.exists()}, is_dir={current_path.is_dir()}"
                )

            assert (temp_dir / nested_path).exists()
            assert (temp_dir / nested_path).is_dir()

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Running simplified tests...")
    test_basic_functionality()
    test_with_different_configs()
    test_error_scenarios()
    test_directory_creation_details()
    print("All tests completed!")
