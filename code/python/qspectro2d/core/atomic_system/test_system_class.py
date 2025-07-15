"""
Tests for the AtomicSystem class.

This module contains comprehensive tests for the AtomicSystem class,
including initialization, properties, serialization, and edge cases.
"""

from matplotlib.pylab import f
import pytest
import numpy as np
from qutip import basis, tensor

from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.core.utils_and_config import convert_cm_to_fs, HBAR


class TestAtomicSystemInitialization:
    """Test AtomicSystem initialization and validation."""

    def test_single_atom_default_initialization(self):
        """Test default initialization for single atom system."""
        system = AtomicSystem()

        assert system.N_atoms == 1
        assert system.freqs_cm == [16000.0]
        assert system.dip_moments == [1.0]
        assert system.J_cm is None
        assert system.Delta_cm is None
        assert system.psi_ini is not None

        print("✓ Single atom default initialization successful")
        print(f"  - N_atoms: {system.N_atoms}")
        print(f"  - freqs_cm: {system.freqs_cm}")
        print(f"  - dip_moments: {system.dip_moments}")

    def test_two_atom_initialization(self):
        """Test initialization for two atom system."""
        system = AtomicSystem(
            N_atoms=2, freqs_cm=[16000.0, 15640.0], dip_moments=[1.0, 1.2]
        )

        assert system.N_atoms == 2
        assert system.freqs_cm == [16000.0, 15640.0]
        assert system.dip_moments == [1.0, 1.2]
        assert system.J_cm == 0.0  # Default coupling

        print("✓ Two atom initialization successful")
        print(f"  - N_atoms: {system.N_atoms}")
        print(f"  - freqs_cm: {system.freqs_cm}")
        print(f"  - dip_moments: {system.dip_moments}")
        print(f"  - J_cm: {system.J_cm}")

    def test_single_frequency_expansion(self):
        """Test automatic expansion of single frequency to multiple atoms."""
        system = AtomicSystem(N_atoms=3, freqs_cm=[16000.0])

        assert len(system.freqs_cm) == 3
        assert all(freq == 16000.0 for freq in system.freqs_cm)

        print("✓ Single frequency expansion successful")
        print(f"  - Original: [16000.0]")
        print(f"  - Expanded: {system.freqs_cm}")

    def test_single_dipole_expansion(self):
        """Test automatic expansion of single dipole moment to multiple atoms."""
        system = AtomicSystem(N_atoms=2, dip_moments=[1.5])

        assert len(system.dip_moments) == 2
        assert all(dip == 1.5 for dip in system.dip_moments)

        print("✓ Single dipole expansion successful")
        print(f"  - Original: [1.5]")
        print(f"  - Expanded: {system.dip_moments}")

    def test_frequency_validation_error(self):
        """Test error handling for mismatched frequency count."""
        with pytest.raises(ValueError, match="freqs_cm has .* elements but N_atoms="):
            AtomicSystem(N_atoms=2, freqs_cm=[16000.0, 15800.0, 15600.0])

        print("✓ Frequency validation error handling works")

    def test_dipole_validation_error(self):
        """Test error handling for mismatched dipole moment count."""
        with pytest.raises(
            ValueError, match="dip_moments has .* elements but N_atoms="
        ):
            AtomicSystem(N_atoms=2, dip_moments=[1.0, 1.2, 1.5])

        print("✓ Dipole validation error handling works")


class TestAtomicSystemProperties:
    """Test AtomicSystem properties and computed values."""

    def test_basis_single_atom(self):
        """Test basis generation for single atom."""
        system = AtomicSystem(N_atoms=1)

        assert len(system.basis) == 2
        assert system.basis[0] == basis(2, 0)  # ground state
        assert system.basis[1] == basis(2, 1)  # excited state

        print("✓ Single atom basis generation successful")
        print(f"  - Basis size: {len(system.basis)}")
        print(f"  - Ground state: {system.basis[0].dims}")
        print(f"  - Excited state: {system.basis[1].dims}")

    def test_basis_two_atoms(self):
        """Test basis generation for two atoms."""
        system = AtomicSystem(N_atoms=2)

        assert len(system.basis) == 4
        # Test tensor product structure
        expected_basis = [
            tensor(basis(2, 0), basis(2, 0)),  # |gg⟩
            tensor(basis(2, 1), basis(2, 0)),  # |eg⟩
            tensor(basis(2, 0), basis(2, 1)),  # |ge⟩
            tensor(basis(2, 1), basis(2, 1)),  # |ee⟩
        ]

        for i, (actual, expected) in enumerate(zip(system.basis, expected_basis)):
            assert actual == expected

        print("✓ Two atom basis generation successful")
        print(f"  - Basis size: {len(system.basis)}")
        print(f"  - Basis dims: {[state.dims for state in system.basis]}")

    def test_frequency_conversion(self):
        """Test frequency conversion from cm^-1 to fs^-1."""
        system = AtomicSystem(N_atoms=1, freqs_cm=[16000.0])

        freq_fs = system.freqs_fs(0)
        expected_freq_fs = convert_cm_to_fs(16000.0)

        assert np.isclose(freq_fs, expected_freq_fs)

        print("✓ Frequency conversion successful")
        print(f"  - Original: {system.freqs_cm[0]} cm^-1")
        print(f"  - Converted: {freq_fs:.6f} fs^-1")

    def test_hamiltonian_single_atom(self):
        """Test Hamiltonian generation for single atom."""
        system = AtomicSystem(N_atoms=1, freqs_cm=[16000.0])

        H = system.H0_N_canonical

        # Check dimensions
        assert H.dims == [[2], [2]]

        # Check that only excited state has non-zero energy
        H_matrix = H.full()
        assert np.isclose(H_matrix[0, 0], 0.0)  # Ground state energy = 0
        assert np.isclose(
            H_matrix[1, 1], convert_cm_to_fs(16000.0)
        )  # Excited state energy = freq

        print("✓ Single atom Hamiltonian generation successful")
        print(f"  - Dimensions: {H.dims}")
        print(f"  - Ground state energy: {H_matrix[0, 0]:.6f}")
        print(f"  - Excited state energy: {H_matrix[1, 1]:.6f}")

    def test_hamiltonian_two_atoms(self):
        """Test Hamiltonian generation for two atoms."""
        system = AtomicSystem(N_atoms=2, freqs_cm=[16000.0, 15800.0], J_cm=100.0)

        H = system.H0_N_canonical

        # Check dimensions (4x4 for 2-atom system)
        assert H.dims == [[2, 2], [2, 2]]

        # Check that ground state has zero energy
        H_matrix = H.full()
        print(f"H_matrix:\n{H_matrix}", flush=True)
        assert np.isclose(H_matrix[0, 0], 0.0)

        # Check that the Hamiltonian is Hermitian
        assert np.allclose(H_matrix, H_matrix.conj().T)

        # Check that eigenvalues are real and ordered
        eigenvals = np.real(np.diag(H_matrix))
        assert eigenvals[0] == 0.0  # Ground state
        assert all(
            eigenvals[i] >= eigenvals[i - 1] for i in range(1, len(eigenvals))
        )  # Ordered

        print("✓ Two atom Hamiltonian generation successful")
        print(f"  - Dimensions: {H.dims}")
        print(f"  - Ground state energy: {H_matrix[0, 0]:.6f}")
        print(f"  - Matrix diagonal: {np.diag(H_matrix)}")
        print(f"  - Is Hermitian: {np.allclose(H_matrix, H_matrix.conj().T)}")

    def test_eigenstates_calculation(self):
        """Test eigenstate calculation."""
        system = AtomicSystem(N_atoms=1, freqs_cm=[16000.0])

        eigenvals, eigenvecs = system.eigenstates

        assert len(eigenvals) == 2
        assert len(eigenvecs) == 2
        assert eigenvals[0] <= eigenvals[1]  # Should be ordered

        print("✓ Eigenstate calculation successful")
        print(f"  - Eigenvalues: {eigenvals}")
        print(f"  - Eigenvector dimensions: {[vec.dims for vec in eigenvecs]}")

    def test_dipole_operator_single_atom(self):
        """Test dipole operator for single atom."""
        system = AtomicSystem(N_atoms=1, dip_moments=[1.5])

        sm_op = system.sm_op
        dip_op = system.dip_op

        # Check dimensions
        assert sm_op.dims == [[2], [2]]
        assert dip_op.dims == [[2], [2]]

        # Check that dip_op is Hermitian
        assert dip_op.isherm

        print("✓ Single atom dipole operator successful")
        print(f"  - sm_op dimensions: {sm_op.dims}")
        print(f"  - dip_op is Hermitian: {dip_op.isherm}")

    def test_update_frequencies(self):
        """Test frequency updating functionality."""
        system = AtomicSystem(N_atoms=2, freqs_cm=[16000.0, 15800.0])

        original_freqs = system.freqs_cm.copy()
        new_freqs = [16100.0, 15700.0]

        system.update_freqs_cm(new_freqs)

        assert system.freqs_cm == new_freqs
        assert len(system.freqs_cm_history) == 2
        assert system.freqs_cm_history[0] == original_freqs
        assert system.freqs_cm_history[1] == new_freqs

        print("✓ Frequency update successful")
        print(f"  - Original: {original_freqs}")
        print(f"  - Updated: {system.freqs_cm}")
        print(f"  - History length: {len(system.freqs_cm_history)}")

    def test_update_frequencies_validation(self):
        """Test frequency update validation."""
        system = AtomicSystem(N_atoms=2)

        with pytest.raises(ValueError, match="Expected .* frequencies"):
            system.update_freqs_cm([16000.0])  # Wrong number of frequencies

        print("✓ Frequency update validation works")


class TestAtomicSystemSerialization:
    """Test AtomicSystem serialization and deserialization."""

    def test_to_dict_single_atom(self):
        """Test dictionary serialization for single atom."""
        system = AtomicSystem(N_atoms=1, freqs_cm=[16000.0], dip_moments=[1.2])

        data = system.to_dict()

        expected_keys = {"N_atoms", "freqs_cm", "dip_moments"}
        assert set(data.keys()) == expected_keys
        assert data["N_atoms"] == 1
        assert data["freqs_cm"] == [16000.0]
        assert data["dip_moments"] == [1.2]

        print("✓ Single atom dictionary serialization successful")
        print(f"  - Keys: {list(data.keys())}")
        print(f"  - Data: {data}")

    def test_to_dict_two_atoms_with_coupling(self):
        """Test dictionary serialization for two atoms with coupling."""
        system = AtomicSystem(
            N_atoms=2,
            freqs_cm=[16000.0, 15800.0],
            dip_moments=[1.0, 1.2],
            J_cm=50.0,
            Delta_cm=10.0,
        )

        data = system.to_dict()

        expected_keys = {"N_atoms", "freqs_cm", "dip_moments", "J_cm", "Delta_cm"}
        assert set(data.keys()) == expected_keys
        assert data["J_cm"] == 50.0
        assert data["Delta_cm"] == 10.0

        print("✓ Two atom dictionary serialization with coupling successful")
        print(f"  - Keys: {list(data.keys())}")
        print(f"  - J_cm: {data['J_cm']}")
        print(f"  - Delta_cm: {data['Delta_cm']}")

    def test_json_serialization_roundtrip(self):
        """Test JSON serialization and deserialization roundtrip."""
        original_system = AtomicSystem(
            N_atoms=2, freqs_cm=[16000.0, 15800.0], dip_moments=[1.0, 1.2], J_cm=100.0
        )

        # Serialize to JSON
        json_str = original_system.to_json()

        # Deserialize from JSON
        reconstructed_system = AtomicSystem.from_json(json_str)

        # Compare essential attributes
        assert reconstructed_system.N_atoms == original_system.N_atoms
        assert reconstructed_system.freqs_cm == original_system.freqs_cm
        assert reconstructed_system.dip_moments == original_system.dip_moments
        assert reconstructed_system.J_cm == original_system.J_cm

        print("✓ JSON serialization roundtrip successful")
        print(f"  - Original N_atoms: {original_system.N_atoms}")
        print(f"  - Reconstructed N_atoms: {reconstructed_system.N_atoms}")
        print(f"  - JSON length: {len(json_str)} characters")

    def test_from_dict_reconstruction(self):
        """Test reconstruction from dictionary."""
        data = {
            "N_atoms": 2,
            "freqs_cm": [16000.0, 15800.0],
            "dip_moments": [1.0, 1.2],
            "J_cm": 75.0,
        }

        system = AtomicSystem.from_dict(data)

        assert system.N_atoms == 2
        assert system.freqs_cm == [16000.0, 15800.0]
        assert system.dip_moments == [1.0, 1.2]
        assert system.J_cm == 75.0

        print("✓ Dictionary reconstruction successful")
        print(f"  - Reconstructed system: N_atoms={system.N_atoms}")
        print(f"  - Frequencies: {system.freqs_cm}")


class TestAtomicSystemEdgeCases:
    """Test edge cases and error conditions."""

    def test_three_atom_system_initialization(self):
        """Test initialization of 3-atom system (single excitation subspace)."""
        system = AtomicSystem(N_atoms=3, freqs_cm=[16000.0, 15800.0, 15600.0])

        assert system.N_atoms == 3
        assert len(system.basis) == 3  # Ground + 2 single excitations

        print("✓ Three atom system initialization successful")
        print(f"  - N_atoms: {system.N_atoms}")
        print(f"  - Basis size: {len(system.basis)}")

    def test_omega_ij_calculation(self):
        """Test energy difference calculation."""
        system = AtomicSystem(N_atoms=1, freqs_cm=[16000.0])

        # For single atom: E_0 = 0, E_1 = hbar * omega
        omega_10 = system.omega_ij(1, 0)
        omega_01 = system.omega_ij(0, 1)

        assert omega_10 > 0
        assert omega_01 == -omega_10

        print("✓ Energy difference calculation successful")
        print(f"  - ω₁₀: {omega_10:.6f} fs^-1")
        print(f"  - ω₀₁: {omega_01:.6f} fs^-1")

    def test_theta_calculation_two_atoms(self):
        """Test theta parameter calculation for two atoms."""
        system = AtomicSystem(N_atoms=2, freqs_cm=[16000.0, 15800.0], J_cm=100.0)

        theta = system.theta

        assert 0 <= theta <= np.pi / 4  # Reasonable range for theta

        print("✓ Theta calculation successful")
        print(f"  - Theta: {theta:.6f} radians")
        print(f"  - Theta: {np.degrees(theta):.2f} degrees")


def test_summary_method():
    """Test the summary method produces reasonable output."""
    system = AtomicSystem(
        N_atoms=2, freqs_cm=[16000.0, 15800.0], dip_moments=[1.0, 1.2], J_cm=50.0
    )

    # Capture summary output
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    system.summary()

    sys.stdout = old_stdout
    output = captured_output.getvalue()

    # Check that key information is present
    assert "AtomicSystem Summary" in output
    assert "N_atoms" in output
    assert "16000.0" in output
    assert "15800.0" in output
    assert "50.0" in output

    print("✓ Summary method works correctly")
    print(f"  - Output length: {len(output)} characters")
    print(f"  - Contains essential info: ✓")


if __name__ == "__main__":
    """Run tests when executed directly."""
    print("=" * 60)
    print("RUNNING ATOMIC SYSTEM TESTS")
    print("=" * 60)

    # Run all test classes
    test_classes = [
        TestAtomicSystemInitialization,
        TestAtomicSystemProperties,
        TestAtomicSystemSerialization,
        TestAtomicSystemEdgeCases,
    ]

    for test_class in test_classes:
        print(f"\n--- Running {test_class.__name__} ---")
        test_instance = test_class()

        # Run all test methods in the class
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                print(f"\n{method_name}:")
                try:
                    method = getattr(test_instance, method_name)
                    method()
                except Exception as e:
                    print(f"  ✗ FAILED: {e}")
                    import traceback

                    traceback.print_exc()

    # Run standalone test
    print(f"\n--- Running standalone tests ---")
    print(f"\ntest_summary_method:")
    try:
        test_summary_method()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
