import pytest
import numpy as np
from qutip import basis, tensor

from qspectro2d.core.atomic_system.system_class import AtomicSystem
from qspectro2d.constants import convert_cm_to_fs


class TestAtomicSystemInitialization:
    """Test AtomicSystem initialization and validation."""

    def test_single_atom_default_initialization(self):
        """Test default initialization for single atom system."""
        system = AtomicSystem()

        assert system.n_atoms == 1
        assert system.frequencies_cm == [16000.0]
        assert system.dip_moments == [1.0]
        assert system.coupling_cm == 0.0
        assert system.delta_cm is None
        assert system.psi_ini is not None

        print("✓ Single atom default initialization successful")
        print(f"  - n_atoms: {system.n_atoms}")
        print(f"  - frequencies_cm: {system.frequencies_cm}")
        print(f"  - dip_moments: {system.dip_moments}")

    def test_two_atom_initialization(self):
        """Test initialization for two atom system."""
        system = AtomicSystem(
            n_atoms=2,
            n_chains=1,  # Linear chain
            frequencies_cm=[16000.0, 15640.0],
            dip_moments=[1.0, 1.2],
            coupling_cm=50.0,
        )

        assert system.n_atoms == 2
        assert system.n_chains == 1
        assert system.n_rings == 2  # n_atoms // n_chains
        assert system.frequencies_cm == [16000.0, 15640.0]
        assert system.dip_moments == [1.0, 1.2]
        assert system.coupling_cm == 50.0
        assert system.positions.shape == (2, 3)  # Always has positions
        assert hasattr(system, "coupling_matrix_cm")  # New coupling matrix property

        print("✓ Two atom initialization successful")
        print(f"  - n_atoms: {system.n_atoms}")
        print(f"  - n_chains: {system.n_chains}")
        print(f"  - n_rings: {system.n_rings}")
        print(f"  - frequencies_cm: {system.frequencies_cm}")
        print(f"  - dip_moments: {system.dip_moments}")
        print(f"  - coupling_cm: {system.coupling_cm}")
        print(f"  - positions shape: {system.positions.shape}")
        print(f"  - coupling_cm: {system.coupling_cm}")

    def test_single_frequency_expansion(self):
        """Test automatic expansion of single frequency to multiple atoms."""
        system = AtomicSystem(n_atoms=3, frequencies_cm=[16000.0])

        assert len(system.frequencies_cm) == 3
        assert all(freq == 16000.0 for freq in system.frequencies_cm)

        print("✓ Single frequency expansion successful")
        print(f"  - Original: [16000.0]")
        print(f"  - Expanded: {system.frequencies_cm}")

    def test_single_dipole_expansion(self):
        """Test automatic expansion of single dipole moment to multiple atoms."""
        system = AtomicSystem(n_atoms=2, dip_moments=[1.5])

        assert len(system.dip_moments) == 2
        assert all(dip == 1.5 for dip in system.dip_moments)

        print("✓ Single dipole expansion successful")
        print(f"  - Original: [1.5]")
        print(f"  - Expanded: {system.dip_moments}")

    def test_frequency_validation_error(self):
        """Test error handling for mismatched frequency count."""
        with pytest.raises(
            ValueError, match="frequencies_cm has .* elements but n_atoms="
        ):
            AtomicSystem(n_atoms=2, frequencies_cm=[16000.0, 15800.0, 15600.0])

        print("✓ Frequency validation error handling works")

    def test_dipole_validation_error(self):
        """Test error handling for mismatched dipole moment count."""
        with pytest.raises(
            ValueError, match="dip_moments has .* elements but n_atoms="
        ):
            AtomicSystem(n_atoms=2, dip_moments=[1.0, 1.2, 1.5])

        print("✓ Dipole validation error handling works")


class TestAtomicSystemProperties:
    """Test AtomicSystem properties and computed values."""

    def test_basis_single_atom(self):
        """Test basis generation for single atom."""
        system = AtomicSystem(n_atoms=1)

        assert system.dimension == 2
        assert len(system.basis) == 2
        # Check basis states are the correct dimensions
        assert system.basis[0].dims == [[2], [1]]  # ground state
        assert system.basis[1].dims == [[2], [1]]  # excited state

        print("✓ Single atom basis generation successful")
        print(f"  - Basis size: {len(system.basis)}")
        print(f"  - Dimension: {system.dimension}")
        print(f"  - Ground state dims: {system.basis[0].dims}")
        print(f"  - Excited state dims: {system.basis[1].dims}")

    def test_basis_two_atoms(self):
        """Test basis generation for two atoms."""
        system = AtomicSystem(n_atoms=2)

        assert (
            system.dimension == 3
        )  # ground + 2 single excitations (max_excitation=1 by default)
        assert len(system.basis) == 3

        # Check basis dimensions - now using computational basis
        for i, state in enumerate(system.basis):
            assert state.dims == [[3], [1]]  # All states in 3D computational basis

        print("✓ Two atom basis generation successful")
        print(f"  - Basis size: {len(system.basis)}")
        print(f"  - Dimension: {system.dimension}")
        print(f"  - Basis dims: {[state.dims for state in system.basis]}")

    def test_basis_two_atoms_double_excitation(self):
        """Test basis generation for two atoms with double excitation."""
        system = AtomicSystem(n_atoms=2, max_excitation=2)

        assert system.dimension == 4  # ground + 2 singles + 1 double
        assert len(system.basis) == 4

        print("✓ Two atom double excitation basis generation successful")
        print(f"  - Basis size: {len(system.basis)}")
        print(f"  - Dimension: {system.dimension}")
        print(f"  - max_excitation: {system.max_excitation}")

        # Check that all basis states have correct dimensions for computational basis
        for i, state in enumerate(system.basis):
            assert state.dims == [[3], [1]]

        print("✓ Two atom basis generation successful")
        print(f"  - Basis size: {len(system.basis)}")
        print(f"  - Dimension: {system.dimension}")
        print(f"  - Basis dims: {[state.dims for state in system.basis]}")

    def test_basis_two_atoms_double_excitation(self):
        """Test basis generation for two atoms with double excitation."""
        system = AtomicSystem(n_atoms=2, max_excitation=2)

        assert system.dimension == 4  # ground + 2 single + 1 double excitation
        assert len(system.basis) == 4

        # Check that all basis states have correct dimensions
        for i, state in enumerate(system.basis):
            assert state.dims == [[4], [1]]

        print("✓ Two atom double excitation basis generation successful")
        print(f"  - Basis size: {len(system.basis)}")
        print(f"  - Dimension: {system.dimension}")
        print(f"  - Basis dims: {[state.dims for state in system.basis]}")

    def test_frequency_conversion(self):
        """Test frequency conversion from cm^-1 to fs^-1."""
        system = AtomicSystem(n_atoms=1, frequencies_cm=[16000.0])

        # Test the new frequencies property (returns array)
        freq_fs = system.frequencies[0]
        expected_freq_fs = convert_cm_to_fs(16000.0)

        assert np.isclose(freq_fs, expected_freq_fs)

        print("✓ Frequency conversion successful")
        print(f"  - Original: {system.frequencies_cm[0]} cm^-1")
        print(f"  - Converted: {freq_fs:.6f} fs^-1")
        print(f"  - All frequencies (fs^-1): {system.frequencies}")

    def test_hamiltonian_single_atom(self):
        """Test Hamiltonian generation for single atom."""
        system = AtomicSystem(n_atoms=1, frequencies_cm=[16000.0])

        H = system.hamiltonian

        # Check dimensions
        assert H.dims == [[2], [2]]

        # Check that only excited state has non-zero energy
        H_matrix = H.full()
        assert np.isclose(H_matrix[0, 0], 0.0)  # Ground state energy = 0
        expected_energy = convert_cm_to_fs(16000.0)
        assert np.isclose(H_matrix[1, 1], expected_energy, rtol=1e-10)  # Excited state

        print("✓ Single atom Hamiltonian generation successful")
        print(f"  - Dimensions: {H.dims}")
        print(f"  - Ground state energy: {H_matrix[0, 0]:.6f}")
        print(f"  - Excited state energy: {H_matrix[1, 1]:.6f}")

    def test_hamiltonian_two_atoms(self):
        """Test Hamiltonian generation for two atoms."""
        system = AtomicSystem(
            n_atoms=2, frequencies_cm=[16000.0, 15800.0], coupling_cm=100.0
        )

        H = system.hamiltonian

        # Check dimensions (3x3 for 2-atom system with single excitation)
        expected_dim = system.dimension
        assert H.dims == [[expected_dim], [expected_dim]]

        # Check that ground state has zero energy
        H_matrix = H.full()
        print(f"H_matrix:\n{H_matrix}", flush=True)
        assert np.isclose(H_matrix[0, 0], 0.0)

        # Check that the Hamiltonian is Hermitian
        assert np.allclose(H_matrix, H_matrix.conj().T)

        print("✓ Two atom Hamiltonian generation successful")
        print(f"  - Dimensions: {H.dims}")
        print(f"  - Ground state energy: {H_matrix[0, 0]:.6f}")
        print(f"  - Matrix diagonal: {np.diag(H_matrix)}")
        print(f"  - Is Hermitian: {np.allclose(H_matrix, H_matrix.conj().T)}")

    def test_eigenstates_calculation(self):
        """Test eigenstate calculation."""
        system = AtomicSystem(n_atoms=1, frequencies_cm=[16000.0])

        eigenvals, eigenvecs = system.eigenstates

        assert len(eigenvals) == 2
        assert len(eigenvecs) == 2
        assert eigenvals[0] <= eigenvals[1]  # Should be ordered

        print("✓ Eigenstate calculation successful")
        print(f"  - Eigenvalues: {eigenvals}")
        print(f"  - Eigenvector dimensions: {[vec.dims for vec in eigenvecs]}")

    def test_dipole_operator_single_atom(self):
        """Test dipole operator for single atom."""
        system = AtomicSystem(n_atoms=1, dip_moments=[1.5])

        lowering_op = system.lowering_op
        dipole_op = system.dipole_op

        # Check dimensions
        assert lowering_op.dims == [[2], [2]]
        assert dipole_op.dims == [[2], [2]]

        # Check that dipole_op is Hermitian
        assert dipole_op.isherm

        print("✓ Single atom dipole operator successful")
        print(f"  - lowering_op dimensions: {lowering_op.dims}")
        print(f"  - dipole_op is Hermitian: {dipole_op.isherm}")

    def test_update_frequencies(self):
        """Test frequency updating functionality."""
        system = AtomicSystem(n_atoms=2, frequencies_cm=[16000.0, 15800.0])

        original_freqs = system.frequencies_cm.copy()
        new_freqs = [16100.0, 15700.0]

        system.update_frequencies_cm(new_freqs)

        assert system.frequencies_cm == new_freqs
        assert len(system.frequencies_cm_history) == 2
        assert system.frequencies_cm_history[0] == original_freqs
        assert system.frequencies_cm_history[1] == new_freqs

        print("✓ Frequency update successful")
        print(f"  - Original: {original_freqs}")
        print(f"  - Updated: {system.frequencies_cm}")
        print(f"  - History length: {len(system.frequencies_cm_history)}")

    def test_update_frequencies_validation(self):
        """Test frequency update validation."""
        system = AtomicSystem(n_atoms=2)

        with pytest.raises(ValueError, match="Expected .* frequencies"):
            system.update_frequencies_cm([16000.0])  # Wrong number of frequencies

        print("✓ Frequency update validation works")


class TestAtomicSystemSerialization:
    """Test AtomicSystem serialization and deserialization."""

    def test_to_dict_single_atom(self):
        """Test dictionary serialization for single atom."""
        system = AtomicSystem(n_atoms=1, frequencies_cm=[16000.0], dip_moments=[1.2])

        data = system.to_dict()

        # Single atom always includes coupling_cm (even if 0.0)
        expected_keys = {"n_atoms", "frequencies_cm", "dip_moments", "coupling_cm"}
        assert set(data.keys()) == expected_keys
        assert data["n_atoms"] == 1
        assert data["frequencies_cm"] == [16000.0]
        assert data["dip_moments"] == [1.2]
        assert data["coupling_cm"] == 0.0

        print("✓ Single atom dictionary serialization successful")
        print(f"  - Keys: {list(data.keys())}")
        print(f"  - Data: {data}")

    def test_to_dict_two_atoms_with_coupling(self):
        """Test dictionary serialization for two atoms with coupling."""
        system = AtomicSystem(
            n_atoms=2,
            frequencies_cm=[16000.0, 15800.0],
            dip_moments=[1.0, 1.2],
            coupling_cm=50.0,
            delta_cm=10.0,
        )

        data = system.to_dict()

        expected_keys = {
            "n_atoms",
            "frequencies_cm",
            "dip_moments",
            "coupling_cm",
            "delta_cm",
        }
        assert set(data.keys()) == expected_keys
        assert data["coupling_cm"] == 50.0
        assert data["delta_cm"] == 10.0

        print("✓ Two atom dictionary serialization with coupling successful")
        print(f"  - Keys: {list(data.keys())}")
        print(f"  - coupling_cm: {data['coupling_cm']}")
        print(f"  - delta_cm: {data['delta_cm']}")

    def test_json_serialization_roundtrip(self):
        """Test JSON serialization and deserialization roundtrip."""
        original_system = AtomicSystem(
            n_atoms=2,
            frequencies_cm=[16000.0, 15800.0],
            dip_moments=[1.0, 1.2],
            coupling_cm=100.0,
        )

        # Serialize to JSON
        json_str = original_system.to_json()

        # Deserialize from JSON
        reconstructed_system = AtomicSystem.from_json(json_str)

        # Compare essential attributes
        assert reconstructed_system.n_atoms == original_system.n_atoms
        assert reconstructed_system.frequencies_cm == original_system.frequencies_cm
        assert reconstructed_system.dip_moments == original_system.dip_moments
        assert reconstructed_system.coupling_cm == original_system.coupling_cm

        print("✓ JSON serialization roundtrip successful")
        print(f"  - Original n_atoms: {original_system.n_atoms}")
        print(f"  - Reconstructed n_atoms: {reconstructed_system.n_atoms}")
        print(f"  - JSON length: {len(json_str)} characters")

    def test_from_dict_reconstruction(self):
        """Test reconstruction from dictionary."""
        data = {
            "n_atoms": 2,
            "frequencies_cm": [16000.0, 15800.0],
            "dip_moments": [1.0, 1.2],
            "coupling_cm": 75.0,
        }

        system = AtomicSystem.from_dict(data)

        assert system.n_atoms == 2
        assert system.frequencies_cm == [16000.0, 15800.0]
        assert system.dip_moments == [1.0, 1.2]
        assert system.coupling_cm == 75.0

        print("✓ Dictionary reconstruction successful")
        print(f"  - Reconstructed system: n_atoms={system.n_atoms}")
        print(f"  - Frequencies: {system.frequencies_cm}")

    def test_serialization_with_geometry_params(self):
        """Test serialization with geometry parameters."""
        data = {
            "n_atoms": 6,
            "n_chains": 2,
            "frequencies_cm": [16000.0],
            "dip_moments": [1.0],
            "coupling_cm": 50.0,
        }

        system = AtomicSystem.from_dict(data)

        assert system.n_atoms == 6
        assert system.n_chains == 2
        assert system.n_rings == 3  # n_atoms // n_chains
        assert system.coupling_cm == 50.0

        # Check that geometry was set up correctly
        assert system.positions.shape == (6, 3)

        print("✓ Geometry parameters serialization successful")
        print(f"  - n_atoms: {system.n_atoms}, n_chains: {system.n_chains}")
        print(f"  - n_rings: {system.n_rings}")
        print(f"  - Positions shape: {system.positions.shape}")


class TestAtomicSystemEdgeCases:
    """Test edge cases and error conditions."""

    def test_three_atom_system_initialization(self):
        """Test initialization of 3-atom system (single excitation subspace)."""
        system = AtomicSystem(n_atoms=3, frequencies_cm=[16000.0, 15800.0, 15600.0])

        assert system.n_atoms == 3
        assert system.dimension == 4  # Ground + 3 single excitations
        assert len(system.basis) == 4

        # Test that positions are set (cylindrical geometry)
        assert system.positions.shape == (3, 3)  # 3 atoms, 3D positions

        # Test that coupling matrix is computed
        coupling_matrix = system.coupling_matrix_cm
        assert coupling_matrix.shape == (3, 3)

        print("✓ Three atom system initialization successful")
        print(f"  - n_atoms: {system.n_atoms}")
        print(f"  - Basis size: {len(system.basis)}")
        print(f"  - Positions shape: {system.positions.shape}")
        print(f"  - Coupling matrix shape: {coupling_matrix.shape}")

    def test_omega_ij_calculation(self):
        """Test energy difference calculation."""
        system = AtomicSystem(n_atoms=1, frequencies_cm=[16000.0])

        # For single atom: E_0 = 0, E_1 = hbar * omega
        omega_10 = system.omega_ij(1, 0)
        omega_01 = system.omega_ij(0, 1)

        assert omega_10 > 0
        assert omega_01 == -omega_10

        print("✓ Energy difference calculation successful")
        print(f"  - ω₁₀: {omega_10:.6f} fs^-1")
        print(f"  - ω₀₁: {omega_01:.6f} fs^-1")

    def test_cylindrical_geometry_linear_chain(self):
        """Test cylindrical geometry for linear chain (n_chains=1)."""
        system = AtomicSystem(n_atoms=4, n_chains=1, coupling_cm=50.0)

        # Should be linear chain
        assert system.n_chains == 1
        assert system.n_rings == 4

        positions = system.positions
        assert positions.shape == (4, 3)

        # Check it's actually linear (all x,y = 0, only z varies)
        assert np.allclose(positions[:, 0], 0)  # x = 0
        assert np.allclose(positions[:, 1], 0)  # y = 0
        assert np.allclose(positions[:, 2], [0, 1, 2, 3])  # z = 0,1,2,3

        print("✓ Linear chain geometry successful")
        print(f"  - n_chains: {system.n_chains}, n_rings: {system.n_rings}")
        print(f"  - Positions:\n{positions}")

    def test_cylindrical_geometry_multiple_chains(self):
        """Test cylindrical geometry for multiple chains."""
        system = AtomicSystem(n_atoms=6, n_chains=2, coupling_cm=20.0)

        # Should be 2 chains × 3 rings
        assert system.n_chains == 2
        assert system.n_rings == 3

        positions = system.positions
        assert positions.shape == (6, 3)

        # Check that we have atoms at different angular positions (distinct (x,y) ring centers)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]

        # With 2 chains at 0° and 180°, y can be ~0 for both; use x and (x,y) uniqueness
        unique_x = np.unique(np.round(x_coords, 6))
        unique_xy = np.unique(np.round(positions[:, :2], 6), axis=0)

        assert len(unique_x) >= 2  # At least 2 different x positions
        assert unique_xy.shape[0] >= system.n_chains  # Distinct centers match n_chains

        print("✓ Cylindrical geometry successful")
        print(f"  - n_chains: {system.n_chains}, n_rings: {system.n_rings}")
        print(f"  - Position range x: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
        print(f"  - Position range y: [{y_coords.min():.3f}, {y_coords.max():.3f}]")

    def test_isotropic_coupling_computation(self):
        """Test isotropic coupling matrix computation."""
        system = AtomicSystem(n_atoms=3, n_chains=1, coupling_cm=100.0)

        coupling_matrix = system.coupling_matrix_cm
        assert coupling_matrix.shape == (3, 3)

        # Check diagonal is zero
        assert np.allclose(np.diag(coupling_matrix), 0)

        # Check symmetry
        assert np.allclose(coupling_matrix, coupling_matrix.T)

        # Check 1/r^3 scaling: closer atoms should have stronger coupling
        # For linear chain: r_01 = 1, r_02 = 2, so J_02 should be ~1/8 of J_01
        J_01 = coupling_matrix[0, 1]
        J_02 = coupling_matrix[0, 2]
        expected_ratio = 1.0 / (2.0**3)  # (r_02/r_01)^3 = 2^3
        actual_ratio = J_02 / J_01

        assert np.isclose(actual_ratio, expected_ratio, rtol=0.1)

        print("✓ Isotropic coupling computation successful")
        print(f"  - Coupling matrix:\n{coupling_matrix}")
        print(f"  - J_01: {J_01:.3f}, J_02: {J_02:.3f}")
        print(
            f"  - Ratio J_02/J_01: {actual_ratio:.3f} (expected: {expected_ratio:.3f})"
        )


def test_summary_method():
    """Test the summary method produces reasonable output."""
    system = AtomicSystem(
        n_atoms=2,
        frequencies_cm=[16000.0, 15800.0],
        dip_moments=[1.0, 1.2],
        coupling_cm=50.0,
    )

    # Get summary string directly (summary() returns a string)
    output = system.summary()

    # Check that key information is present
    assert "=== AtomicSystem Summary ===" in output
    assert "n_atoms" in output
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
