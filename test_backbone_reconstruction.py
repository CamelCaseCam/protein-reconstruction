#!/usr/bin/env python3
"""
Comprehensive test suite for protein backbone reconstruction.
Focuses on identifying bugs that could lead to incorrect omega angles and "atomic spaghetti" structures.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import inspect
import sys
import os

# Add the reconstruct module to path (assuming it's in the same directory structure)
sys.path.insert(0, '/tmp/inputs/protein-reconstruction')

from reconstruct import reconstruct, assign_backbone, constants
import PeptideBuilder
import PeptideBuilder.Geometry as geo
import warnings
# Raise exception for numerical issues
np.seterr(all='raise')
warnings.filterwarnings("ignore")

# Helper function for CA envs
from get_calpha_env import extract_alpha_carbon_envs

class TestBackboneReconstructionBugs:
    """Test suite designed to catch backbone reconstruction bugs"""
    
    @pytest.fixture
    def simple_test_data(self):             
            """Create simple but realistic test data"""
            # Simple 3-residue peptide in extended conformation
            sequence = "ALA"
            # Peptidebuilder uses numerical warnings, so we can't have them error here
            with np.errstate(all="ignore"):
                structure = PeptideBuilder.make_extended_structure(sequence)
            alpha_carbon_envs, alpha_carbon_elem_types, reconst_seq = extract_alpha_carbon_envs(structure)
            assert reconst_seq == sequence, f"Reconstructed sequence {reconst_seq} does not match expected {sequence}"
            
            return alpha_carbon_envs, alpha_carbon_elem_types, sequence, structure
    
    @pytest.fixture
    def polypeptide_test_data(self):
        '''
        Larger test case in a non-extended conformation
        '''
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        with np.errstate(all="ignore"):
            structure = PeptideBuilder.initialize_res(sequence[0])
            for res in sequence[1:]:
                structure = PeptideBuilder.add_residue(structure, res)

        alpha_carbon_envs, alpha_carbon_elem_types, reconst_seq = extract_alpha_carbon_envs(structure)
        assert reconst_seq == sequence, f"Reconstructed sequence {reconst_seq} does not match expected {sequence}"
        
        return alpha_carbon_envs, alpha_carbon_elem_types, sequence, structure
    
    @pytest.fixture
    def dipeptide_angles_and_offsets(self):
        # Source: real M-I dipeptide
        offsets = [
            # Second alpha carbon
            [
                np.array([np.nan, np.nan, np.nan]),                           # No CA_i-1
                np.array([np.nan, np.nan, np.nan]),                           # No carbonyl_i-1
                np.array([-4.205,-15.679,-1.731]) - np.array([-3.888,-16.403,-0.487]),  # nitrogen_i to CA_i
                np.array([-4.399,-14.199,-1.444]) - np.array([-4.205,-15.679,-1.731]),  # CA_i to carbonyl_i
                np.array([-4.173,-13.345,-2.442]) - np.array([-4.399,-14.199,-1.444]),  # carbonyl_i to nitrogen_i+1
                np.array([-4.391,-11.897,-2.373]) - np.array([-4.173,-13.345,-2.442]),  # nitrogen_i+1 to CA_i+1
            ],
            [
                np.array([-4.399,-14.199,-1.444]) - np.array([-4.205,-15.679,-1.731]),  # CA_i-1 to carbonyl_i-1
                np.array([-4.173,-13.345,-2.442]) - np.array([-4.399,-14.199,-1.444]),  # carbonyl_i-1 to nitrogen_i
                np.array([-4.391,-11.897,-2.373]) - np.array([-4.173,-13.345,-2.442]),  # nitrogen_i to CA_i
                np.array([-5.359,-11.482,-3.479]) - np.array([-4.391,-11.897,-2.373]),  # CA_i to carbonyl_i
                np.array([np.nan, np.nan, np.nan]),                           # No nitrogen_i+1
                np.array([np.nan, np.nan, np.nan]),                           # No CA_i+1
            ]
        ]

        expected_output = [
            # Calculated using Avogadro
            {
                'peptide_bond': 1.333,
                'N_CA_C_angle': 110.267,    # Angle in degrees
                'CA_N_length': 1.474,
                'CA_C_length': 1.520,
                'phi': None,  # No phi for first residue
                'psi': 154.379,
                'omega': 175.901
            },
            {
                'peptide_bond': None,
                'N_CA_C_angle': 109.182,
                'CA_N_length': 1.466,
                'CA_C_length': 1.527,
                'phi': 122.914,
                'psi':  None,
                'omega': None
            }
        ]

        return np.array(offsets), expected_output
    
    @pytest.fixture
    def dipeptide_averaging_data(self):
        offsets = [
            # Second alpha carbon
            [
                np.array([np.nan, np.nan, np.nan]),                           # No CA_i-1
                np.array([np.nan, np.nan, np.nan]),                           # No carbonyl_i-1
                np.array([-4.205,-15.679,-1.731]) - np.array([-3.888,-16.403,-0.487]),  # nitrogen_i to CA_i
                # Couldn't find carbonyl_0
                np.array([np.nan, np.nan, np.nan]),                           # CA_i to carbonyl_i
                np.array([np.nan, np.nan, np.nan]),                           # carbonyl_i to nitrogen_i+1
                np.array([-4.391,-11.897,-2.373]) - np.array([-4.173,-13.345,-2.442]),  # nitrogen_i+1 to CA_i+1
            ],
            [
                np.array([np.nan, np.nan, np.nan]),                           # CA_i-1 to carbonyl_i-1
                np.array([np.nan, np.nan, np.nan]),                           # CA_i-1 to carbonyl_i-1
                np.array([-4.391,-11.897,-2.373]) - np.array([-4.173,-13.345,-2.442]),  # nitrogen_i to CA_i
                np.array([-5.359,-11.482,-3.479]) - np.array([-4.391,-11.897,-2.373]),  # CA_i to carbonyl_i
                np.array([np.nan, np.nan, np.nan]),                           # No nitrogen_i+1
                np.array([np.nan, np.nan, np.nan]),                           # No CA_i+1
            ]
        ]

        expected_output = [
            # Calculated using Avogadro
            {
                'peptide_bond': None,  # No peptide bond for first residue
                'N_CA_C_angle': None,    #  Missing carbonyl for the angle
                'CA_N_length': 1.474,
                'CA_C_length': None,
                'phi': None,  # No phi for first residue
                'psi': None,
                'omega': None
            },
            {
                'peptide_bond': None,
                'N_CA_C_angle': 109.182,
                'CA_N_length': 1.466,
                'CA_C_length': 1.527,
                'phi': None,
                'psi':  None,
                'omega': None
            }
        ]

        return np.array(offsets), expected_output
    
    # ===== TESTS FOR ATOM ASSIGNMENT BUGS =====
    
    def test_atom_assignment_basic_sanity(self, simple_test_data):
        """Test that atom assignment produces reasonable results"""
        alpha_carbon_envs, alpha_carbon_elem_types, sequence, structure = simple_test_data
        
        offsets = assign_backbone.assign_backbone_atoms(alpha_carbon_envs, alpha_carbon_elem_types)
        
        # Check output shape
        assert offsets.shape == (3, 6, 3), f"Expected shape (3, 6, 3), got {offsets.shape}"
        
        # Check that bond lengths are all correct
        model = structure[0]  # Use first model
        chain = model["A"]
        residues = list(chain.get_residues())
        for i in range(len(sequence)):
            res = residues[i]
            NtoCa_true = (res["CA"].get_vector() - res["N"].get_vector()).get_array()
            Ca_toC_true = (res["C"].get_vector() - res["CA"].get_vector()).get_array()
            CtoNip1 = (residues[i+1]["N"].get_vector() - res["C"].get_vector()).get_array() if i < len(sequence) - 1 else None

            # Just check N_i -> N_i+1 as other offsets will be checked in other iterations
            NtoCa_pred = offsets[i, 2]
            Ca_toC_pred = offsets[i, 3]
            CtoNip1_pred = offsets[i, 4] if CtoNip1 is not None else None
            assert np.allclose(NtoCa_true, NtoCa_pred, atol=0.1), f"N to CA mismatch at residue {i}: {NtoCa_true} vs {NtoCa_pred}"
            assert np.allclose(Ca_toC_true, Ca_toC_pred, atol=0.1), f"CA to C mismatch at residue {i}: {Ca_toC_true} vs {Ca_toC_pred}"
            if CtoNip1 is not None:
                assert np.allclose(CtoNip1, CtoNip1_pred, atol=0.1), f"C to N(i+1) mismatch at residue {i}: {CtoNip1} vs {CtoNip1_pred}"

    # ===== TESTS FOR DIHEDRAL ANGLE CALCULATION BUGS =====
    
    def test_dihedral_calculation_basic(self, dipeptide_angles_and_offsets):
        """Test basic dihedral angle calculation"""
        from reconstruct.reconstruct import dihedral, rad2deg
        
        # Calculate the first residue's omega dihedral angle
        offsets, expected_output = dipeptide_angles_and_offsets
        b1 = offsets[0][3]  # CA_i to C_i
        b2 = offsets[0][4]  # C_i to N_i+1
        b3 = offsets[0][5]  # N_i+1 to CA_i+1
        angle = dihedral(b1, b2, b3)
        angle_deg = rad2deg(angle)
        assert abs(angle_deg - expected_output[0]['omega']) < 1e-3, f"Expected omega angle {expected_output[0]['omega']}°, got {angle_deg}°"

    
    def test_dihedral_calculation_edge_cases(self):
        """Test dihedral calculation edge cases that could cause omega bugs"""
        from reconstruct.reconstruct import dihedral, rad2deg
        
        # Test case: parallel vectors (undefined dihedral)
        b1 = np.array([1, 0, 0])
        b2 = np.array([1, 0, 0])
        b3 = np.array([1, 0, 0])
        
        angle = dihedral(b1, b2, b3)
        # Should handle gracefully, not produce NaN or extreme values
        assert not np.isnan(angle), "Dihedral should not be NaN even for degenerate cases"
        
        # Test case: very small vectors
        b1 = np.array([1e-8, 0, 0])
        b2 = np.array([0, 1e-8, 0])
        b3 = np.array([0, 0, 1e-8])
        
        angle = dihedral(b1, b2, b3)
        angle_deg = rad2deg(angle)
        assert abs(angle_deg) < 1000, f"Dihedral of small vectors gave extreme angle: {angle_deg}°"
    
    def test_full_angle_calculation(self, dipeptide_angles_and_offsets):
        from reconstruct.reconstruct import integrate_neighbour_information
        """Test full angle calculation for a dipeptide"""
        offsets, expected_output = dipeptide_angles_and_offsets
        angle_information = integrate_neighbour_information(offsets, 2)
        # Check that the calculated angles match expected output
        for i, info in enumerate(expected_output):
            for key in info:
                if info[key] is not None:
                    assert abs(angle_information[i][key] - info[key]) < 1e-3, f"Angle {key} mismatch at residue {i}: expected {info[key]}, got {angle_information[i][key]}"
                else:
                    assert angle_information[i][key] is None, f"Expected None for {key} at residue {i}, got {angle_information[i][key]}"


    # ===== TESTS FOR FRAME COMPUTATION BUGS =====
    
    def test_frame_computation_boundary_conditions(self):
        """Test frame computation at sequence boundaries"""
        from reconstruct.reconstruct import compute_frame
        
        # Create minimal offsets for 3-residue sequence
        offsets = np.zeros((3, 6, 3))
        
        # Test first frame (frame_index=0)
        frame_info_0 = compute_frame(offsets, 0)
        # Should handle missing previous residue gracefully
        assert isinstance(frame_info_0, dict)
        
        # Test last frame (frame_index=2)
        frame_info_2 = compute_frame(offsets, 2)
        # Should handle missing next residue gracefully
        assert isinstance(frame_info_2, dict)
        
        # Test middle frame (frame_index=1)
        frame_info_1 = compute_frame(offsets, 1)
        assert isinstance(frame_info_1, dict)
    
    def test_frame_computation_with_nans(self, dipeptide_angles_and_offsets):
        """Test frame computation when some offsets are NaN"""
        from reconstruct.reconstruct import compute_frame
        
        offsets, expected_output = dipeptide_angles_and_offsets
        # Introduce NaN values in the offsets
        offsets[0, 2] = np.array([np.nan, np.nan, np.nan])  # Set nitrogen_i->CA_i to NaN
        
        frame_info = compute_frame(offsets, 0)
        
        # Should handle NaN gracefully and not propagate to all results
        assert not all(v is None for v in frame_info.values()), "Frame computation failed with NaN inputs"
    
    def test_frame_computation_vector_averaging(self, dipeptide_averaging_data):
        """Test that vector averaging in frame computation is sensible"""
        from reconstruct.reconstruct import integrate_neighbour_information

        offsets, expected_output = dipeptide_averaging_data
        neighbour_info = integrate_neighbour_information(offsets, 2)

        # Check that the averaged vectors are correct
        for i, info in enumerate(expected_output):
            for key in info:
                if info[key] is not None:
                    assert abs(neighbour_info[i][key] - info[key]) < 1e-3, f"Angle {key} mismatch at residue {i}: expected {info[key]}, got {neighbour_info[i][key]}"
                else:
                    assert neighbour_info[i][key] is None, f"Expected None for {key} at residue {i}, got {neighbour_info[i][key]}"

    # ===== TESTS FOR COORDINATE TRANSFORMATION BUGS =====
    
    def test_coordinate_space_consistency(self, polypeptide_test_data):
        """Test that coordinates remain consistent through transformations"""
        alpha_carbon_envs, alpha_carbon_elem_types, sequence, structure = polypeptide_test_data
        
        # Check that alpha carbon environments are relative to their own alpha carbon
        for i in range(len(sequence)):
            ca_indices = np.where(alpha_carbon_elem_types[i] == constants.elem.index('C'))[0]
            if len(ca_indices) > 0:
                ca_pos = alpha_carbon_envs[i, ca_indices[0]]  # First carbon (should be CA)
                ca_distance = np.linalg.norm(ca_pos)
                assert ca_distance < 0.5, f"Alpha carbon not at origin in frame {i}: distance = {ca_distance}"
    
    def test_world_space_transformation(self):
        """Test transformation from local to world coordinates"""
        # This tests the make_atom_cloud_from_envs function
        
        # Create a simple mock structure to test coordinate transformation
        from Bio.PDB import Structure, Model, Chain, Residue, Atom
        
        # Mock a simple structure
        structure = Structure.Structure("test")
        model = Model.Model(0)
        chain = Chain.Chain("A")
        
        # Add a few residues with CA atoms
        for i in range(3):
            residue = Residue.Residue(("", i+1, ""), "ALA", "")
            ca_atom = Atom.Atom("CA", [float(i*3.8), 0.0, 0.0], 0, 0, "", "CA", 0, "C")
            residue.add(ca_atom)
            chain.add(residue)
        
        model.add(chain)
        structure.add(model)
        
        # Test coordinates
        alpha_carbon_envs = np.array([
            [[0, 0, 0], [1, 0, 0]],  # Frame 0: CA at origin, another atom offset
            [[0, 0, 0], [0, 1, 0]],  # Frame 1: CA at origin, another atom offset  
            [[0, 0, 0], [0, 0, 1]]   # Frame 2: CA at origin, another atom offset
        ])
        alpha_carbon_elem_types = np.array([
            [constants.elem.index('C'), constants.elem.index('N')],
            [constants.elem.index('C'), constants.elem.index('N')],
            [constants.elem.index('C'), constants.elem.index('N')]
        ])
        
        positions, types, alpha_offsets = reconstruct.make_atom_cloud_from_envs(
            alpha_carbon_envs, alpha_carbon_elem_types, structure
        )
        
        # Check that world coordinates make sense
        assert len(positions) == 6, f"Expected 6 atoms, got {len(positions)}"
        assert len(types) == 6, f"Expected 6 atom types, got {len(types)}"
        
        # Check that alpha carbon offsets are as expected
        expected_ca_positions = np.array([[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]])
        np.testing.assert_allclose(alpha_offsets, expected_ca_positions, atol=1e-6)


    # ===== TESTS FOR BOND LENGTH AND ANGLE VALIDATION =====
    
    def test_bond_length_validation(self, polypeptide_test_data):
        """Test that computed bond lengths are chemically reasonable"""
        alpha_carbon_envs, alpha_carbon_elem_types, sequence, structure = polypeptide_test_data
        
        offsets = assign_backbone.assign_backbone_atoms(alpha_carbon_envs, alpha_carbon_elem_types)
        neighbour_info = reconstruct.integrate_neighbour_information(offsets, len(sequence))
        
        for i, info in enumerate(neighbour_info):
            # Check peptide bond length
            if info['peptide_bond'] is not None:
                assert 1.2 < info['peptide_bond'] < 1.5, f"Peptide bond {info['peptide_bond']} Å unrealistic at residue {i}"
            
            # Check CA-N distance
            if info['CA_N_length'] is not None:
                assert 1.2 < info['CA_N_length'] < 1.6, f"CA-N distance {info['CA_N_length']} Å unrealistic at residue {i}"
            
            # Check CA-C distance
            if info['CA_C_length'] is not None:
                assert 1.3 < info['CA_C_length'] < 1.7, f"CA-C distance {info['CA_C_length']} Å unrealistic at residue {i}"
    
    def test_bond_angle_validation(self, polypeptide_test_data):
        """Test that computed bond angles are chemically reasonable"""
        alpha_carbon_envs, alpha_carbon_elem_types, sequence, structure = polypeptide_test_data
        
        offsets = assign_backbone.assign_backbone_atoms(alpha_carbon_envs, alpha_carbon_elem_types)
        neighbour_info = reconstruct.integrate_neighbour_information(offsets, len(sequence))
        
        for i, info in enumerate(neighbour_info):
            # We'll only check the omega angle, since it has very clear chemical constraints
            if info['omega'] is not None:
                angle = info['omega']
                is_trans = 170 < abs(angle) < 190
                is_cis = -10 < angle < 10
                assert is_trans or is_cis, f"Omega angle {angle}° unrealistic at residue {i}: expected trans (+/- 170-190°) or cis (-10 to 10°) conformation"


    # ===== TESTS FOR PEPTIDE BUILDER INTEGRATION =====
    
    def test_peptide_builder_geometry_creation(self, polypeptide_test_data):
        """Test integration with PeptideBuilder"""
        alpha_carbon_envs, alpha_carbon_elem_types, sequence, structure = polypeptide_test_data
        
        offsets = assign_backbone.assign_backbone_atoms(alpha_carbon_envs, alpha_carbon_elem_types)
        neighbour_info = reconstruct.integrate_neighbour_information(offsets, len(sequence))
        geos = reconstruct.fit_backbone(neighbour_info, sequence)
        
        assert len(geos) == len(sequence), f"Expected {len(sequence)} geometries, got {len(geos)}"
        
        # Check that geometry objects have reasonable values
        for i, geo in enumerate(geos):
            # Check that critical angles are not extreme
            if hasattr(geo, 'omega') and geo.omega is not None:
                assert abs(geo.omega) < 190, f"PeptideBuilder omega {geo.omega}° is extreme at residue {i}"
            
            if hasattr(geo, 'phi') and geo.phi is not None:
                assert -190 < geo.phi < 190, f"PeptideBuilder phi {geo.phi}° is extreme at residue {i}"
            
            if hasattr(geo, 'psi_im1') and geo.psi_im1 is not None:
                assert -190 < geo.psi_im1 < 190, f"PeptideBuilder psi_im1 {geo.psi_im1}° is extreme at residue {i}"


    # ===== INTEGRATION TESTS =====
    
    def test_full_reconstruction_pipeline_basic(self, simple_test_data):
        """Test the full reconstruction pipeline doesn't crash"""
        alpha_carbon_envs, alpha_carbon_elem_types, sequence, structure = simple_test_data
        
        # This tests the main reconstruct_structure function
        try:
            # We need to relax numerical error handling here because peptidebuilder intentionally raises + catches numerical warnings
            with np.errstate(all="ignore"):
                structure = reconstruct.reconstruct_structure(
                    alpha_carbon_envs, alpha_carbon_elem_types, sequence
                )
            assert structure is not None, "Reconstruction returned None"
        except Exception as e:
            pytest.fail(f"Full reconstruction pipeline crashed: {e}")
    
    def test_reconstruction_with_realistic_noise(self, polypeptide_test_data):
        """Test reconstruction with added noise (simulates real conditions)"""
        alpha_carbon_envs, alpha_carbon_elem_types, sequence, structure = polypeptide_test_data
        
        # Add some noise to simulate real conditions
        noise_level = 0.2  # 0.2 Å noise
        noisy_envs = alpha_carbon_envs + np.random.normal(0, noise_level, alpha_carbon_envs.shape)
        
        # We need to relax numerical error handling here because peptidebuilder intentionally raises + catches numerical warnings
        with np.errstate(all="ignore"):
            structure = reconstruct.reconstruct_structure(
                noisy_envs, alpha_carbon_elem_types, sequence
            )
        assert structure is not None, "Reconstruction with noise returned None"
        # Any errors raised during reconstruction should fail the test


    # ===== SPECIAL BUG-HUNTING TESTS =====
    
    def test_atomic_spaghetti_detection(self, polypeptide_test_data):
        """Test to detect 'atomic spaghetti' - unrealistic structure geometry"""
        alpha_carbon_envs, alpha_carbon_elem_types, sequence, structure = polypeptide_test_data
        
        try:
            # We need to relax numerical error handling here because peptidebuilder intentionally raises + catches numerical warnings
            with np.errstate(all="ignore"):
                structure = reconstruct.reconstruct_structure(
                    alpha_carbon_envs, alpha_carbon_elem_types, sequence
                )
            
            if structure is not None:
                # Extract coordinates and check for unrealistic geometries
                atoms = list(structure.get_atoms())
                if len(atoms) > 1:
                    coords = np.array([atom.get_coord() for atom in atoms])
                    
                    # Check for extreme coordinate values
                    assert np.all(np.abs(coords) < 1000), "Extreme coordinate values detected (atomic spaghetti)"
                    
                    # Check for unrealistic inter-atomic distances
                    for i in range(len(coords)):
                        for j in range(i+1, len(coords)):
                            dist = np.linalg.norm(coords[i] - coords[j])
                            assert dist < 100, f"Unrealistic inter-atomic distance: {dist} Å between atoms {i} and {j}"
                    
                    # Check for reasonable CA-CA distances
                    ca_coords = []
                    for atom in atoms:
                        if atom.get_name() == "CA":
                            ca_coords.append(atom.get_coord())
                    
                    if len(ca_coords) > 1:
                        ca_coords = np.array(ca_coords)
                        for i in range(len(ca_coords)-1):
                            ca_dist = np.linalg.norm(ca_coords[i+1] - ca_coords[i])
                            assert 2.0 < ca_dist < 6.0, f"Unrealistic CA-CA distance: {ca_dist} Å"
        
        except Exception as e:
            pytest.fail(f"Structure reconstruction or validation crashed: {e}")


    # ===== DEBUGGING HELPER TESTS =====
    
    def test_debug_omega_calculation_step_by_step(self, polypeptide_test_data):
        """Detailed debugging test for omega angle calculation"""
        alpha_carbon_envs, alpha_carbon_elem_types, sequence, structure = polypeptide_test_data
        
        print("\n=== DEBUGGING OMEGA CALCULATION ===")
        
        offsets = assign_backbone.assign_backbone_atoms(alpha_carbon_envs, alpha_carbon_elem_types)
        
        print(f"Offsets shape: {offsets.shape}")
        print(f"Sequence length: {len(sequence)}")
        
        for i in range(len(sequence)):
            print(f"\nFrame {i}:")
            frame_offsets = offsets[i]
            
            print(f"  CA_i-1 -> C_i-1: {frame_offsets[0]}")
            print(f"  C_i-1 -> N_i:    {frame_offsets[1]}")  
            print(f"  N_i -> CA_i:     {frame_offsets[2]}")
            print(f"  CA_i -> C_i:     {frame_offsets[3]}")
            print(f"  C_i -> N_i+1:    {frame_offsets[4]}")
            print(f"  N_i+1 -> CA_i+1: {frame_offsets[5]}")
            
            # Check for NaN values that might propagate to omega calculation
            nan_count = np.sum(np.isnan(frame_offsets))
            print(f"  NaN values: {nan_count}/18")
            
        neighbour_info = reconstruct.integrate_neighbour_information(offsets, len(sequence))
        
        for i, info in enumerate(neighbour_info):
            print(f"\nResidue {i} angles:")
            for key, value in info.items():
                print(f"  {key}: {value}")
                
        # This test always passes - it's just for debugging output
        assert True


if __name__ == "__main__":
    # Run tests with verbose output for debugging
    pytest.main([__file__, "-v", "--tb=short", "-s"])
