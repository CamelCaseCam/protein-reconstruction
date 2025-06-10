#!/usr/bin/env python3
"""
Focused test suite specifically for debugging omega angle calculation bugs.
This addresses the specific issue with omega angles being "all over the place"
including extreme values like 45 degrees.
"""

import numpy as np
import sys
import os

# Add the reconstruct module to path
sys.path.insert(0, '/tmp/inputs/protein-reconstruction')

from reconstruct import reconstruct, assign_backbone, constants
from get_calpha_env import extract_alpha_carbon_envs

import PeptideBuilder
import PeptideBuilder.Geometry as geo


def create_ideal_extended_peptide():
    """Create an ideal extended peptide for testing omega calculations"""
    # Extended peptide with ideal geometry
    sequence = "AAAA"  # 4 alanines for testing omega between residues
    
    with np.errstate(all='ignore'):
        structure = PeptideBuilder.make_extended_structure(sequence)
    alpha_carbon_envs, alpha_carbon_elem_types, reconst_seq = extract_alpha_carbon_envs(structure)
    assert reconst_seq == sequence, f"Reconstructed sequence {reconst_seq} does not match expected {sequence}"
    
    return alpha_carbon_envs, alpha_carbon_elem_types, sequence


def test_omega_ideal_extended():
    """Test omega calculation with ideal extended peptide (should give ~180°)"""
    print("\n=== TESTING OMEGA ANGLES WITH IDEAL EXTENDED PEPTIDE ===")
    
    alpha_carbon_envs, alpha_carbon_elem_types, sequence = create_ideal_extended_peptide()
    
    # Run through the pipeline
    offsets = assign_backbone.assign_backbone_atoms(alpha_carbon_envs, alpha_carbon_elem_types)
    neighbour_info = reconstruct.integrate_neighbour_information(offsets, len(sequence))
    
    print(f"Sequence: {sequence}")
    print(f"Length: {len(sequence)}")
    
    omega_angles = []
    
    for i, info in enumerate(neighbour_info):
        omega = info['omega']
        print(f"\nResidue {i} ({sequence[i]}):")
        print(f"  omega = {omega}°")
        
        if omega is not None:
            omega_angles.append(omega)
            
            # For an ideal extended peptide, omega should be close to 180°
            if abs(abs(omega) - 180) > 20:
                print(f"  ⚠️  WARNING: Omega {omega}° deviates significantly from 180°!")
                
            if abs(omega) > 200:
                print(f"  🚨 ERROR: Extreme omega angle {omega}°!")
                
            if 40 < abs(omega) < 50:
                print(f"  🚨 ERROR: Suspicious omega near 45°: {omega}°!")
    
    print(f"\n--- Omega Angle Summary ---")
    print(f"Number of omega angles calculated: {len(omega_angles)}")
    if omega_angles:
        print(f"Mean omega: {np.mean(omega_angles):.1f}°")
        print(f"Std omega: {np.std(omega_angles):.1f}°")
        print(f"Range: {np.min(omega_angles):.1f}° to {np.max(omega_angles):.1f}°")
        
        # Check for problematic values
        extreme_count = sum(1 for omega in omega_angles if abs(omega) > 200)
        suspicious_45_count = sum(1 for omega in omega_angles if 40 < abs(omega) < 50)
        
        print(f"Extreme angles (>200°): {extreme_count}")
        print(f"Suspicious ~45° angles: {suspicious_45_count}")
        
        return extreme_count == 0 and suspicious_45_count == 0
    
    return True


def debug_omega_calculation_step_by_step():
    """Detailed step-by-step debugging of omega calculation"""
    print("\n=== STEP-BY-STEP OMEGA CALCULATION DEBUGGING ===")
    
    alpha_carbon_envs, alpha_carbon_elem_types, sequence = create_ideal_extended_peptide()
    
    print("1. Testing atom assignment...")
    offsets = assign_backbone.assign_backbone_atoms(alpha_carbon_envs, alpha_carbon_elem_types)
    
    print(f"   Offsets shape: {offsets.shape}")
    
    # Focus on the middle residue (index 1) which should have good omega calculation
    focus_residue = 1
    print(f"\n2. Analyzing residue {focus_residue} offsets:")
    
    offset_names = [
        "CA_i-1 -> C_i-1",
        "C_i-1 -> N_i", 
        "N_i -> CA_i",
        "CA_i -> C_i",
        "C_i -> N_i+1",
        "N_i+1 -> CA_i+1"
    ]
    
    for j, (name, offset) in enumerate(zip(offset_names, offsets[focus_residue])):
        length = np.linalg.norm(offset) if not np.isnan(offset).any() else "NaN"
        print(f"   {j}: {name:<15} = {offset} (length: {length})")
    
    print(f"\n3. Testing frame computation...")
    frame_info = reconstruct.compute_frame(offsets, focus_residue)
    
    print(f"   Frame {focus_residue} computed values:")
    for key, value in frame_info.items():
        print(f"     {key}: {value}")
    
    print(f"\n4. Manual omega calculation verification...")
    
    # Extract vectors needed for omega calculation
    # Omega = dihedral(CA_C_vec, C_Np1, np1_CAp1)
    CA_C_vec = offsets[focus_residue, 3]  # CA_i -> C_i
    C_Np1 = offsets[focus_residue, 4]    # C_i -> N_i+1
    np1_CAp1 = offsets[focus_residue, 5] # N_i+1 -> CA_i+1
    
    print(f"   CA_C_vec (CA->C): {CA_C_vec}")
    print(f"   C_Np1 (C->N+1):   {C_Np1}")
    print(f"   np1_CAp1 (N+1->CA+1): {np1_CAp1}")
    
    # Check if any vectors are problematic
    vectors_ok = True
    for name, vec in [("CA_C_vec", CA_C_vec), ("C_Np1", C_Np1), ("np1_CAp1", np1_CAp1)]:
        if np.isnan(vec).any():
            print(f"   🚨 {name} contains NaN!")
            vectors_ok = False
        elif np.linalg.norm(vec) < 1e-6:
            print(f"   🚨 {name} is too small: {np.linalg.norm(vec)}")
            vectors_ok = False
        elif np.linalg.norm(vec) > 10:
            print(f"   🚨 {name} is too large: {np.linalg.norm(vec)}")
            vectors_ok = False
    
    if vectors_ok:
        # Manual dihedral calculation
        try:
            manual_dihedral = reconstruct.dihedral(CA_C_vec, C_Np1, np1_CAp1)
            manual_omega = reconstruct.rad2deg(manual_dihedral)
            print(f"   Manual omega calculation: {manual_omega}°")
            
            computed_omega = frame_info['omega']
            print(f"   Computed omega: {computed_omega}°")
            
            if abs(manual_omega - computed_omega) > 1e-6:
                print(f"   🚨 Mismatch between manual and computed omega!")
                
        except Exception as e:
            print(f"   🚨 Manual dihedral calculation failed: {e}")
    
    print(f"\n5. Testing dihedral calculation robustness...")
    test_dihedral_calculation()


def test_dihedral_calculation():
    """Test the dihedral calculation function for edge cases"""
    from reconstruct.reconstruct import dihedral, rad2deg
    
    print("   Testing known dihedral cases...")
    
    # Test case 1: 0° dihedral
    b1 = np.array([1, 0, 0])
    b2 = np.array([0, 1, 0])
    b3 = np.array([0, 0, 1])
    angle = rad2deg(dihedral(b1, b2, b3))
    print(f"     0° case: {angle:.1f}°")
    
    # Test case 2: 180° dihedral
    b1 = np.array([1, 0, 0])
    b2 = np.array([0, 1, 0])
    b3 = np.array([-1, 0, 0])
    angle = rad2deg(dihedral(b1, b2, b3))
    print(f"     180° case: {angle:.1f}°")
    
    # Test case 3: 90° dihedral
    b1 = np.array([1, 0, 0])
    b2 = np.array([0, 1, 0])
    b3 = np.array([0, 1, 1])
    angle = rad2deg(dihedral(b1, b2, b3))
    print(f"     90° case: {angle:.1f}°")
    
    # Test problematic cases
    print("   Testing problematic cases...")
    
    # Parallel vectors
    try:
        b1 = np.array([1, 0, 0])
        b2 = np.array([1, 0, 0])  # Parallel to b1
        b3 = np.array([1, 0, 0])  # Parallel to b1 and b2
        angle = rad2deg(dihedral(b1, b2, b3))
        print(f"     Parallel vectors: {angle:.1f}°")
        if abs(angle) > 360:
            print(f"     🚨 Extreme angle from parallel vectors!")
    except Exception as e:
        print(f"     Parallel vectors failed: {e}")
    
    # Very small vectors
    try:
        scale = 1e-8
        b1 = np.array([scale, 0, 0])
        b2 = np.array([0, scale, 0])
        b3 = np.array([0, 0, scale])
        angle = rad2deg(dihedral(b1, b2, b3))
        print(f"     Tiny vectors: {angle:.1f}°")
        if abs(angle) > 360:
            print(f"     🚨 Extreme angle from tiny vectors!")
    except Exception as e:
        print(f"     Tiny vectors failed: {e}")

def main():
    """Run all omega angle debugging tests"""
    print("🔍 OMEGA ANGLE BUG HUNTING")
    print("="*50)
    
    # Test 1: Ideal case
    ideal_passed = test_omega_ideal_extended()
    
    # Test 2: Detailed debugging
    debug_omega_calculation_step_by_step()

if __name__ == "__main__":
    main()
