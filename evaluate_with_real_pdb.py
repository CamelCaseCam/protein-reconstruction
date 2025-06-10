#!/usr/bin/env python3
"""
Test the backbone reconstruction using a real PDB file to see if we can reproduce
the omega angle bug and atomic spaghetti issues.
"""

import numpy as np
import sys
import os
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings("ignore")

# Add the reconstruct module to path
sys.path.insert(0, '/tmp/inputs/protein-reconstruction')

from reconstruct import reconstruct, assign_backbone, constants


def extract_environment_from_pdb(pdb_file, max_atoms_per_frame=20, cutoff_distance=5.0):
    """
    Extract alpha carbon environments from a real PDB file
    This simulates what your algorithm would receive as input
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('test', pdb_file)
    
    # Get first model and chain
    model = structure[0]
    chain = next(iter(model))
    
    # Get CA atoms and sequence
    ca_atoms = []
    sequence = ""
    
    for residue in chain:
        if residue.has_id('CA'):
            ca_atom = residue['CA']
            ca_atoms.append(ca_atom)
            # Convert 3-letter to 1-letter amino acid code
            aa_map = {
                'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
            }
            aa_3letter = residue.get_resname()
            sequence += aa_map.get(aa_3letter, 'A')  # Default to alanine if unknown
    
    L = len(ca_atoms)
    alpha_carbon_envs = np.zeros((L, max_atoms_per_frame, 3))
    alpha_carbon_elem_types = np.full((L, max_atoms_per_frame), constants.PAD_ID)
    
    # For each CA, find nearby atoms
    for i, ca_atom in enumerate(ca_atoms):
        ca_coord = ca_atom.get_coord()
        nearby_atoms = []
        
        # Find all atoms within cutoff distance
        for atom in structure.get_atoms():
            if atom.get_parent().get_parent() == chain:  # Same chain
                dist = np.linalg.norm(atom.get_coord() - ca_coord)
                if dist <= cutoff_distance:
                    element = atom.element.upper()
                    if element in constants.elem:
                        nearby_atoms.append((atom.get_coord() - ca_coord, element))
        
        # Sort by distance and take the closest ones
        nearby_atoms.sort(key=lambda x: np.linalg.norm(x[0]))
        
        for j, (relative_coord, element) in enumerate(nearby_atoms[:max_atoms_per_frame]):
            alpha_carbon_envs[i, j] = relative_coord
            alpha_carbon_elem_types[i, j] = constants.elem.index(element)
    
    return alpha_carbon_envs, alpha_carbon_elem_types, sequence


def test_real_pdb_reconstruction():
    """Test reconstruction with the provided PDB file"""
    pdb_file = 'testing/9nwz.pdb'
    
    if not os.path.exists(pdb_file):
        print(f"❌ PDB file not found: {pdb_file}")
        return False
    
    print(f"🧬 Testing with real PDB: {pdb_file}")
    print("="*60)
    
    try:
        # Extract environment data
        print("1. Extracting alpha carbon environments from PDB...")
        alpha_carbon_envs, alpha_carbon_elem_types, sequence = extract_environment_from_pdb(
            pdb_file, max_atoms_per_frame=30, cutoff_distance=5.0
        )
        
        print(f"   Sequence length: {len(sequence)}")
        print(f"   Environment shape: {alpha_carbon_envs.shape}")
        print(f"   Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
        
        # Test reconstruction
        print("\n2. Running backbone reconstruction...")
        
        # First test atom assignment
        offsets = assign_backbone.assign_backbone_atoms(alpha_carbon_envs, alpha_carbon_elem_types)
        print(f"   Offset shape: {offsets.shape}")
        
        # Count non-NaN offsets
        non_nan_count = np.sum(~np.isnan(offsets))
        total_count = np.prod(offsets.shape)
        print(f"   Non-NaN offsets: {non_nan_count}/{total_count} ({100*non_nan_count/total_count:.1f}%)")
        
        # Check neighbor info and look for omega issues
        print("\n3. Computing backbone angles...")
        neighbour_info = reconstruct.integrate_neighbour_information(offsets, len(sequence))
        
        omega_angles = []
        phi_angles = []
        psi_angles = []
        extreme_omega_residues = []
        suspicious_45_residues = []
        
        for i, info in enumerate(neighbour_info):
            omega = info['omega']
            phi = info['phi'] 
            psi = info['psi']
            
            if omega is not None:
                omega_angles.append(omega)
                
                # Check for the specific bugs mentioned
                if abs(omega) > 200:
                    extreme_omega_residues.append((i, omega))
                    
                if 40 <= abs(omega) <= 50:  # The "45 degree" bug mentioned
                    suspicious_45_residues.append((i, omega))
            
            if phi is not None:
                phi_angles.append(phi)
            if psi is not None:
                psi_angles.append(psi)
        
        # Report results
        print(f"\n4. Backbone angle analysis:")
        print(f"   Omega angles calculated: {len(omega_angles)}/{len(sequence)}")
        print(f"   Phi angles calculated: {len(phi_angles)}/{len(sequence)}")
        print(f"   Psi angles calculated: {len(psi_angles)}/{len(sequence)}")
        
        if omega_angles:
            print(f"\n   Omega angle statistics:")
            print(f"     Mean: {np.mean(omega_angles):.1f}°")
            print(f"     Std:  {np.std(omega_angles):.1f}°")
            print(f"     Min:  {np.min(omega_angles):.1f}°")
            print(f"     Max:  {np.max(omega_angles):.1f}°")
            
            # Check for issues
            if extreme_omega_residues:
                print(f"\n   🚨 EXTREME OMEGA ANGLES DETECTED:")
                for res_i, omega in extreme_omega_residues:
                    print(f"     Residue {res_i}: ω = {omega:.1f}°")
            
            if suspicious_45_residues:
                print(f"\n   🚨 SUSPICIOUS ~45° OMEGA ANGLES DETECTED:")
                for res_i, omega in suspicious_45_residues:
                    print(f"     Residue {res_i}: ω = {omega:.1f}°")
                print(f"   This matches the reported bug!")
            
            # Test full reconstruction
            print(f"\n5. Testing full structure reconstruction...")
            try:
                structure = reconstruct.reconstruct_structure(
                    alpha_carbon_envs, alpha_carbon_elem_types, sequence
                )
                
                if structure is not None:
                    print(f"   ✅ Reconstruction completed without crashing")
                    
                    # Check for "atomic spaghetti"
                    atoms = list(structure.get_atoms())
                    if len(atoms) > 0:
                        coords = np.array([atom.get_coord() for atom in atoms])
                        
                        # Check coordinate ranges
                        coord_ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
                        max_coord = np.max(np.abs(coords))
                        
                        print(f"   Structure coordinate ranges: {coord_ranges}")
                        print(f"   Maximum coordinate value: {max_coord:.1f}")
                        
                        if max_coord > 1000:
                            print(f"   🚨 ATOMIC SPAGHETTI DETECTED: Extreme coordinates!")
                        
                        # Check CA-CA distances
                        ca_coords = []
                        for atom in atoms:
                            if atom.get_name() == "CA":
                                ca_coords.append(atom.get_coord())
                        
                        if len(ca_coords) > 1:
                            ca_coords = np.array(ca_coords)
                            ca_distances = []
                            for i in range(len(ca_coords)-1):
                                dist = np.linalg.norm(ca_coords[i+1] - ca_coords[i])
                                ca_distances.append(dist)
                            
                            print(f"   CA-CA distances: {np.min(ca_distances):.1f} - {np.max(ca_distances):.1f} Å")
                            
                            extreme_ca_distances = [d for d in ca_distances if d > 10 or d < 1]
                            if extreme_ca_distances:
                                print(f"   🚨 ATOMIC SPAGHETTI: Extreme CA-CA distances detected!")
                                print(f"     {len(extreme_ca_distances)} bonds with extreme distances")
                    
                else:
                    print(f"   ❌ Reconstruction returned None")
                    
            except Exception as e:
                print(f"   ❌ Reconstruction crashed: {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the real PDB test"""
    print("🧬 REAL PDB BACKBONE RECONSTRUCTION TEST")
    print("="*60)
    print("This test uses a real PDB file to try to reproduce")
    print("the omega angle bugs and atomic spaghetti issues.")
    print()
    
    success = test_real_pdb_reconstruction()
    
    print("\n" + "="*60)
    if success:
        print("✅ Real PDB test completed")
        print("Check the output above for detected bugs:")
        print("- Extreme omega angles (>200°)")
        print("- Suspicious ~45° omega angles") 
        print("- Atomic spaghetti (extreme coordinates)")
        print("- Unrealistic CA-CA distances")
    else:
        print("❌ Real PDB test failed")
        print("The reconstruction pipeline may have fundamental issues.")
    
    print("\nThis test helps identify if the bugs occur with real data")
    print("and can guide debugging efforts to the specific problem areas.")


if __name__ == "__main__":
    main()
