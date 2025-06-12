# test_reconstruction.py
'''
Test script for protein structure reconstruction.
'''

import numpy as np
from Bio.PDB import PDBParser, PDBIO
from reconstruct.reconstruct import reconstruct_structure
from reconstruct.optimize_rotamers import THREE_LETTER_TO_ONE_LETTER
from reconstruct.constants import PAD_ID, elem
import os

def extract_alpha_carbon_envs(pdb_file, max_atoms_per_env=100):
    """
    Extract alpha carbon environments from a PDB file.
    
    Args:
        pdb_file: Path to the PDB file
        max_atoms_per_env: Maximum number of atoms to include in each environment
        
    Returns:
        tuple: (alpha_carbon_envs, alpha_carbon_elem_types, sequence)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    # Extract alpha carbons
    alpha_carbons = []
    sequence = []
    
    model = structure[0]  # We only want to take the first model, since NMR structures could have multiple models
    for chain in model:
        for residue in chain:
            if residue.has_id("CA"):
                alpha_carbons.append(residue["CA"])
                sequence.append(residue.get_resname())
                
    sequence = ''.join([THREE_LETTER_TO_ONE_LETTER.get(res, 'X') for res in sequence])  # Convert to one-letter code, default to 'X' for unknown

    # Extract environments
    alpha_carbon_envs = []
    alpha_carbon_elem_types = []
    all_atoms = []
    atom_positions = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element != "H":  # Skip hydrogens
                        all_atoms.append(atom)
                        atom_positions.append(atom.get_coord())
    atom_positions = np.array(atom_positions)
    
    for ca in alpha_carbons:
        # Get all atoms within 5A of the alpha carbon
        ca_pos = ca.get_coord()
        env_atoms = []
        
        atoms_within_distance = np.where(np.linalg.norm(atom_positions - ca_pos, axis=1) < 5.0)[0]
        env_atom_indices = np.argsort(np.linalg.norm(atom_positions[atoms_within_distance] - ca_pos, axis=1))[:max_atoms_per_env]
        env_atoms = [(all_atoms[i], i) for i in atoms_within_distance[env_atom_indices]]
        
        # Extract positions and element types
        env_positions = np.zeros((max_atoms_per_env, 3))
        env_elem_types = np.full(max_atoms_per_env, PAD_ID)
        
        for i, (atom, posindex) in enumerate(env_atoms):
            if i >= max_atoms_per_env:
                break
            env_positions[i] = atom_positions[posindex] - ca_pos  # Relative to alpha carbon in angstroms
            elem_type = atom.element.upper()
            if elem_type in elem:
                env_elem_types[i] = elem.index(elem_type)
        
        alpha_carbon_envs.append(env_positions)
        alpha_carbon_elem_types.append(env_elem_types)
    
    return np.array(alpha_carbon_envs), np.array(alpha_carbon_elem_types), sequence

def main():
    # Path to test PDB file
    test_pdb = "testing/2spz.pdb"
    output_pdb = "reconstructed_2spz.pdb"
    
    # Extract alpha carbon environments
    print(f"Extracting alpha carbon environments from {test_pdb}...")
    alpha_carbon_envs, alpha_carbon_elem_types, sequence = extract_alpha_carbon_envs(test_pdb)
    
    # Reconstruct structure
    print(f"Reconstructing structure for sequence of length {len(sequence)}...")
    reconstructed_structure = reconstruct_structure(alpha_carbon_envs, alpha_carbon_elem_types, sequence)
    
    # Save reconstructed structure
    io = PDBIO()
    io.set_structure(reconstructed_structure)
    io.save(output_pdb)
    print(f"Reconstructed structure saved to {output_pdb}")

if __name__ == "__main__":
    main()