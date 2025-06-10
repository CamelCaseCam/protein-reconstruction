# reconstruct/optimize_rotamers.py (updated with implementation details)
'''
Utility for optimizing side chain rotamers during protein structure reconstruction.
'''

import numpy as np
from Bio.PDB import Structure, Model, Chain, Residue, Atom
import PeptideBuilder
from PeptideBuilder import Geometry
import os
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree

# Mapping from 3-letter to 1-letter amino acid codes
THREE_LETTER_TO_ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

# Reverse mapping for quick lookup
ONE_LETTER_TO_THREE_LETTER = {v: k for k, v in THREE_LETTER_TO_ONE_LETTER.items()}

def compute_residue_cost(
        residue: Geometry.Geo,
        res_alpha_pos: np.ndarray,
        atom_cloud: np.ndarray,
        atom_kd_trees: Dict[int, cKDTree],
        element_map: Dict[str, int]
    ) -> float:
    """
    Compute the cost (MSE) of a residue by finding the nearest atom of each type in the global point cloud.
    
    Args:
        residue_atoms: List of (atom_name, position) tuples for the residue
        atom_cloud: Global point cloud of atom positions
        atom_types: Element types of atoms in the global point cloud
        element_map: Mapping from element names to indices
        
    Returns:
        float: Mean squared error between residue atoms and their nearest matching atoms in the cloud
    """
    cost = 0.0
    # Convert the residue to Biopython so we can use its methods
    bio_residue_structure = PeptideBuilder.make_structure_from_geos([residue])
    # Get the residue
    bio_residue = bio_residue_structure[0]["A"][(' ', 1, ' ')]
    # Get the calpha position
    bio_alpha_pos = bio_residue["CA"].get_coord()
    pos_offset = res_alpha_pos - bio_alpha_pos

    # Iterate over each atom in the residue
    num = 0
    for atom in bio_residue.get_atoms():
        # Exclude backbone atoms
        if atom.name in ["N", "CA", "C", "O"]:
            continue
        # Get the atom type
        atom_type = atom.element
        if atom_type == "H":
            continue
        atom_index = element_map.get(atom_type, None)
        if atom_index is None:
            raise ValueError(f"Unknown atom type: {atom_type}")
        # Get the position of the atom
        atom_pos = atom.get_coord() + pos_offset
        # Find the nearest atom in the cloud of the same type
        if atom_index in atom_kd_trees:
            kd_tree = atom_kd_trees[atom_index]
            nearest_index = kd_tree.query(atom_pos)[1]
            nearest_atom_pos = atom_cloud[nearest_index]
            # Compute the squared distance
            dist_sq = np.sum((atom_pos - nearest_atom_pos) ** 2)
            cost += dist_sq
            num += 1
    return cost / num if num > 0 else 0
    

# Enum for whether it's a scalar or sin'd angle
from enum import Enum
class Dtype(Enum):
    SCALAR = 1  # Scalar value
    ANGLE = 2    # Sine of the angle

# Rotamer index/variable name mapping - we'll use reflection to access these dynamically
ROTAMER_VARS = {
    "A": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE)],
    "S": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_OG_length", Dtype.SCALAR), ("CA_CB_OG_angle", Dtype.ANGLE), ("N_CA_CB_OG_diangle", Dtype.ANGLE)],
    "C": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_SG_length", Dtype.SCALAR), ("CA_CB_SG_angle", Dtype.ANGLE), ("N_CA_CB_SG_diangle", Dtype.ANGLE)],
    "V": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG1_length", Dtype.SCALAR), ("CA_CB_CG1_angle", Dtype.ANGLE), ("N_CA_CB_CG1_diangle", Dtype.ANGLE), 
            ("CB_CG2_length", Dtype.SCALAR), ("CA_CB_CG2_angle", Dtype.ANGLE), ("N_CA_CB_CG2_diangle", Dtype.ANGLE)],
    "I": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG1_length", Dtype.SCALAR), ("CA_CB_CG1_angle", Dtype.ANGLE), ("N_CA_CB_CG1_diangle", Dtype.ANGLE), 
            ("CB_CG2_length", Dtype.SCALAR), ("CA_CB_CG2_angle", Dtype.ANGLE), ("N_CA_CB_CG2_diangle", Dtype.ANGLE), 
            ("CG1_CD1_length", Dtype.SCALAR), ("CB_CG1_CD1_angle", Dtype.ANGLE), ("CA_CB_CG1_CD1_diangle", Dtype.ANGLE)],
    "L": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_CD1_length", Dtype.SCALAR), ("CB_CG_CD1_angle", Dtype.ANGLE), ("CA_CB_CG_CD1_diangle", Dtype.ANGLE), 
            ("CG_CD2_length", Dtype.SCALAR), ("CB_CG_CD2_angle", Dtype.ANGLE), ("CA_CB_CG_CD2_diangle", Dtype.ANGLE)],
    "T": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_OG1_length", Dtype.SCALAR), ("CA_CB_OG1_angle", Dtype.ANGLE), ("N_CA_CB_OG1_diangle", Dtype.ANGLE), 
            ("CB_CG2_length", Dtype.SCALAR), ("CA_CB_CG2_angle", Dtype.ANGLE), ("N_CA_CB_CG2_diangle", Dtype.ANGLE)],
    "R": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_CD_length", Dtype.SCALAR), ("CB_CG_CD_angle", Dtype.ANGLE), ("CA_CB_CG_CD_diangle", Dtype.ANGLE), 
            ("CD_NE_length", Dtype.SCALAR), ("CG_CD_NE_angle", Dtype.ANGLE), ("CB_CG_CD_NE_diangle", Dtype.ANGLE), 
            ("NE_CZ_length", Dtype.SCALAR), ("CD_NE_CZ_angle", Dtype.ANGLE), ("CG_CD_NE_CZ_diangle", Dtype.ANGLE), 
            ("CZ_NH1_length", Dtype.SCALAR), ("NE_CZ_NH1_angle", Dtype.ANGLE), ("CD_NE_CZ_NH1_diangle", Dtype.ANGLE), 
            ("CZ_NH2_length", Dtype.SCALAR), ("NE_CZ_NH2_angle", Dtype.ANGLE), ("CD_NE_CZ_NH2_diangle", Dtype.ANGLE)],
    "K": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_CD_length", Dtype.SCALAR), ("CB_CG_CD_angle", Dtype.ANGLE), ("CA_CB_CG_CD_diangle", Dtype.ANGLE), 
            ("CD_CE_length", Dtype.SCALAR), ("CG_CD_CE_angle", Dtype.ANGLE), ("CB_CG_CD_CE_diangle", Dtype.ANGLE), 
            ("CE_NZ_length", Dtype.SCALAR), ("CD_CE_NZ_angle", Dtype.ANGLE), ("CG_CD_CE_NZ_diangle", Dtype.ANGLE)],
    "D": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_OD1_length", Dtype.SCALAR), ("CB_CG_OD1_angle", Dtype.ANGLE), ("CA_CB_CG_OD1_diangle", Dtype.ANGLE), 
            ("CG_OD2_length", Dtype.SCALAR), ("CB_CG_OD2_angle", Dtype.ANGLE), ("CA_CB_CG_OD2_diangle", Dtype.ANGLE)],
    "N": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_OD1_length", Dtype.SCALAR), ("CB_CG_OD1_angle", Dtype.ANGLE), ("CA_CB_CG_OD1_diangle", Dtype.ANGLE), 
            ("CG_ND2_length", Dtype.SCALAR), ("CB_CG_ND2_angle", Dtype.ANGLE), ("CA_CB_CG_ND2_diangle", Dtype.ANGLE)],
    "E": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_CD_length", Dtype.SCALAR), ("CB_CG_CD_angle", Dtype.ANGLE), ("CA_CB_CG_CD_diangle", Dtype.ANGLE), 
            ("CD_OE1_length", Dtype.SCALAR), ("CG_CD_OE1_angle", Dtype.ANGLE), ("CB_CG_CD_OE1_diangle", Dtype.ANGLE), 
            ("CD_OE2_length", Dtype.SCALAR), ("CG_CD_OE2_angle", Dtype.ANGLE), ("CB_CG_CD_OE2_diangle", Dtype.ANGLE)],
    "Q": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_CD_length", Dtype.SCALAR), ("CB_CG_CD_angle", Dtype.ANGLE), ("CA_CB_CG_CD_diangle", Dtype.ANGLE), 
            ("CD_OE1_length", Dtype.SCALAR), ("CG_CD_OE1_angle", Dtype.ANGLE), ("CB_CG_CD_OE1_diangle", Dtype.ANGLE), 
            ("CD_NE2_length", Dtype.SCALAR), ("CG_CD_NE2_angle", Dtype.ANGLE), ("CB_CG_CD_NE2_diangle", Dtype.ANGLE)],
    "M": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_SD_length", Dtype.SCALAR), ("CB_CG_SD_angle", Dtype.ANGLE), ("CA_CB_CG_SD_diangle", Dtype.ANGLE), 
            ("SD_CE_length", Dtype.SCALAR), ("CG_SD_CE_angle", Dtype.ANGLE), ("CB_CG_SD_CE_diangle", Dtype.ANGLE)],
    "H": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_ND1_length", Dtype.SCALAR), ("CB_CG_ND1_angle", Dtype.ANGLE), ("CA_CB_CG_ND1_diangle", Dtype.ANGLE), 
            ("CG_CD2_length", Dtype.SCALAR), ("CB_CG_CD2_angle", Dtype.ANGLE), ("CA_CB_CG_CD2_diangle", Dtype.ANGLE), 
            ("ND1_CE1_length", Dtype.SCALAR), ("CG_ND1_CE1_angle", Dtype.ANGLE), ("CB_CG_ND1_CE1_diangle", Dtype.ANGLE), 
            ("CD2_NE2_length", Dtype.SCALAR), ("CG_CD2_NE2_angle", Dtype.ANGLE), ("CB_CG_CD2_NE2_diangle", Dtype.ANGLE)],
    "P": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_CD_length", Dtype.SCALAR), ("CB_CG_CD_angle", Dtype.ANGLE), ("CA_CB_CG_CD_diangle", Dtype.ANGLE)],
    "F": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_CD1_length", Dtype.SCALAR), ("CB_CG_CD1_angle", Dtype.ANGLE), ("CA_CB_CG_CD1_diangle", Dtype.ANGLE), 
            ("CG_CD2_length", Dtype.SCALAR), ("CB_CG_CD2_angle", Dtype.ANGLE), ("CA_CB_CG_CD2_diangle", Dtype.ANGLE), 
            ("CD1_CE1_length", Dtype.SCALAR), ("CG_CD1_CE1_angle", Dtype.ANGLE), ("CB_CG_CD1_CE1_diangle", Dtype.ANGLE), 
            ("CD2_CE2_length", Dtype.SCALAR), ("CG_CD2_CE2_angle", Dtype.ANGLE), ("CB_CG_CD2_CE2_diangle", Dtype.ANGLE), 
            ("CE1_CZ_length", Dtype.SCALAR), ("CD1_CE1_CZ_angle", Dtype.ANGLE), ("CG_CD1_CE1_CZ_diangle", Dtype.ANGLE)],
    "Y": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_CD1_length", Dtype.SCALAR), ("CB_CG_CD1_angle", Dtype.ANGLE), ("CA_CB_CG_CD1_diangle", Dtype.ANGLE), 
            ("CG_CD2_length", Dtype.SCALAR), ("CB_CG_CD2_angle", Dtype.ANGLE), ("CA_CB_CG_CD2_diangle", Dtype.ANGLE), 
            ("CD1_CE1_length", Dtype.SCALAR), ("CG_CD1_CE1_angle", Dtype.ANGLE), ("CB_CG_CD1_CE1_diangle", Dtype.ANGLE), 
            ("CD2_CE2_length", Dtype.SCALAR), ("CG_CD2_CE2_angle", Dtype.ANGLE), ("CB_CG_CD2_CE2_diangle", Dtype.ANGLE), 
            ("CE1_CZ_length", Dtype.SCALAR), ("CD1_CE1_CZ_angle", Dtype.ANGLE), ("CG_CD1_CE1_CZ_diangle", Dtype.ANGLE), 
            ("CZ_OH_length", Dtype.SCALAR), ("CE1_CZ_OH_angle", Dtype.ANGLE), ("CD1_CE1_CZ_OH_diangle", Dtype.ANGLE)],
    "W": [("CA_CB_length", Dtype.SCALAR), ("C_CA_CB_angle", Dtype.ANGLE), ("N_C_CA_CB_diangle", Dtype.ANGLE), 
            ("CB_CG_length", Dtype.SCALAR), ("CA_CB_CG_angle", Dtype.ANGLE), ("N_CA_CB_CG_diangle", Dtype.ANGLE), 
            ("CG_CD1_length", Dtype.SCALAR), ("CB_CG_CD1_angle", Dtype.ANGLE), ("CA_CB_CG_CD1_diangle", Dtype.ANGLE), 
            ("CG_CD2_length", Dtype.SCALAR), ("CB_CG_CD2_angle", Dtype.ANGLE), ("CA_CB_CG_CD2_diangle", Dtype.ANGLE), 
            ("CD1_NE1_length", Dtype.SCALAR), ("CG_CD1_NE1_angle", Dtype.ANGLE), ("CB_CG_CD1_NE1_diangle", Dtype.ANGLE), 
            ("CD2_CE2_length", Dtype.SCALAR), ("CG_CD2_CE2_angle", Dtype.ANGLE), ("CB_CG_CD2_CE2_diangle", Dtype.ANGLE), 
            ("CD2_CE3_length", Dtype.SCALAR), ("CG_CD2_CE3_angle", Dtype.ANGLE), ("CB_CG_CD2_CE3_diangle", Dtype.ANGLE), 
            ("CE2_CZ2_length", Dtype.SCALAR), ("CD2_CE2_CZ2_angle", Dtype.ANGLE), ("CG_CD2_CE2_CZ2_diangle", Dtype.ANGLE), 
            ("CE3_CZ3_length", Dtype.SCALAR), ("CD2_CE3_CZ3_angle", Dtype.ANGLE), ("CG_CD2_CE3_CZ3_diangle", Dtype.ANGLE), 
            ("CZ2_CH2_length", Dtype.SCALAR), ("CE2_CZ2_CH2_angle", Dtype.ANGLE), ("CD2_CE2_CZ2_CH2_diangle", Dtype.ANGLE)],
}


def apply_rotamer_to_residue(
        structure: List[Geometry.Geo],
        index: int,
        rotamer: np.ndarray,
    ):
    """
    Apply a rotamer to a residue in the structure.
    
    Args:
        structure: Biopython Structure object
        model_id: Model identifier
        chain_id: Chain identifier
        res_id: Residue identifier
        rotamer: Rotamer parameters
        res_name: Residue name (3-letter code)
        in_place: If True, modify the structure in place; otherwise, create a copy
        
    Returns:
        Structure.Structure: Modified structure
    """

    # Applies a rotamer to a residue in the structure. 
    thisres = structure[index]
    rotamer_var_list = ROTAMER_VARS.get(thisres.residue_name, None)
    if rotamer_var_list is None:
        raise ValueError(f"Unknown residue type: {thisres.residue_name}")
    
    # Use reflection to access the rotamer variables dynamically
    for i, (var, vartype) in enumerate(rotamer_var_list):
        if vartype == Dtype.SCALAR:
            setattr(thisres, var, rotamer[i])
        elif vartype == Dtype.ANGLE:
            setattr(thisres, var, np.arcsin(rotamer[i]))

def optimize_rotamers(
        structure: List[Geometry.Geo],
        atom_cloud: np.ndarray,
        atom_types: np.ndarray,
        element_map: Dict[str, int],
        alpha_offsets: np.ndarray,
        rotamer_library_path: str = "rotamer_library.npz"
    ) -> Structure.Structure:
    """
    Optimize the side chain rotamers of a protein structure to minimize the cost function.
    
    Args:
        structure: Biopython Structure object with the backbone
        atom_cloud: Global point cloud of atom positions
        atom_types: Element types of atoms in the global point cloud
        element_map: Mapping from element names to indices
        rotamer_library_path: Path to the rotamer library file
        
    Returns:
        Structure.Structure: Optimized protein structure
    """
    # Load rotamer library
    if not os.path.exists(rotamer_library_path):
        raise FileNotFoundError(f"Rotamer library not found at {rotamer_library_path}")
    
    rotamer_library = np.load(rotamer_library_path, allow_pickle=True)

    # Generate kd-trees for fast nearest neighbor search
    atom_kd_trees = {}
    for atom_type in np.unique(atom_types):
        indices = np.where(atom_types == atom_type)[0]
        if len(indices) > 0:
            atom_kd_trees[atom_type] = cKDTree(atom_cloud[indices])
    
    # loop through each residue in the structure
    for i, res in enumerate(structure):
        three_letter = ONE_LETTER_TO_THREE_LETTER.get(res.residue_name, None)
        if three_letter == "GLY":
            # Glycine has no side chain, skip optimization
            continue
        if three_letter is None:
            raise ValueError(f"Unknown residue name: {res.residue_name}")
        if three_letter not in rotamer_library:
            raise ValueError(f"No rotamers found for residue: {three_letter}")

        rotamers = rotamer_library[three_letter]
        best_cost = float('inf')
        best_rotamer = None
        # For each rotamer, compute the cost
        for rotamer in rotamers:
            # Apply the rotamer to the residue
            apply_rotamer_to_residue(structure, i, rotamer)
            
            # Compute the cost of this residue with the applied rotamer
            cost = compute_residue_cost(structure[i],
                                        alpha_offsets[i],
                                        atom_cloud,
                                        atom_kd_trees,
                                        element_map)
            
            # Check if this is the best cost so far
            if cost < best_cost:
                best_cost = cost
                best_rotamer = rotamer
        # Apply the best rotamer found
        if best_rotamer is not None:
            apply_rotamer_to_residue(structure, i, best_rotamer)
    
    return structure