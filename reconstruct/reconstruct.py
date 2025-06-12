'''
Contains an algorithm to convert noisy alpha carbon environments into a protein structure. The algorithm is designed to always produce a valid
protein structure, so its results should be used alongside a refinement algorithm.
'''

import numpy as np
from .assign_backbone import assign_backbone_atoms
import PeptideBuilder
from PeptideBuilder import Geometry
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
from .constants import *
from .utils import *

def reconstruct_structure(
        alpha_carbon_envs : np.ndarray,
        alpha_carbon_elem_types : np.ndarray,
        sequence : str
    ):
    '''
    Reconstructs a protein structure from noisy alpha carbon environments and their corresponding element types.

    1. Assign likely "nearest neighbour" backbone atoms in each frame using the algorithm defined in assign_backbone.py.
    2. Use the neighbour alpha carbon relative positions to get backbone lengths and angles
    3. Fit the known sequence backbone to the lengths and angles
    4. Transform all atoms from all alpha carbon envs to world space. For each side chain, find the rotamer that minimizes the
        cost function. 
    5. Do local greedy optimization of the molecule where each reconstructed atom is moved to the position the nearest 
        predicted atom of the same type, with some "smoothing" that prevents it from wandering too far from its current 
        position.

    Args:
        alpha_carbon_envs (np.ndarray): An array of shape (L, N, 3) representing all predicted atoms within 5A of each alpha carbon atom. 
            Alpha carbons are included, and coordinates are in Angstroms relative to the alpha carbon atom.
        alpha_carbon_elem_types (np.ndarray): An array of shape (L, N) representing the element types of the atoms in alpha_carbon_envs.
        sequence (str): The amino acid sequence of the protein, where each character corresponds to an amino acid.
    
    Returns:
        Bio.PDB.Structure.Structure: The reconstructed protein structure
    '''
    
    # 1. Assign nearest neighbour atoms to get offsets for each alpha carbon
    offsets = assign_backbone_atoms(alpha_carbon_envs, alpha_carbon_elem_types)
    L = len(sequence)

    # 2. Get neighbour information to compute backbone angles
    neighbour_info = integrate_neighbour_information(offsets, L)
    # 3. Fit the known sequence backbone to the alpha carbon positions
    geos = fit_backbone(neighbour_info, sequence)

    unfit_backbone = PeptideBuilder.make_structure_from_geos(geos)
    atom_positions, atom_types, alpha_offsets = make_atom_cloud_from_envs(alpha_carbon_envs, alpha_carbon_elem_types, unfit_backbone)
    
    # 4. Optimize rotamers
    from .optimize_rotamers import optimize_rotamers
    element_map = {e.upper(): i for i, e in enumerate(elem)}
    #optimized_structure = optimize_rotamers(geos, atom_positions, atom_types, element_map, alpha_offsets)
    # Convert the optimized structure to a Bio.PDB Structure object
    #optimized_structure = PeptideBuilder.make_structure_from_geos(optimized_structure)
    
    return unfit_backbone

def compute_frame(
        ca_offsets : np.ndarray,
        frame_index : int
    ):
    '''
    Computes the following for a specific frame:
        - peptide_bond: The length of the peptide bond.
        - N_CA_C_angle: The angle between the nitrogen, alpha carbon, and carbon atoms.
        - CA_N_length: The length bet   ween the alpha carbon and nitrogen atoms.
        - CA_C_length: The length between the alpha carbon and carbon atoms.
        - phi: The phi dihedral angle.
        - psi: The psi dihedral angle.
        - omega: The omega dihedral angle.
    '''
    Cim1_N_1 = None
    Cim1_N_2 = None
    N_CA_1 = None
    N_CA_2 = None
    CA_C_1 = None
    CA_C_2 = None
    C_Np1_1 = None
    C_Np1_2 = None
    pep1 = None
    pep2 = None
    np1_CAp1_1 = None
    np1_CAp1_2 = None
    if frame_index == 0:
        Cim1_N_1 = ca_offsets[0, 1]
        Cim1_N_1 = Cim1_N_1 if not np.isnan(Cim1_N_1).any() else None
        N_CA_1 = ca_offsets[0, 2]
        N_CA_1 = N_CA_1 if not np.isnan(N_CA_1).any() else None
        CA_C_1 = ca_offsets[0, 3]
        CA_C_1 = CA_C_1 if not np.isnan(CA_C_1).any() else None
        CA_C_2 = ca_offsets[1, 0]
        CA_C_2 = CA_C_2 if not np.isnan(CA_C_2).any() else None
        pep1 = ca_offsets[0, 4]
        pep1 = pep1 if not np.isnan(pep1).any() else None
        pep2 = ca_offsets[1, 1]
        pep2 = pep2 if not np.isnan(pep2).any() else None
        C_Np1_1 = ca_offsets[0, 4]
        C_Np1_2 = ca_offsets[1, 1]
        C_Np1_1 = C_Np1_1 if not np.isnan(C_Np1_1).any() else None
        C_Np1_2 = C_Np1_2 if not np.isnan(C_Np1_2).any() else None
        np1_CAp1_1 = ca_offsets[0, 5]
        np1_CAp1_2 = ca_offsets[1, 2]
        np1_CAp1_1 = np1_CAp1_1 if not np.isnan(np1_CAp1_1).any() else None
        np1_CAp1_2 = np1_CAp1_2 if not np.isnan(np1_CAp1_2).any() else None
    elif frame_index == (ca_offsets.shape[0] - 1):
        Cim1_N_1 = ca_offsets[frame_index - 1, 4]
        Cim1_N_2 = ca_offsets[frame_index, 1]
        Cim1_N_1 = Cim1_N_1 if not np.isnan(Cim1_N_1).any() else None
        Cim1_N_2 = Cim1_N_2 if not np.isnan(Cim1_N_2).any() else None
        N_CA_1 = ca_offsets[frame_index - 1, 5]
        N_CA_2 = ca_offsets[frame_index, 2]
        N_CA_1 = N_CA_1 if not np.isnan(N_CA_1).any() else None
        N_CA_2 = N_CA_2 if not np.isnan(N_CA_2).any() else None
        CA_C_1 = ca_offsets[frame_index, 3]
        CA_C_1 = CA_C_1 if not np.isnan(CA_C_1).any() else None
        # No N+1
    else:
        Cim1_N_1 = ca_offsets[frame_index - 1, 4]
        Cim1_N_2 = ca_offsets[frame_index, 1]
        Cim1_N_1 = Cim1_N_1 if not np.isnan(Cim1_N_1).any() else None
        Cim1_N_2 = Cim1_N_2 if not np.isnan(Cim1_N_2).any() else None
        N_CA_1 = ca_offsets[frame_index - 1, 5]
        N_CA_2 = ca_offsets[frame_index, 2]
        N_CA_1 = N_CA_1 if not np.isnan(N_CA_1).any() else None
        N_CA_2 = N_CA_2 if not np.isnan(N_CA_2).any() else None
        CA_C_1 = ca_offsets[frame_index, 3]
        CA_C_2 = ca_offsets[frame_index + 1, 0]
        CA_C_1 = CA_C_1 if not np.isnan(CA_C_1).any() else None
        CA_C_2 = CA_C_2 if not np.isnan(CA_C_2).any() else None
        pep1 = ca_offsets[frame_index, 4]
        pep2 = ca_offsets[frame_index + 1, 1]
        pep1 = pep1 if not np.isnan(pep1).any() else None
        pep2 = pep2 if not np.isnan(pep2).any() else None
        C_Np1_1 = ca_offsets[frame_index, 4]
        C_Np1_2 = ca_offsets[frame_index + 1, 1]
        C_Np1_1 = C_Np1_1 if not np.isnan(C_Np1_1).any() else None
        C_Np1_2 = C_Np1_2 if not np.isnan(C_Np1_2).any() else None
        np1_CAp1_1 = ca_offsets[frame_index, 5]
        np1_CAp1_2 = ca_offsets[frame_index + 1, 2]
        np1_CAp1_1 = np1_CAp1_1 if not np.isnan(np1_CAp1_1).any() else None
        np1_CAp1_2 = np1_CAp1_2 if not np.isnan(np1_CAp1_2).any() else None
    # Compute lengths and angles
    peptide_bond = None
    N_CA_C_angle = None
    CA_N_length = None
    CA_C_length = None
    N_CA_C_angle = None
    phi = None
    psi = None
    omega = None

    if pep1 is not None and pep2 is not None:
        peptide_bond = (pep1 + pep2) / 2
    elif pep1 is not None:
        peptide_bond = pep1
    elif pep2 is not None:
        peptide_bond = pep2

    CA_N_vec = None
    CA_C_vec = None
    if N_CA_1 is not None and N_CA_2 is not None:
        CA_N_vec = -(N_CA_1 + N_CA_2) / 2
    elif N_CA_1 is not None:
        CA_N_vec = -N_CA_1
    elif N_CA_2 is not None:
        CA_N_vec = -N_CA_2

    if CA_C_1 is not None and CA_C_2 is not None:
        CA_C_vec = (CA_C_1 + CA_C_2) / 2
    elif CA_C_1 is not None:
        CA_C_vec = CA_C_1
    elif CA_C_2 is not None:
        CA_C_vec = CA_C_2
    
    CA_N_length = np.linalg.norm(CA_N_vec) if CA_N_vec is not None else None
    if CA_N_length is not None and CA_N_length == 0:
        CA_N_length = None
        CA_N_vec = None
    CA_C_length = np.linalg.norm(CA_C_vec) if CA_C_vec is not None else None
    if CA_C_length is not None and CA_C_length == 0:
        CA_C_length = None
        CA_C_vec = None
    if CA_N_vec is not None and CA_C_vec is not None:
        N_CA_C_angle = np.arccos(np.dot(CA_N_vec, CA_C_vec) / (CA_N_length * CA_C_length))
    
    Cim1_N = None
    if Cim1_N_1 is not None and Cim1_N_2 is not None:
        Cim1_N = (Cim1_N_1 + Cim1_N_2) / 2
    elif Cim1_N_1 is not None:
        Cim1_N = Cim1_N_1
    elif Cim1_N_2 is not None:
        Cim1_N = Cim1_N_2

    if Cim1_N is not None and CA_N_vec is not None and CA_C_vec is not None:
        phi = dihedral(
            Cim1_N, CA_N_vec, CA_C_vec
        )

    C_Np1 = None
    if C_Np1_1 is not None and C_Np1_2 is not None:
        C_Np1 = (C_Np1_1 + C_Np1_2) / 2
    elif C_Np1_1 is not None:
        C_Np1 = C_Np1_1
    elif C_Np1_2 is not None:
        C_Np1 = C_Np1_2
    if CA_N_vec is not None and CA_C_vec is not None and C_Np1 is not None:
        psi = dihedral(
            -CA_N_vec, CA_C_vec, C_Np1
        )
    
    np1_CAp1 = None
    if np1_CAp1_1 is not None and np1_CAp1_2 is not None:
        np1_CAp1 = (np1_CAp1_1 + np1_CAp1_2) / 2
    elif np1_CAp1_1 is not None:
        np1_CAp1 = np1_CAp1_1
    elif np1_CAp1_2 is not None:
        np1_CAp1 = np1_CAp1_2
    # Omega is from this frame's CA to the next frame's CA
    if CA_C_vec is not None and C_Np1 is not None and np1_CAp1 is not None:
        omega = dihedral(
            CA_C_vec, C_Np1, np1_CAp1
        )

    return {
        'peptide_bond': np.linalg.norm(peptide_bond) if peptide_bond is not None else None,
        'N_CA_C_angle': rad2deg(N_CA_C_angle) if N_CA_C_angle is not None else None,
        'CA_N_length': CA_N_length,
        'CA_C_length': CA_C_length,
        'phi': rad2deg(phi) if phi is not None else None,
        'psi': rad2deg(psi) if psi is not None else None,
        'omega': rad2deg(omega) if omega is not None else None
    }

def integrate_neighbour_information(
        offsets : np.ndarray,
        L : int
    ):
    '''
    Integrates the neighbour information to compute the backbone angles.

    Args:
        offsets (np.ndarray): An array of shape (L, 6, 3) representing the CA_i-1 - C - N_i - CA - C - N_i+1 - CA offsets from each other in each frame.
        L (int): The length of the protein sequence, which corresponds to the number of alpha carbon atoms.
    Returns:
        np.ndarray: A list of dictionaries, each containing the following keys:
            - peptide_bond: The length of the peptide bond.
            - N_CA_C_angle: The angle between the nitrogen, alpha carbon, and carbon atoms.
            - CA_N_length: The length between the alpha carbon and nitrogen atoms.
            - CA_C_length: The length between the alpha carbon and carbon atoms.
            - phi: The phi dihedral angle.
            - psi: The psi dihedral angle.
            - omega: The omega dihedral angle.
            - ca_im1_ca_offset: The offset vector from the alpha carbon of the previous residue to the alpha carbon of the current residue.
    '''
    output = []
    # First frame
    for frame_index in range(L):
        frame_info = compute_frame(offsets, frame_index)
        output.append(frame_info)
    return output   

def fit_backbone(
        neighbour_info : np.ndarray,
        sequence : str
    ):
    '''
    Uses PeptideBuilder to fit the known sequence backbone to the alpha carbon positions.

    Args:
        neighbour_info (np.ndarray): An array of dictionaries containing the neighbour information for each frame.
        sequence (str): The amino acid sequence of the protein, where each character corresponds to an amino acid.
    Returns:
        list: A list of PeptideBuilder.Geometry objects representing the fitted backbone geometries for each residue in the sequence.
    '''

    geos = []
    for i, aa in enumerate(sequence):
        # Get the neighbour information for this residue
        info = neighbour_info[i]
        # Create a Geometry object for this residue
        geo = Geometry.geometry(aa)
        # Set the backbone lengths and angles
        pep_bond = info['peptide_bond']
        pep_bond = pep_bond if pep_bond is not None else geo.peptide_bond
        geo.peptide_bond = pep_bond
        N_CA_C_angle = info['N_CA_C_angle']
        N_CA_C_angle = N_CA_C_angle if N_CA_C_angle is not None else geo.N_CA_C_angle
        geo.N_CA_C_angle = N_CA_C_angle
        CA_N_length = info['CA_N_length']
        CA_N_length = CA_N_length if CA_N_length is not None else geo.CA_N_length
        geo.CA_N_length = CA_N_length
        CA_C_length = info['CA_C_length']
        CA_C_length = CA_C_length if CA_C_length is not None else geo.CA_C_length
        geo.CA_C_length = CA_C_length
        phi = info['phi']
        phi = phi if phi is not None else geo.phi
        geo.phi = phi
        # Instead of psi, give psi_i-1
        psi_im1 = geo.psi_im1
        if i > 0:
            psi_im1 = neighbour_info[i - 1]['psi']
        psi_im1 = psi_im1 if psi_im1 is not None else geo.psi_im1
        geo.psi_im1 = psi_im1
        omega = info['omega']
        omega = omega if omega is not None else geo.omega
        geo.omega = omega
        geos.append(geo)
    return geos
    
def make_atom_cloud_from_envs(
        alpha_carbon_envs : np.ndarray,
        alpha_carbon_elem_types : np.ndarray,
        unfit_backbone : Structure.Structure
    ):
    '''
    Converts the alpha carbon environments into a point cloud of atoms in world space.

    Args:
        alpha_carbon_envs (np.ndarray): An array of shape (L, N, 3) representing all predicted atoms within 5A of each alpha carbon atom.
        alpha_carbon_elem_types (np.ndarray): An array of shape (L, N) representing the element types of the atoms in alpha_carbon_envs.
        unfit_backbone (Structure.Structure): The unfit backbone geometry to use as a reference.
    Returns:
        tuple: A tuple containing:
            - positions (np.ndarray): An array of shape (M, 3) representing the positions of the atoms in world space.
            - types (np.ndarray): An array of shape (M,) representing the element types of the atoms in the point cloud.
    '''
    # Placeholder for the point cloud
    positions = []
    types = []
    
    # Iterate through each frame and convert to world space
    model = unfit_backbone[0]
    chain = model["A"]  # Assuming we are working with chain A
    alpha_offsets = np.array([res['CA'].get_coord() for res in chain.get_residues() if res.has_id('CA')])
    for i, env in enumerate(alpha_carbon_envs):
        # The env is already relative to the alpha carbon, so we need to get the alpha carbon position and ignore pads
        alpha_carbon_position = alpha_offsets[i]
        non_pad_indices = np.where(alpha_carbon_elem_types[i] != PAD_ID)[0]
        for j in non_pad_indices:
            atom_position = env[j] + alpha_carbon_position
            atom_type = elem[alpha_carbon_elem_types[i][j]]
            
            positions.append(atom_position)
            types.append(atom_type)
    
    return np.array(positions), np.array(types), alpha_offsets