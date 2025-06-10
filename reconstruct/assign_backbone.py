'''
Utility for assigning likely backbone atoms from nearby frames for reconstruction.
'''

import numpy as np
from .constants import *

def assign_backbone_atoms(
        alpha_carbon_envs: np.ndarray,
        alpha_carbon_elem_types: np.ndarray
    ):
    '''
    Assigns likely backbone atoms from nearby frames based on the alpha carbon environments and their element types. Used to compute relative 
    alpha carbon offsets. 

    The algorithm works as follows:
    1. This frame's alpha carbon is the carbon closest to the origin (should be at the origin +/- some noise)
    2. This frame's nitrogen is the nearest nitrogen to the alpha carbon.
    3. If this isn't the first amino acid, the previous frame's carbonyl carbon is the closest carbon to this frame's nitrogen that is also 
        near an oxygen. If there is no oxygen nearby, it's just the closest carbon to this frame's nitrogen. If this frame doesn't have a 
        nitrogen, it's the closest carbon to the alpha carbon that's near an oxygen and not this frame's carbonyl carbon.
    4. The previous frame's alpha carbon is the closest carbon to its carbonyl carbon.
    5. If this is the last amino acid, this frame's carbonyl carbon is the nearest carbon to the alpha carbon that is near two oxygens
    6. If this isn't the last amino acid, this frame's carbonyl carbon is the nearest carbon to the alpha carbon that is near an oxygen and a 
        nitrogen.
    7. The next frame's nitrogen is the nearest nitrogen to this frame's carbonyl carbon
    8. The next frame's alpha carbon is the nearest carbon to this frame's carbonyl carbon.

    Args:
        alpha_carbon_envs (np.ndarray): An array of shape (L, N, 3) representing all predicted atoms within 5A of each alpha carbon atom. 
            Alpha carbons are included, and coordinates are in Angstroms relative to the alpha carbon atom.
        alpha_carbon_elem_types (np.ndarray): An array of shape (L, N) representing the element types of the atoms in alpha_carbon_envs.
    Returns:
        np.ndarray: An array of shape (L, 2, 3) containing the relative distance between CA_i-1 to CA_i and CA_i to CA_i+1 for each position i in the sequence. 
            The first position is set to [0, 0, 0] since there is no previous alpha carbon.
    '''
    L, N, _ = alpha_carbon_envs.shape
    output = np.zeros((L, 2, 3), dtype=np.float32)
    # Precompute global carbon, nitrogen, and oxygen positions
    carbons_global = alpha_carbon_envs[alpha_carbon_elem_types == 'C']
    nitrogens_global = alpha_carbon_envs[alpha_carbon_elem_types == 'N']
    oxygens_global = alpha_carbon_envs[alpha_carbon_elem_types == 'O']

    # Iterate over each frame and assign backbone atoms
    for frame_index in range(L):
        output[frame_index] = assign_frame_atoms(
            alpha_carbon_envs[frame_index],
            alpha_carbon_elem_types[frame_index],
            frame_index,
            carbons_global,
            nitrogens_global,
            oxygens_global
        )
    return output
    
def assign_frame_atoms(
        alpha_carbon_envs: np.ndarray,
        alpha_carbon_elem_types: np.ndarray,
        frame_index: int,
        carbons_global: np.ndarray,
        nitrogens_global: np.ndarray,
        oxygens_global: np.ndarray
    ):
    '''
    Assigns backbone atoms for a specific frame based on the algorithm described above.

    Args:
        alpha_carbon_envs (np.ndarray): An array of shape (L, N, 3) representing all predicted atoms within 5A of each alpha carbon atom.
        alpha_carbon_elem_types (np.ndarray): An array of shape (L, N) representing the element types of the atoms in alpha_carbon_envs.
        frame_index (int): The index of the frame for which to assign backbone atoms.
        carbons_global (np.ndarray): An array of shape (L, N, 3) representing the carbon atoms in the alpha carbon environments.
        nitrogens_global (np.ndarray): An array of shape (L, N, 3) representing the nitrogen atoms in the alpha carbon environments.
        oxygens_global (np.ndarray): An array of shape (L, N, 3) representing the oxygen atoms in the alpha carbon environments.
    
    Returns:
        np.ndarray: An array of shape (2, 3) containing the relative positions of the assigned alpha carbon atoms for the specified frame.
    '''
    
    is_first = frame_index == 0
    is_last = frame_index == (alpha_carbon_envs.shape[0] - 1)

    # Precompute indexes for carbon, nitrogen, and oxygen
    carbons = carbons_global[frame_index]
    nitrogens = nitrogens_global[frame_index]
    oxygens = oxygens_global[frame_index]

    # To store alpha carbons
    prev_ca = None  # Previous frame's alpha carbon
    ca = None  # Current frame's alpha carbon
    next_ca = None  # Next frame's alpha carbon

    # Find this frame's CA
    ca_index = np.argmin(np.linalg.norm(carbons, axis=1))
    ca = carbons[ca_index]

    # Find this frame's N
    closest_nitrogen_index = np.argmin(np.linalg.norm(nitrogens - ca, axis=1))
    n = nitrogens[closest_nitrogen_index]
    # Check that it's within the maximum distance
    n = n if np.linalg.norm(n - ca) < MAX_CA_N_DISTANCE else None
    # We'll have to handle the case where there is no nitrogen later

    # Get all carbonyl carbons
    carbonyl_carbons = carbons[np.linalg.norm(carbons[None, :] - oxygens[None, :], axis=2) < CARBONYL_OXYGEN_DISTANCE]
    # Exclude alpha carbon as it is likely to be included in the carbonyl carbons
    carbonyl_carbons = carbonyl_carbons[carbonyl_carbons != ca]
    # Assign this frame's carbonyl carbon
    # If it's the last frame, we need to find the carbonyl carbon that is near two oxygens
    if is_last:
        # Get carbonyl carbons with two nearby oxygens
        num_nearby_oxygens = np.sum(np.linalg.norm(oxygens[:, None] - carbonyl_carbons[None, :], axis=2) < CARBONYL_OXYGEN_DISTANCE, axis=1)
        valid_carbonyls = carbonyl_carbons[num_nearby_oxygens >= 2]
        if valid_carbonyls.size > 0:
            carbonyl_c = valid_carbonyls[np.argmin(np.linalg.norm(valid_carbonyls - ca, axis=1))]
        else:
            # Fallback: just take the closest carbonyl carbon to the alpha carbon
            carbonyl_c = carbonyl_carbons[np.argmin(np.linalg.norm(carbonyl_carbons - ca, axis=1))]
    else:
        # Get carbonyl carbons with a nearby nitrogen
        amide_carbonyls = carbonyl_carbons[np.linalg.norm(carbonyl_carbons - n, axis=1) < MAX_AMIDE_C_DISTANCE]
        if amide_carbonyls.size > 0:
            carbonyl_c = amide_carbonyls[np.argmin(np.linalg.norm(amide_carbonyls - ca, axis=1))]
        else:
            # Fallback: just take the closest carbonyl carbon to the alpha carbon
            carbonyl_c = carbonyl_carbons[np.argmin(np.linalg.norm(carbonyl_carbons - ca, axis=1))]

    if not is_first:
        if n is not None:
            # Get the previous frame's carbonyl carbon
            # The closest carbonyl to the nitrogen is likely the previous frame's carbonyl carbon
            prev_carbonyl_index = np.argmin(np.linalg.norm(carbonyl_carbons - n, axis=1))
            prev_carbonyl = carbonyl_carbons[prev_carbonyl_index]

            # Now, find possible alpha carbons. Score them based on distance to carbonyl carbons and proximity to closest nitrogen
            scores = np.linalg.norm(carbons - prev_carbonyl, axis=1) * CA_SCORE_CARBONYL_C + np.linalg.norm(carbons - n, axis=1) * CA_SCORE_N
            # Exclude the carbonyl carbon itself
            scores[prev_carbonyl_index] = np.inf
            # Find the best alpha carbon
            prev_ca_index = np.argmin(scores)
            prev_ca = carbons[prev_ca_index]
        else:
            # Fallback: find the best candidate for other frame's carbonyl carbon
            candidate_carbonyls = carbonyl_carbons[carbonyl_carbons != carbonyl_c]

            # Just take the closest carbon to the alpha carbon that is near an oxygen
            if candidate_carbonyls.size > 0:
                prev_carbonyl_index = np.argmin(np.linalg.norm(candidate_carbonyls - ca, axis=1))
                prev_carbonyl = candidate_carbonyls[prev_ca_index]

                # Now, find possible alpha carbons. Score them based on distance to carbonyl carbons and proximity to closest nitrogen
                scores = np.linalg.norm(carbons - prev_carbonyl, axis=1) * CA_SCORE_CARBONYL_C + np.linalg.norm(carbons - n, axis=1) * CA_SCORE_N
                # Exclude the carbonyl carbon itself
                scores[prev_carbonyl_index] = np.inf
                # Find the best alpha carbon
                prev_ca_index = np.argmin(scores)
                prev_ca = carbons[prev_ca_index]
            else:
                # At this point, results will be garbage anyways so just return the nearest carbon to the alpha carbon
                candidates = carbons[carbons != ca]
                candidates = candidates[candidates != carbonyl_c]
                if candidates.size > 0:
                    prev_ca_index = np.argmin(np.linalg.norm(candidates - ca, axis=1))
                    prev_ca = candidates[prev_ca_index]
                else:
                    prev_ca = ca

    # Find the next frame's nitrogen
    if not is_last:
        next_frame_nitrogens = nitrogens_global[frame_index + 1]
        next_n_index = np.argmin(np.linalg.norm(next_frame_nitrogens - carbonyl_c, axis=1))
        next_n = next_frame_nitrogens[next_n_index]

        # Get the next frame's alpha carbon
        next_frame_carbons = carbons_global[frame_index + 1]
        next_ca_index = np.argmin(np.linalg.norm(next_frame_carbons - next_n, axis=1))
        next_ca = next_frame_carbons[next_ca_index]

    # Now return the relative positions
    result = np.zeros((2, 3))
    if prev_ca is not None:
        result[0] = prev_ca - ca  # CA_i-1 to CA_i
    if next_ca is not None:
        result[1] = next_ca - ca  # CA_i to CA_i+1
    return result

