'''
Utility for assigning likely backbone atoms from nearby frames for reconstruction.
'''

import numpy as np
from .constants import *
from .utils import *

Carbon = elem.index('C')
Nitrogen = elem.index('N')
Oxygen = elem.index('O')

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
        np.ndarray: An array of shape (L, 6, 3) containing the CA_i-1 - C - N_i - CA - C - N_i+1 - CA offsets for each position i in the sequence. 
            Any uncomputed offsets will be set to [NaN, NaN, NaN]. 
    '''
    L, N, _ = alpha_carbon_envs.shape
    output = np.zeros((L, 6, 3), dtype=np.float32)
    # Precompute global carbon, nitrogen, and oxygen positions
    carbons_global = alpha_carbon_elem_types == Carbon
    nitrogens_global = alpha_carbon_elem_types == Nitrogen
    oxygens_global = alpha_carbon_elem_types == Oxygen

    # Iterate over each frame and assign backbone atoms
    for frame_index in range(L):
        output[frame_index] = assign_frame_atoms(
            alpha_carbon_envs[frame_index],
            alpha_carbon_elem_types[frame_index],
            frame_index,
            L,
            carbons_global,
            nitrogens_global,
            oxygens_global
        )
    return output
    
def assign_frame_atoms(
        alpha_carbon_envs: np.ndarray,
        alpha_carbon_elem_types: np.ndarray,
        frame_index: int,
        L: int,
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
        np.ndarray: An array of shape (6, 3) containing the offsets for the specified frame.
    '''
    
    is_first = frame_index == 0
    is_last = frame_index == L - 1

    # Precompute indexes for carbon, nitrogen, and oxygen
    carbons = alpha_carbon_envs[carbons_global[frame_index]]
    nitrogens = alpha_carbon_envs[nitrogens_global[frame_index]]
    oxygens = alpha_carbon_envs[oxygens_global[frame_index]]

    # To store alpha carbons
    prev_ca = None  # Previous frame's alpha carbon
    ca = None  # Current frame's alpha carbon
    next_ca = None  # Next frame's alpha carbon

    # Find this frame's CA
    ca_index = np.argmin(np.linalg.norm(carbons, axis=1))
    ca = carbons[ca_index]

    # Find this frame's N
    dists = np.linalg.norm(nitrogens - ca, axis=1)
    deviation_from_ideal = np.abs(dists - IDEAL_CA_N_DISTANCE)
    consideration_threshold = CONSIDERATION_DIST + np.min(deviation_from_ideal)
    num_to_consider = np.sum(deviation_from_ideal < consideration_threshold)
    if num_to_consider > 2:
        best_assignment = None
        best_deviation = np.inf
        ns_to_consider_indices = np.argsort(deviation_from_ideal)[:num_to_consider]
        ns_to_consider = nitrogens[ns_to_consider_indices]
        for n in ns_to_consider:
            assignment, total_deviation = compute_frame_with_n(
                carbons, oxygens, nitrogens, is_first, is_last, frame_index, ca, n
            )
            if total_deviation < best_deviation:
                best_assignment = assignment
                best_deviation = total_deviation
        prev_ca, prev_carbonyl, ca, n, carbonyl_c, next_n, next_ca = best_assignment
        if best_deviation / 2 > 30:
            print(f"Warning: High deviation for frame {frame_index}: {best_deviation}")
    else:
        n = nitrogens[np.argmin(deviation_from_ideal)]
        assignment, total_deviation = compute_frame_with_n(
            carbons, oxygens, nitrogens, is_first, is_last, frame_index, ca, n
        )
        prev_ca, prev_carbonyl, ca, n, carbonyl_c, next_n, next_ca = assignment

    # Now return the relative positions
    result = np.zeros((6, 3)) + np.nan  # Initialize with NaN values
    # CA_i-1 -> C_i-1
    if prev_ca is not None and prev_carbonyl is not None:
        result[0] = prev_carbonyl - prev_ca
    # C_i-1 -> N_i
    if prev_carbonyl is not None and n is not None:
        result[1] = n - prev_carbonyl
    # N_i -> CA_i
    if n is not None and ca is not None:
        result[2] = ca - n
    # CA_i -> C_i
    if ca is not None and carbonyl_c is not None:
        result[3] = carbonyl_c - ca
    # C_i -> N_i+1
    if carbonyl_c is not None and next_n is not None:
        result[4] = next_n - carbonyl_c
    # N_i+1 -> CA_i+1
    if next_n is not None and next_ca is not None:
        result[5] = next_ca - next_n
    return result

def compute_frame_with_n(carbons, oxygens, nitrogens, is_first, is_last, frame_index, 
                         ca, n):
    total_deviation = np.inf
    # Get all carbonyl carbons
    carbonyl_carbons = carbons[
        np.any(
            np.linalg.norm(carbons[:, None, :] - oxygens[None, :, :], axis=2)
            < CARBONYL_OXYGEN_DISTANCE,
            axis=1
        )
    ]
    # Exclude alpha carbon as it is likely to be included in the carbonyl carbons
    carbonyl_carbons = carbonyl_carbons[np.any(carbonyl_carbons != ca, axis = -1)]
    # Assign this frame's carbonyl carbon
    # If it's the last frame, we need to find the carbonyl carbon that is near two oxygens
    if is_last:
        # Get carbonyl carbons with two nearby oxygens
        num_nearby_oxygens = np.sum(
            np.linalg.norm(carbonyl_carbons[:, None, :] - oxygens[None, :, :], axis=2)
            < CARBONYL_OXYGEN_DISTANCE,
            axis=1
        )
        valid_carbonyls = carbonyl_carbons[num_nearby_oxygens >= 2]
        if valid_carbonyls.size > 0:
            carbonyl_c = valid_carbonyls[np.argmin(np.linalg.norm(valid_carbonyls - ca, axis=1))]
        else:
            # Fallback: just take the closest carbonyl carbon to the alpha carbon
            carbonyl_c = carbonyl_carbons[np.argmin(np.linalg.norm(carbonyl_carbons - ca, axis=1))]
    else:
        # Get carbonyl carbons with a nearby nitrogen
        amide_carbonyls = carbonyl_carbons[
            np.any(
                np.linalg.norm(carbonyl_carbons[:, None, :] - nitrogens[None, :, :], axis=2)
                < MAX_AMIDE_C_DISTANCE,
                axis=1
            )
        ]
        if amide_carbonyls.size > 0:
            dists = np.linalg.norm(amide_carbonyls - ca, axis=1)
            deviation_from_ideal = np.abs(dists - IDEAL_CA_C_DISTANCE)
            carbonyl_c = amide_carbonyls[np.argmin(deviation_from_ideal)]
        else:
            # Fallback: just take the closest carbonyl carbon to the alpha carbon
            dists = np.linalg.norm(carbonyl_carbons - ca, axis=1)
            deviation_from_ideal = np.abs(dists - IDEAL_CA_C_DISTANCE)
            carbonyl_c = carbonyl_carbons[np.argmin(deviation_from_ideal)]

    prev_carbonyl = None  # Previous frame's carbonyl carbon
    prev_ca = None  # Previous frame's alpha carbon
    if not is_first:
        if n is not None:
            # Get the previous frame's carbonyl carbon
            # The closest carbonyl to the nitrogen is likely the previous frame's carbonyl carbon
            dists = np.linalg.norm(carbonyl_carbons - n, axis=1)
            deviation_from_ideal = np.abs(dists - IDEAL_AMIDE_C_DISTANCE)
            prev_carbonyl_index = np.argmin(deviation_from_ideal)
            prev_carbonyl = carbonyl_carbons[prev_carbonyl_index]

            # Now, find possible alpha carbons. Score them based on distance to carbonyl carbons and proximity to closest nitrogen
            dists_carbonyl = np.linalg.norm(carbons - prev_carbonyl, axis=1)
            dists_to_all_nitrogens = np.linalg.norm(carbons[:, None, :] - nitrogens[None, :, :], axis=2)
            deviation_carbonyl = np.abs(dists_carbonyl - IDEAL_CA_C_DISTANCE)
            deviations_from_all_nitrogens = np.abs(dists_to_all_nitrogens - IDEAL_CA_N_DISTANCE)
            best_n_deviation = np.min(deviations_from_all_nitrogens, axis=1)
            scores = deviation_carbonyl * CA_SCORE_CARBONYL_C + best_n_deviation * CA_SCORE_N
            # Exclude the carbonyl carbon itself
            scores[np.all(carbons == prev_carbonyl, axis = -1)] = np.inf
            scores[np.all(carbons == ca, axis = -1)] = np.inf  # Exclude the current alpha carbon as well
            
            consideration_threshold = (CA_SCORE_CARBONYL_C + CA_SCORE_N) * CONSIDERATION_DIST + np.min(scores)
            num_to_consider = np.sum(scores < consideration_threshold)
            if num_to_consider > 2:
                to_consider = np.argsort(scores)[:num_to_consider]
                considered_scores = scores[to_consider]
                candidate_cas = carbons[to_consider]
                b1s = prev_carbonyl - candidate_cas
                b2s = n - prev_carbonyl
                b3s = ca - n
                omega_angles = dihedral(b1s, b2s, b3s)
                omega_angles = rad2deg(omega_angles)
                trans_deviations = np.abs(np.abs(omega_angles) - 180.0)  # Ideally, omega should be 180 degrees
                cis_deviations = np.abs(omega_angles)  # Allow cis angles as well
                deviations = np.minimum(trans_deviations, cis_deviations)
                # Find the best candidate based on the lowest deviation
                best_candidate_index = np.argmin(deviations)
                best_angle_deviation = deviations[best_candidate_index]
                best_angle = omega_angles[best_candidate_index]
                prev_ca = candidate_cas[best_candidate_index]
            else:
                # Find the best alpha carbon
                prev_ca_index = np.argmin(scores)
                prev_ca = carbons[prev_ca_index]

        else:
            # Fallback: find the best candidate for other frame's carbonyl carbon
            candidate_carbonyls = carbonyl_carbons[np.all(carbonyl_carbons != carbonyl_c, axis=-1)]

            # Just take the closest carbon to the alpha carbon that is near an oxygen
            if candidate_carbonyls.size > 0:
                prev_carbonyl_index = np.argmin(np.linalg.norm(candidate_carbonyls - ca, axis=1))
                prev_carbonyl = candidate_carbonyls[prev_carbonyl_index]

                # Now, find possible alpha carbons. Score them based on distance to carbonyl carbons and proximity to closest nitrogen
                dists_carbonyl = np.linalg.norm(carbons - prev_carbonyl, axis=1)
                dists_to_all_nitrogens = np.linalg.norm(carbons[:, None, :] - nitrogens[None, :, :], axis=2)
                deviation_carbonyl = np.abs(dists_carbonyl - IDEAL_CA_C_DISTANCE)
                deviations_from_all_nitrogens = np.abs(dists_to_all_nitrogens - IDEAL_CA_N_DISTANCE)
                best_n_deviation = np.min(deviations_from_all_nitrogens, axis=1)
                scores = deviation_carbonyl * CA_SCORE_CARBONYL_C + best_n_deviation * CA_SCORE_N
                # Exclude the carbonyl carbon itself
                scores[np.all(carbons == prev_carbonyl, axis = -1)] = np.inf
                scores[np.all(carbons == ca, axis = -1)] = np.inf  # Exclude the current alpha carbon as well
                # Find the best alpha carbon
                prev_ca_index = np.argmin(scores)
                prev_ca = carbons[prev_ca_index]
            else:
                # At this point, results will be garbage anyways so just return the nearest carbon to the alpha carbon
                candidates = carbons[np.any(carbons != ca, axis = -1)]
                candidates = candidates[np.any(candidates != carbonyl_c, axis = -1)]
                if candidates.size > 0:
                    prev_ca_index = np.argmin(np.linalg.norm(candidates - ca, axis=1))
                    prev_ca = candidates[prev_ca_index]
                else:
                    prev_ca = ca

    if not is_first:
        # Calculate the dihedral angle deviation
        b1 = prev_carbonyl - prev_ca
        b2 = n - prev_carbonyl
        b3 = ca - n
        omega_angle = dihedral(b1, b2, b3)
        omega_angle = rad2deg(omega_angle)
        trans_deviation = np.abs(np.abs(omega_angle) - 180.0)  # Ideally, omega should be 180 degrees
        cis_deviation = np.abs(omega_angle)  # Allow cis angles as well
    
        total_deviation = min(trans_deviation, cis_deviation)
    else:
        total_deviation = 0.0  # No deviation for the first frame

    # Find the next frame's nitrogen
    next_n = None  # Next frame's nitrogen
    next_ca = None  # Next frame's alpha carbon
    if not is_last:
        dists = np.linalg.norm(nitrogens - carbonyl_c, axis=1)
        deviation_from_ideal = np.abs(dists - IDEAL_AMIDE_C_DISTANCE)
        consideration_threshold = CONSIDERATION_DIST + np.min(deviation_from_ideal)
        num_to_consider = np.sum(deviation_from_ideal < consideration_threshold)
        if num_to_consider > 2:
            best_assignment = None
            best_deviation = np.inf
            ns_to_consider_indices = np.argsort(deviation_from_ideal)[:num_to_consider]
            ns_to_consider = nitrogens[ns_to_consider_indices]
            for next_n_candidate in ns_to_consider:
                assignment, deviation = compute_frame_with_last_n(
                    carbons, oxygens, nitrogens, prev_ca, prev_carbonyl, ca, n, carbonyl_c, next_n_candidate
                )
                if deviation < best_deviation:
                    best_assignment = assignment
                    best_deviation = deviation
            prev_ca, prev_carbonyl, ca, n, carbonyl_c, next_n, next_ca = best_assignment
        else:
            # Find the best nitrogen
            next_n_index = np.argmin(deviation_from_ideal)
            next_n = nitrogens[next_n_index]
            assignment, deviation = compute_frame_with_last_n(
                carbons, oxygens, nitrogens, prev_ca, prev_carbonyl, ca, n, carbonyl_c, next_n
            )
            prev_ca, prev_carbonyl, ca, n, carbonyl_c, next_n, next_ca = assignment

    if not is_last:
        # Calculate the dihedral angle deviation for the next frame's alpha carbon
        b1 = carbonyl_c - ca
        b2 = next_n - carbonyl_c
        b3 = next_ca - next_n
        omega_angle = dihedral(b1, b2, b3)
        omega_angle = rad2deg(omega_angle)
        trans_deviation = np.abs(np.abs(omega_angle) - 180.0)  # Ideally, omega should be 180 degrees
        cis_deviation = np.abs(omega_angle)  # Allow cis angles as well

        total_deviation += min(trans_deviation, cis_deviation)

    return (prev_ca, prev_carbonyl, ca, n, carbonyl_c, next_n, next_ca), total_deviation

def compute_frame_with_last_n(carbons, oxygens, nitrogens, prev_ca, prev_carbonyl, ca, n, carbonyl_c, next_n):
    # Get the next frame's alpha carbon
    dists = np.linalg.norm(carbons - next_n, axis=1)
    # Exclude the current carbonyl carbon and alpha carbon from consideration
    dists[np.all(carbons == carbonyl_c, axis=-1)] = np.inf
    dists[np.all(carbons == ca, axis=-1)] = np.inf
    deviation_from_ideal = np.abs(dists - IDEAL_CA_N_DISTANCE)
    consideration_threshold = CONSIDERATION_DIST + np.min(deviation_from_ideal)
    num_to_consider = np.sum(deviation_from_ideal < consideration_threshold)
    if num_to_consider > 2:
        to_consider = np.argsort(dists)[:num_to_consider]
        considered_scores = dists[to_consider]
        candidate_cas = carbons[to_consider]
        b1s = carbonyl_c - ca
        b2s = next_n - carbonyl_c
        b3s = candidate_cas - next_n
        omega_angles = dihedral(b1s, b2s, b3s)
        omega_angles = rad2deg(omega_angles)
        trans_deviations = np.abs(np.abs(omega_angles) - 180.0)  # Ideally, omega should be 180 degrees
        cis_deviations = np.abs(omega_angles)  # Allow cis angles as well
        deviations = np.minimum(trans_deviations, cis_deviations)
        # Find the best candidate based on the lowest deviation
        best_candidate_index = np.argmin(deviations)
        best_angle_deviation = deviations[best_candidate_index]
        best_angle = omega_angles[best_candidate_index]
        next_ca = candidate_cas[best_candidate_index]
    else:
        next_ca_index = np.argmin(deviation_from_ideal)
        next_ca = carbons[next_ca_index]
        
        # Calculate dihedral angle deviation for the next frame's alpha carbon
        b1 = carbonyl_c - ca
        b2 = next_n - carbonyl_c
        b3 = next_ca - next_n
        omega_angle = dihedral(b1, b2, b3)
        omega_angle = rad2deg(omega_angle)
        trans_deviation = np.abs(np.abs(omega_angle) - 180.0)
        cis_deviation = np.abs(omega_angle)
        best_angle_deviation = min(trans_deviation, cis_deviation)

    return (prev_ca, prev_carbonyl, ca, n, carbonyl_c, next_n, next_ca), best_angle_deviation