'''
Constants for reconstruction algorithm
'''

MAX_CA_N_DISTANCE = 2.0  # Maximum distance between alpha carbon and nitrogen in Angstroms
CARBONYL_OXYGEN_DISTANCE = 2.0  # Maximum distance between carbonyl carbon and oxygen in Angstroms
MAX_CA_C_DISTANCE = 2.0  # Maximum distance between alpha carbon and carbonyl carbon in Angstroms
MAX_AMIDE_C_DISTANCE = 2.0  # Maximum distance between amide carbon and nitrogen in Angstroms

CA_SCORE_CARBONYL_C = 0.5  # Score for carbonyl carbon in alpha carbon assignment
CA_SCORE_N = 0.5  # Score for nitrogen in alpha carbon assignment