'''
Constants for reconstruction algorithm
'''

MAX_CA_N_DISTANCE = 2.0  # Maximum distance between alpha carbon and nitrogen in Angstroms
IDEAL_CA_N_DISTANCE = 1.46  # Ideal distance between alpha carbon and nitrogen in Angstroms
CARBONYL_OXYGEN_DISTANCE = 2.0  # Maximum distance between carbonyl carbon and oxygen in Angstroms
MAX_CA_C_DISTANCE = 2.0  # Maximum distance between alpha carbon and carbonyl carbon in Angstroms
IDEAL_CA_C_DISTANCE = 1.52  # Ideal distance between alpha carbon and carbonyl carbon in Angstroms
MAX_AMIDE_C_DISTANCE = 2.0  # Maximum distance between amide carbon and nitrogen in Angstroms
IDEAL_AMIDE_C_DISTANCE = 1.33  # Ideal distance between amide carbon and nitrogen in Angstroms

CONSIDERATION_DIST = 0.1  # Distance threshold for computing omega angle to compare similarly-scored ca residues

CA_SCORE_CARBONYL_C = 0.5  # Score for carbonyl carbon in alpha carbon assignment
CA_SCORE_N = 0.5  # Score for nitrogen in alpha carbon assignment

element_list_stripped = [
    "H",
    "Li",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Mb",
    "Pd",
    "Ag",
]
elem = [e.upper() for e in element_list_stripped]
PAD_ID = len(elem)