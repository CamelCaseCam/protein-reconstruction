"""
Generates representative side‑chain **rotamers** by clustering bond / angle /
dihedral vectors extracted from PDB structures under ``training/pdbs``.

Key points vs the draft you saw earlier
--------------------------------------
* **No PeptideBuilder / copy_all_geometry** – we use a *minimal* ``SimpleGeo``
  class that stores heavy‑atom Cartesian coordinates direct from each residue.
* All geometric features (bond lengths, bond angles, dihedrals) are computed
  **purely with NumPy** – see the `dist`, `angle`, and `dihedral` helpers.
* Every `_get_*` helper is now fully implemented – nothing left as a placeholder.
* Safe: any residue missing a required atom is silently skipped, so vectors fed
  to K‑means never contain NaNs.

Copy‑paste this whole file; it stands alone except for your existing Python
packages (*Bio.PDB*, *NumPy*, *SciPy*, *tqdm*).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
from Bio.PDB import PDBParser, Residue
from scipy.cluster.vq import kmeans2
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Basic linear‑algebra helpers (angles in **radians**; dihedral is signed)
# ──────────────────────────────────────────────────────────────────────────────


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n != 0 else v


def dist(a: np.ndarray, b: np.ndarray) -> float:  # |b‑a|
    return _norm(b - a)


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:  # ∠ABC
    v1, v2 = a - b, c - b
    cos_theta = np.dot(_unit(v1), _unit(v2))
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Signed torsion angle between planes (p1,p2,p3) and (p2,p3,p4)."""
    b0, b1, b2 = p2 - p1, p3 - p2, p4 - p3
    b1_u = _unit(b1)
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u
    x = np.dot(_unit(v), _unit(w))
    y = np.dot(np.cross(b1_u, v), _unit(w))
    return float(np.arctan2(y, x))


# ──────────────────────────────────────────────────────────────────────────────
# Minimal container holding heavy‑atom coordinates
# ──────────────────────────────────────────────────────────────────────────────


class SimpleGeo:
    """A barebones replacement for ``PeptideBuilder.Geometry``"""

    def __init__(self, residue: Residue):
        self.atoms: Dict[str, np.ndarray] = {
            atom.get_name().strip(): np.asarray(atom.get_coord(), dtype=np.float64)
            for atom in residue
            if atom.element != "H"  # skip hydrogens
        }


# ──────────────────────────────────────────────────────────────────────────────
# Constants / lookup tables
# ──────────────────────────────────────────────────────────────────────────────

THREE_LETTER_TO_SINGLE_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

AMINO_NUM_COORDS: Dict[str, int] = {
    "GLY": 0,
    "ALA": 3,
    "SER": 6,
    "CYS": 6,
    "VAL": 9,
    "ILE": 12,
    "LEU": 12,
    "THR": 9,
    "ARG": 21,
    "LYS": 15,
    "ASP": 12,
    "ASN": 12,
    "GLU": 15,
    "GLN": 15,
    "MET": 12,
    "HIS": 18,
    "PRO": 9,
    "PHE": 21,
    "TYR": 24,
    "TRP": 30,
}


# ──────────────────────────────────────────────────────────────────────────────
# Per‑residue geometry‑to‑vector helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_ala(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
    ])


def _get_ser(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["OG"]),
        np.sin(angle(p["CA"], p["CB"], p["OG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["OG"])),
    ])


def _get_cys(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["SG"]),
        np.sin(angle(p["CA"], p["CB"], p["SG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["SG"])),
    ])


def _get_val(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG1"]),
        np.sin(angle(p["CA"], p["CB"], p["CG1"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG1"])),
        dist(p["CB"], p["CG2"]),
        np.sin(angle(p["CA"], p["CB"], p["CG2"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG2"])),
    ])


def _get_ile(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG1"]),
        np.sin(angle(p["CA"], p["CB"], p["CG1"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG1"])),
        dist(p["CB"], p["CG2"]),
        np.sin(angle(p["CA"], p["CB"], p["CG2"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG2"])),
        dist(p["CG1"], p["CD1"]),
        np.sin(angle(p["CB"], p["CG1"], p["CD1"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG1"], p["CD1"])),
    ])


def _get_leu(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["CD1"]),
        np.sin(angle(p["CB"], p["CG"], p["CD1"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD1"])),
        dist(p["CG"], p["CD2"]),
        np.sin(angle(p["CB"], p["CG"], p["CD2"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD2"])),
    ])


def _get_thr(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["OG1"]),
        np.sin(angle(p["CA"], p["CB"], p["OG1"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["OG1"])),
        dist(p["CB"], p["CG2"]),
        np.sin(angle(p["CA"], p["CB"], p["CG2"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG2"])),
    ])


def _get_arg(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["CD"]),
        np.sin(angle(p["CB"], p["CG"], p["CD"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD"])),
        dist(p["CD"], p["NE"]),
        np.sin(angle(p["CG"], p["CD"], p["NE"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD"], p["NE"])),
        dist(p["NE"], p["CZ"]),
        np.sin(angle(p["CD"], p["NE"], p["CZ"])),
        np.sin(dihedral(p["CG"], p["CD"], p["NE"], p["CZ"])),
        dist(p["CZ"], p["NH1"]),
        np.sin(angle(p["NE"], p["CZ"], p["NH1"])),
        np.sin(dihedral(p["CD"], p["NE"], p["CZ"], p["NH1"])),
        dist(p["CZ"], p["NH2"]),
        np.sin(angle(p["NE"], p["CZ"], p["NH2"])),
        np.sin(dihedral(p["CD"], p["NE"], p["CZ"], p["NH2"])),
    ])


def _get_lys(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["CD"]),
        np.sin(angle(p["CB"], p["CG"], p["CD"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD"])),
        dist(p["CD"], p["CE"]),
        np.sin(angle(p["CG"], p["CD"], p["CE"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD"], p["CE"])),
        dist(p["CE"], p["NZ"]),
        np.sin(angle(p["CD"], p["CE"], p["NZ"])),
        np.sin(dihedral(p["CG"], p["CD"], p["CE"], p["NZ"])),
    ])


def _get_asp(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["OD1"]),
        np.sin(angle(p["CB"], p["CG"], p["OD1"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["OD1"])),
        dist(p["CG"], p["OD2"]),
        np.sin(angle(p["CB"], p["CG"], p["OD2"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["OD2"])),
    ])


def _get_asn(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["OD1"]),
        np.sin(angle(p["CB"], p["CG"], p["OD1"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["OD1"])),
        dist(p["CG"], p["ND2"]),
        np.sin(angle(p["CB"], p["CG"], p["ND2"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["ND2"])),
    ])


def _get_glu(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["CD"]),
        np.sin(angle(p["CB"], p["CG"], p["CD"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD"])),
        dist(p["CD"], p["OE1"]),
        np.sin(angle(p["CG"], p["CD"], p["OE1"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD"], p["OE1"])),
        dist(p["CD"], p["OE2"]),
        np.sin(angle(p["CG"], p["CD"], p["OE2"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD"], p["OE2"])),
    ])


def _get_gln(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["CD"]),
        np.sin(angle(p["CB"], p["CG"], p["CD"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD"])),
        dist(p["CD"], p["OE1"]),
        np.sin(angle(p["CG"], p["CD"], p["OE1"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD"], p["OE1"])),
        dist(p["CD"], p["NE2"]),
        np.sin(angle(p["CG"], p["CD"], p["NE2"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD"], p["NE2"])),
    ])


def _get_met(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["SD"]),
        np.sin(angle(p["CB"], p["CG"], p["SD"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["SD"])),
        dist(p["SD"], p["CE"]),
        np.sin(angle(p["CG"], p["SD"], p["CE"])),
        np.sin(dihedral(p["CB"], p["CG"], p["SD"], p["CE"])),
    ])


def _get_his(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["ND1"]),
        np.sin(angle(p["CB"], p["CG"], p["ND1"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["ND1"])),
        dist(p["CG"], p["CD2"]),
        np.sin(angle(p["CB"], p["CG"], p["CD2"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD2"])),
        dist(p["ND1"], p["CE1"]),
        np.sin(angle(p["CG"], p["ND1"], p["CE1"])),
        np.sin(dihedral(p["CB"], p["CG"], p["ND1"], p["CE1"])),
        dist(p["CD2"], p["NE2"]),
        np.sin(angle(p["CG"], p["CD2"], p["NE2"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD2"], p["NE2"])),
    ])


def _get_pro(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["CD"]),
        np.sin(angle(p["CB"], p["CG"], p["CD"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD"])),
    ])


def _get_phe(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["CD1"]),
        np.sin(angle(p["CB"], p["CG"], p["CD1"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD1"])),
        dist(p["CG"], p["CD2"]),
        np.sin(angle(p["CB"], p["CG"], p["CD2"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD2"])),
        dist(p["CD1"], p["CE1"]),
        np.sin(angle(p["CG"], p["CD1"], p["CE1"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD1"], p["CE1"])),
        dist(p["CD2"], p["CE2"]),
        np.sin(angle(p["CG"], p["CD2"], p["CE2"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD2"], p["CE2"])),
        dist(p["CE1"], p["CZ"]),
        np.sin(angle(p["CD1"], p["CE1"], p["CZ"])),
        np.sin(dihedral(p["CG"], p["CD1"], p["CE1"], p["CZ"])),
    ])


def _get_tyr(p):
    return np.array([
        *_get_phe(p),  # first 21 scalars identical to Phe
        dist(p["CZ"], p["OH"]),
        np.sin(angle(p["CE1"], p["CZ"], p["OH"])),
        np.sin(dihedral(p["CD1"], p["CE1"], p["CZ"], p["OH"])),
    ])


def _get_trp(p):
    return np.array([
        dist(p["CA"], p["CB"]),
        np.sin(angle(p["C"],  p["CA"], p["CB"])),
        np.sin(dihedral(p["N"], p["C"], p["CA"], p["CB"])),
        dist(p["CB"], p["CG"]),
        np.sin(angle(p["CA"], p["CB"], p["CG"])),
        np.sin(dihedral(p["N"], p["CA"], p["CB"], p["CG"])),
        dist(p["CG"], p["CD1"]),
        np.sin(angle(p["CB"], p["CG"], p["CD1"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD1"])),
        dist(p["CG"], p["CD2"]),
        np.sin(angle(p["CB"], p["CG"], p["CD2"])),
        np.sin(dihedral(p["CA"], p["CB"], p["CG"], p["CD2"])),
        dist(p["CD1"], p["NE1"]),
        np.sin(angle(p["CG"], p["CD1"], p["NE1"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD1"], p["NE1"])),
        dist(p["CD2"], p["CE2"]),
        np.sin(angle(p["CG"], p["CD2"], p["CE2"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD2"], p["CE2"])),
        dist(p["CD2"], p["CE3"]),
        np.sin(angle(p["CG"], p["CD2"], p["CE3"])),
        np.sin(dihedral(p["CB"], p["CG"], p["CD2"], p["CE3"])),
        dist(p["CE2"], p["CZ2"]),
        np.sin(angle(p["CD2"], p["CE2"], p["CZ2"])),
        np.sin(dihedral(p["CG"], p["CD2"], p["CE2"], p["CZ2"])),
        dist(p["CE3"], p["CZ3"]),
        np.sin(angle(p["CD2"], p["CE3"], p["CZ3"])),
        np.sin(dihedral(p["CG"], p["CD2"], p["CE3"], p["CZ3"])),
        dist(p["CZ2"], p["CH2"]),
        np.sin(angle(p["CE2"], p["CZ2"], p["CH2"])),
        np.sin(dihedral(p["CD2"], p["CE2"], p["CZ2"], p["CH2"])),
    ])

# ──────────────────────────────────────────────────────────────────────────────
# Dispatch table
# ──────────────────────────────────────────────────────────────────────────────

GET_COORD_DICT = {
    "ALA": lambda g: _get_ala(g.atoms),
    "ARG": lambda g: _get_arg(g.atoms),
    "ASN": lambda g: _get_asn(g.atoms),
    "ASP": lambda g: _get_asp(g.atoms),
    "CYS": lambda g: _get_cys(g.atoms),
    "GLN": lambda g: _get_gln(g.atoms),
    "GLU": lambda g: _get_glu(g.atoms),
    "HIS": lambda g: _get_his(g.atoms),
    "ILE": lambda g: _get_ile(g.atoms),
    "LEU": lambda g: _get_leu(g.atoms),
    "LYS": lambda g: _get_lys(g.atoms),
    "MET": lambda g: _get_met(g.atoms),
    "PHE": lambda g: _get_phe(g.atoms),
    "PRO": lambda g: _get_pro(g.atoms),
    "SER": lambda g: _get_ser(g.atoms),
    "THR": lambda g: _get_thr(g.atoms),
    "TRP": lambda g: _get_trp(g.atoms),
    "TYR": lambda g: _get_tyr(g.atoms),
    "VAL": lambda g: _get_val(g.atoms),
}


# ──────────────────────────────────────────────────────────────────────────────
# PDB → list[SimpleGeo]
# ──────────────────────────────────────────────────────────────────────────────

def get_file_geos(file_path: str) -> Dict[str, List[SimpleGeo]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", file_path)
    geos = {k: [] for k in AMINO_NUM_COORDS}

    for model in structure:
        for chain in model:
            for residue in chain.get_residues():
                if residue.id[0] != " " or not residue.has_id("CA"):
                    continue
                resn = residue.get_resname().upper()
                if resn == "GLY":
                    continue
                geo = SimpleGeo(residue)
                try:
                    _ = GET_COORD_DICT[resn](geo)  # test completeness
                except (KeyError, ValueError):
                    continue
                geos[resn].append(geo)

    return geos


def get_all_geos(pdb_dir: str = "training/pdbs") -> Dict[str, List[SimpleGeo]]:
    geos = {k: [] for k in AMINO_NUM_COORDS}
    for fname in tqdm(os.listdir(pdb_dir), desc="Parsing PDBs"):
        if not fname.endswith(".pdb"):
            continue
        file_geos = get_file_geos(os.path.join(pdb_dir, fname))
        for resn, lst in file_geos.items():
            geos[resn].extend(lst)
    return geos


# ──────────────────────────────────────────────────────────────────────────────
# Vectorisation + clustering
# ──────────────────────────────────────────────────────────────────────────────

def geos_to_vector(geos: List[SimpleGeo], amino: str) -> np.ndarray:
    dim = AMINO_NUM_COORDS[amino]
    if dim == 0:
        return np.zeros((0, 0))
    vecs = []
    for g in geos:
        try:
            v = GET_COORD_DICT[amino](g)
            if v.shape[0] == dim and not np.any(np.isnan(v)):
                vecs.append(v)
        except KeyError:
            continue
    return np.stack(vecs) if vecs else np.zeros((0, dim))


def generate_rotamers(N: int = 15, pdb_dir: str = "training/pdbs") -> Dict[str, np.ndarray]:
    geos = get_all_geos(pdb_dir)
    rotamers: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros((0, 0)))
    for resn, res_geos in geos.items():
        if resn == "GLY" or not res_geos:
            continue
        vec = geos_to_vector(res_geos, resn)
        if vec.shape[0] == 0:
            continue
        print(f"Clustering {resn}: {vec.shape[0]} samples → {N} rotamers")
        k = min(N, vec.shape[0])
        centers, _ = kmeans2(vec, k, iter=20, minit="points")
        rotamers[resn] = centers
    return rotamers
