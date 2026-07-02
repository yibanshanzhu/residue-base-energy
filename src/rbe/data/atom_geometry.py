from __future__ import annotations

import numpy as np

from rbe.constants import DNA_BACKBONE_ATOMS
from rbe.data.structure_types import ResidueRecord


def heavy_atom_coords(residue: ResidueRecord) -> np.ndarray:
    coords = [
        atom.coord
        for atom in residue.atoms
        if atom.element.upper() != "H" and not atom.name.upper().startswith("H")
    ]
    if not coords:
        raise ValueError(f"Residue {residue.residue_id} has no heavy atoms.")
    return np.stack(coords).astype(np.float32)


def base_heavy_atom_coords(residue: ResidueRecord) -> np.ndarray:
    coords = [
        atom.coord
        for atom in residue.atoms
        if atom.element.upper() != "H"
        and not atom.name.upper().startswith("H")
        and atom.name.strip().upper() not in DNA_BACKBONE_ATOMS
    ]
    if not coords:
        raise ValueError(f"DNA residue {residue.residue_id} has no base heavy atoms.")
    return np.stack(coords).astype(np.float32)


def backbone_heavy_atom_coords(residue: ResidueRecord) -> np.ndarray:
    coords = [
        atom.coord
        for atom in residue.atoms
        if atom.element.upper() != "H"
        and not atom.name.upper().startswith("H")
        and atom.name.strip().upper() in DNA_BACKBONE_ATOMS
    ]
    if not coords:
        raise ValueError(f"DNA residue {residue.residue_id} has no backbone heavy atoms.")
    return np.stack(coords).astype(np.float32)


def ca_or_centroid(residue: ResidueRecord) -> np.ndarray:
    for atom in residue.atoms:
        if atom.name == "CA":
            return atom.coord.astype(np.float32)
    return heavy_atom_coords(residue).mean(axis=0).astype(np.float32)
