from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from rbe.constants import AA3_TO_1, DNA_BACKBONE_ATOMS, DNA_RESNAMES


@dataclass(frozen=True)
class AtomRecord:
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    coord: np.ndarray
    element: str


@dataclass
class ResidueRecord:
    resname: str
    chain: str
    resseq: int
    icode: str
    atoms: list[AtomRecord]

    @property
    def residue_id(self) -> str:
        suffix = self.icode if self.icode else ""
        return f"{self.chain}:{self.resseq}{suffix}"


def _infer_element(atom_name: str) -> str:
    stripped = atom_name.strip()
    if not stripped:
        return ""
    if stripped[0].isdigit() and len(stripped) > 1:
        return stripped[1].upper()
    return stripped[0].upper()


def parse_pdb(path: str | Path) -> list[ResidueRecord]:
    residues: "OrderedDict[tuple[str, int, str, str], ResidueRecord]" = OrderedDict()
    with Path(path).open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "_"
            resseq = int(line[22:26])
            icode = line[26].strip()
            coord = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            element = element or _infer_element(atom_name)
            atom = AtomRecord(atom_name, resname, chain, resseq, icode, coord, element)
            key = (chain, resseq, icode, resname)
            if key not in residues:
                residues[key] = ResidueRecord(resname, chain, resseq, icode, [])
            residues[key].atoms.append(atom)
    return list(residues.values())


def _chain_allowed(chain: str, chains: Optional[set[str]]) -> bool:
    return chains is None or chain in chains


def parse_chain_list(value: Optional[str]) -> Optional[set[str]]:
    if value is None or value.strip() == "":
        return None
    return {item.strip() or "_" for item in value.split(",")}


def is_protein_residue(residue: ResidueRecord) -> bool:
    return residue.resname in AA3_TO_1


def is_dna_residue(residue: ResidueRecord) -> bool:
    return residue.resname in DNA_RESNAMES


def select_protein_residues(
    residues: Sequence[ResidueRecord], chains: Optional[set[str]] = None
) -> list[ResidueRecord]:
    return [r for r in residues if is_protein_residue(r) and _chain_allowed(r.chain, chains)]


def select_dna_residues(
    residues: Sequence[ResidueRecord], chains: Optional[set[str]] = None
) -> list[ResidueRecord]:
    return [r for r in residues if is_dna_residue(r) and _chain_allowed(r.chain, chains)]


def residue_one_letter(residue: ResidueRecord) -> str:
    return AA3_TO_1[residue.resname]


def residue_sequence(residues: Iterable[ResidueRecord]) -> str:
    return "".join(residue_one_letter(residue) for residue in residues)


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
