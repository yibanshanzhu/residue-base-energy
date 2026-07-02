from __future__ import annotations

from typing import Iterable, Optional, Sequence

from rbe.constants import AA3_TO_1, DNA_RESNAMES
from rbe.data.structure_types import ResidueRecord


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


def _chain_allowed(chain: str, chains: Optional[set[str]]) -> bool:
    return chains is None or chain in chains
