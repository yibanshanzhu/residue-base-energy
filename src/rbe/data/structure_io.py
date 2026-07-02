from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import shlex
from typing import Sequence

import numpy as np

from rbe.data.structure_types import AtomRecord, ResidueRecord


def parse_structure(path: str | Path) -> list[ResidueRecord]:
    structure_path = Path(path)
    suffix = structure_path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return parse_mmcif(structure_path)
    return parse_pdb(structure_path)


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


def parse_mmcif(path: str | Path) -> list[ResidueRecord]:
    residues: "OrderedDict[tuple[str, int, str, str], ResidueRecord]" = OrderedDict()
    tokens = _tokenize_mmcif(Path(path))
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token != "loop_":
            i += 1
            continue

        i += 1
        tags = []
        while i < len(tokens) and tokens[i].startswith("_"):
            tags.append(tokens[i])
            i += 1

        if not tags:
            continue
        if not all(tag.startswith("_atom_site.") for tag in tags):
            while i < len(tokens) and tokens[i] != "loop_":
                i += 1
            continue

        tag_index = {tag: idx for idx, tag in enumerate(tags)}
        width = len(tags)
        while i + width <= len(tokens):
            if tokens[i] == "loop_" or tokens[i].startswith("data_"):
                break
            if tokens[i].startswith("_") and (len(tokens) - i) < width:
                break

            row = tokens[i : i + width]
            i += width
            atom = _atom_from_mmcif_row(row, tag_index)
            if atom is None:
                continue
            key = (atom.chain, atom.resseq, atom.icode, atom.resname)
            if key not in residues:
                residues[key] = ResidueRecord(
                    atom.resname, atom.chain, atom.resseq, atom.icode, []
                )
            residues[key].atoms.append(atom)

    return list(residues.values())


def _infer_element(atom_name: str) -> str:
    stripped = atom_name.strip()
    if not stripped:
        return ""
    if stripped[0].isdigit() and len(stripped) > 1:
        return stripped[1].upper()
    return stripped[0].upper()


def _tokenize_mmcif(path: Path) -> list[str]:
    tokens: list[str] = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            i += 1
            text_lines = []
            while i < len(lines) and not lines[i].startswith(";"):
                text_lines.append(lines[i])
                i += 1
            tokens.append("\n".join(text_lines))
            i += 1
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        lexer = shlex.shlex(stripped, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens.extend(list(lexer))
        i += 1
    return tokens


def _atom_from_mmcif_row(
    row: list[str], tag_index: dict[str, int]
) -> AtomRecord | None:
    group = _mmcif_value(row, tag_index, "_atom_site.group_PDB").upper()
    if group not in {"ATOM", "HETATM"}:
        return None

    altloc = _mmcif_value(row, tag_index, "_atom_site.label_alt_id")
    if altloc not in {"", ".", "?", "A"}:
        return None

    atom_name = _first_mmcif_value(
        row,
        tag_index,
        ["_atom_site.auth_atom_id", "_atom_site.label_atom_id"],
    )
    resname = _first_mmcif_value(
        row,
        tag_index,
        ["_atom_site.auth_comp_id", "_atom_site.label_comp_id"],
    ).upper()
    chain = _first_mmcif_value(
        row,
        tag_index,
        ["_atom_site.auth_asym_id", "_atom_site.label_asym_id"],
    )
    if chain in {"", ".", "?"}:
        chain = "_"
    seq_value = _first_mmcif_value(
        row,
        tag_index,
        ["_atom_site.auth_seq_id", "_atom_site.label_seq_id"],
    )
    if seq_value in {"", ".", "?"}:
        return None
    try:
        resseq = int(float(seq_value))
    except ValueError:
        return None

    icode = _mmcif_value(row, tag_index, "_atom_site.pdbx_PDB_ins_code")
    if icode in {".", "?"}:
        icode = ""

    try:
        coord = np.array(
            [
                float(_mmcif_value(row, tag_index, "_atom_site.Cartn_x")),
                float(_mmcif_value(row, tag_index, "_atom_site.Cartn_y")),
                float(_mmcif_value(row, tag_index, "_atom_site.Cartn_z")),
            ],
            dtype=np.float32,
        )
    except ValueError:
        return None

    element = _mmcif_value(row, tag_index, "_atom_site.type_symbol").upper()
    element = element or _infer_element(atom_name)
    return AtomRecord(atom_name, resname, chain, resseq, icode, coord, element)


def _mmcif_value(row: list[str], tag_index: dict[str, int], tag: str) -> str:
    idx = tag_index.get(tag)
    if idx is None or idx >= len(row):
        return ""
    value = row[idx]
    return "" if value in {".", "?"} else value


def _first_mmcif_value(
    row: list[str], tag_index: dict[str, int], tags: Sequence[str]
) -> str:
    for tag in tags:
        value = _mmcif_value(row, tag_index, tag)
        if value:
            return value
    return ""
