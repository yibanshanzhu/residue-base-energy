from __future__ import annotations

from rbe.data.pdb import parse_structure, select_dna_residues, select_protein_residues


def test_parse_minimal_mmcif_atom_site(tmp_path):
    cif = tmp_path / "toy.cif"
    cif.write_text(
        """data_toy
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 N N . ARG A A 1 10 ? 1.0 2.0 3.0
ATOM 2 C CA . ARG A A 1 10 ? 2.0 2.0 3.0
HETATM 3 P P . DA B B 1 1 ? 5.0 6.0 7.0
HETATM 4 N N1 . DA B B 1 1 ? 5.5 6.0 7.0
#
"""
    )

    residues = parse_structure(cif)
    protein = select_protein_residues(residues)
    dna = select_dna_residues(residues)
    assert len(protein) == 1
    assert protein[0].residue_id == "A:10"
    assert protein[0].resname == "ARG"
    assert len(dna) == 1
    assert dna[0].residue_id == "B:1"
    assert dna[0].resname == "DA"
