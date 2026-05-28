from __future__ import annotations

import numpy as np

from rbe.data.alignment import align_pwm_to_dna
from rbe.data.pdb import AtomRecord, ResidueRecord
from rbe.data.process_complex import _select_contact_constrained_alignment


def _dna(sequence: str, chain: str = "B") -> list[ResidueRecord]:
    return [
        ResidueRecord(
            resname=f"D{base}",
            chain=chain,
            resseq=i + 1,
            icode="",
            atoms=[],
        )
        for i, base in enumerate(sequence)
    ]


def _pwm(consensus: str) -> np.ndarray:
    order = "ACGT"
    pwm = np.full((len(consensus), 4), 0.01, dtype=np.float32)
    for i, base in enumerate(consensus):
        pwm[i, order.index(base)] = 0.97
    return pwm


def _atom(name: str, coord: tuple[float, float, float], element: str) -> AtomRecord:
    return AtomRecord(
        name=name,
        resname="",
        chain="",
        resseq=0,
        icode="",
        coord=np.asarray(coord, dtype=np.float32),
        element=element,
    )


def _dna_with_coords(sequence: str, contacted_index: int) -> list[ResidueRecord]:
    residues = []
    for i, base in enumerate(sequence):
        base_coord = (0.0, 0.0, 0.0) if i == contacted_index else (100.0 + i, 0.0, 0.0)
        residues.append(
            ResidueRecord(
                resname=f"D{base}",
                chain="B",
                resseq=i + 1,
                icode="",
                atoms=[
                    _atom("N1", base_coord, "N"),
                    _atom("P", (100.0 + i, 5.0, 0.0), "P"),
                ],
            )
        )
    return residues


def _dna_with_missing_base(
    sequence: str, contacted_index: int, missing_base_indices: set[int]
) -> list[ResidueRecord]:
    residues = []
    for i, base in enumerate(sequence):
        base_coord = (0.0, 0.0, 0.0) if i == contacted_index else (100.0 + i, 0.0, 0.0)
        atoms = [_atom("P", (100.0 + i, 5.0, 0.0), "P")]
        if i not in missing_base_indices:
            atoms.insert(0, _atom("N1", base_coord, "N"))
        residues.append(
            ResidueRecord(
                resname=f"D{base}",
                chain="B",
                resseq=i + 1,
                icode="",
                atoms=atoms,
            )
        )
    return residues


def _protein_near_origin() -> list[ResidueRecord]:
    return [
        ResidueRecord(
            resname="ARG",
            chain="A",
            resseq=1,
            icode="",
            atoms=[_atom("CZ", (0.0, 0.0, 0.0), "C")],
        )
    ]


def _low_ic_pwm() -> np.ndarray:
    return np.asarray(
        [
            [0.30, 0.20, 0.20, 0.30],
            [0.97, 0.01, 0.01, 0.01],
            [0.30, 0.20, 0.20, 0.30],
        ],
        dtype=np.float32,
    )


def test_align_pwm_to_forward_chain():
    alignment = align_pwm_to_dna(_pwm("ACG"), _dna("TTACGAA"))
    assert alignment.chain == "B"
    assert alignment.score_mode == "ic_log_likelihood"
    assert alignment.start == 2
    assert not alignment.reverse_complement
    assert alignment.aligned_sequence == "ACG"
    assert alignment.slot_to_dna_index.tolist() == [2, 3, 4]


def test_align_pwm_to_reverse_complement_chain():
    alignment = align_pwm_to_dna(_pwm("ACG"), _dna("TTCGTAA"))
    assert alignment.chain == "B"
    assert alignment.reverse_complement
    assert alignment.aligned_sequence == "ACG"
    assert alignment.slot_to_dna_index.tolist() == [4, 3, 2]


def test_ic_weighted_log_likelihood_prioritizes_informative_columns():
    alignment = align_pwm_to_dna(_low_ic_pwm(), _dna("TTATG"))
    assert alignment.aligned_sequence == "TAT"
    assert alignment.slot_to_dna_index.tolist() == [1, 2, 3]


def test_contact_constrained_alignment_skips_noncontact_sequence_match():
    args = type(
        "Args",
        (),
        {
            "alignment_score": "ic_log_likelihood",
            "base_contact_cutoff": 4.5,
            "backbone_contact_cutoff": 5.0,
            "alignment_min_base_pairs": 0,
            "alignment_min_contact_pairs": 1,
            "alignment_min_site_residues": 1,
        },
    )()
    alignment, candidate_count, contact_candidate_count = (
        _select_contact_constrained_alignment(
            pwm_target=_pwm("AAA"),
            protein=_protein_near_origin(),
            dna=_dna_with_coords("AAATTTAAA", contacted_index=6),
            args=args,
        )
    )
    assert candidate_count > contact_candidate_count >= 1
    assert alignment.start == 6
    assert not alignment.reverse_complement
    assert alignment.slot_to_dna_index.tolist() == [6, 7, 8]


def test_contact_constrained_alignment_skips_missing_base_atoms():
    args = type(
        "Args",
        (),
        {
            "alignment_score": "ic_log_likelihood",
            "base_contact_cutoff": 4.5,
            "backbone_contact_cutoff": 5.0,
            "alignment_min_base_pairs": 0,
            "alignment_min_contact_pairs": 1,
            "alignment_min_site_residues": 1,
        },
    )()
    alignment, _, contact_candidate_count = _select_contact_constrained_alignment(
        pwm_target=_pwm("AAA"),
        protein=_protein_near_origin(),
        dna=_dna_with_missing_base(
            "AAATTTAAA", contacted_index=6, missing_base_indices={0}
        ),
        args=args,
    )
    assert contact_candidate_count >= 1
    assert alignment.start == 6
    assert alignment.slot_to_dna_index.tolist() == [6, 7, 8]
