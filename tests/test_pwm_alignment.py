from __future__ import annotations

import numpy as np

from rbe.data.alignment import align_pwm_to_dna
from rbe.data.pdb import ResidueRecord


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
