from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from rbe.constants import BASE_TO_IDX, DNA_BASE_TO_1
from rbe.data.pwm import normalize_pwm
from rbe.data.structure_types import ResidueRecord


@dataclass(frozen=True)
class PWMAlignment:
    chain: str
    start: int
    reverse_complement: bool
    score_mode: str
    score: float
    slot_to_dna_index: np.ndarray
    aligned_sequence: str


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement_sequence(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def dna_residue_base(residue: ResidueRecord) -> str:
    try:
        return DNA_BASE_TO_1[residue.resname]
    except KeyError as exc:
        raise ValueError(f"Unsupported DNA residue name: {residue.resname}") from exc


def dna_chains_with_indices(
    dna_residues: list[ResidueRecord],
) -> list[tuple[str, str, np.ndarray]]:
    by_chain: "OrderedDict[str, list[tuple[int, str]]]" = OrderedDict()
    for idx, residue in enumerate(dna_residues):
        by_chain.setdefault(residue.chain, []).append((idx, dna_residue_base(residue)))

    chains = []
    for chain, values in by_chain.items():
        indices = np.asarray([idx for idx, _ in values], dtype=np.int64)
        sequence = "".join(base for _, base in values)
        chains.append((chain, sequence, indices))
    return chains


def pwm_information_content(pwm: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    pwm = normalize_pwm(pwm, eps=eps)
    return 2.0 + np.sum(pwm * np.log2(pwm), axis=1)


def _score_pwm_against_sequence(
    pwm: np.ndarray,
    sequence: str,
    score_mode: str,
    eps: float = 1e-8,
) -> float:
    if score_mode == "deeppbs_ic_pcc":
        weights = pwm_information_content(pwm)
        column_scores = []
        for row, base in enumerate(sequence):
            one_hot = np.zeros(4, dtype=np.float32)
            one_hot[BASE_TO_IDX[base]] = 1.0
            pwm_col = pwm[row].astype(np.float32)
            x = one_hot - one_hot.mean()
            y = pwm_col - pwm_col.mean()
            denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
            pcc = float(np.sum(x * y) / denom) if denom > eps else 0.0
            column_scores.append(pcc * float(weights[row]))
        return float(np.mean(column_scores))

    log_probs = []
    for row, base in enumerate(sequence):
        log_probs.append(float(np.log(pwm[row, BASE_TO_IDX[base]] + eps)))
    log_probs_arr = np.asarray(log_probs, dtype=np.float32)

    if score_mode == "log_likelihood":
        return float(log_probs_arr.mean())
    if score_mode == "ic_log_likelihood":
        weights = pwm_information_content(pwm)
        weight_sum = float(weights.sum())
        if weight_sum <= eps:
            return float(log_probs_arr.mean())
        return float(np.sum(weights * log_probs_arr) / weight_sum)
    raise ValueError(
        "Unsupported alignment score mode. Use 'ic_log_likelihood', "
        "'log_likelihood', or 'deeppbs_ic_pcc'."
    )


def align_pwm_to_dna(
    pwm: np.ndarray,
    dna_residues: list[ResidueRecord],
    score_mode: str = "ic_log_likelihood",
) -> PWMAlignment:
    candidates = enumerate_pwm_to_dna_alignments(
        pwm, dna_residues, score_mode=score_mode
    )
    if not candidates:
        motif_len = normalize_pwm(pwm).shape[0]
        lengths = {chain: len(seq) for chain, seq, _ in dna_chains_with_indices(dna_residues)}
        raise ValueError(
            f"No selected DNA chain is long enough for PWM length {motif_len}. "
            f"DNA chain lengths: {lengths}"
        )
    return max(candidates, key=lambda candidate: candidate.score)


def enumerate_pwm_to_dna_alignments(
    pwm: np.ndarray,
    dna_residues: list[ResidueRecord],
    score_mode: str = "ic_log_likelihood",
) -> list[PWMAlignment]:
    pwm = normalize_pwm(pwm)
    motif_len = pwm.shape[0]
    alignments = []

    for chain, sequence, indices in dna_chains_with_indices(dna_residues):
        if len(sequence) < motif_len:
            continue
        candidates = [
            (False, sequence, indices),
            (True, reverse_complement_sequence(sequence), indices[::-1]),
        ]
        for is_rc, oriented_sequence, oriented_indices in candidates:
            for start in range(0, len(oriented_sequence) - motif_len + 1):
                window = oriented_sequence[start : start + motif_len]
                score = _score_pwm_against_sequence(
                    pwm, window, score_mode=score_mode
                )
                slot_to_dna_index = oriented_indices[start : start + motif_len].copy()
                candidate = PWMAlignment(
                    chain=chain,
                    start=start,
                    reverse_complement=is_rc,
                    score_mode=score_mode,
                    score=score,
                    slot_to_dna_index=slot_to_dna_index,
                    aligned_sequence=window,
                )
                alignments.append(candidate)

    return alignments
