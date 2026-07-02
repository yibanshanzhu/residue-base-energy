from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rbe.data.alignment import (
    PWMAlignment,
    align_pwm_to_dna,
    enumerate_pwm_to_dna_alignments,
    enumerate_partial_pwm_to_dna_alignments,
)
from rbe.data.contact_labels import ContactCutoffs, compute_contact_labels
from rbe.data.structure_types import ResidueRecord


@dataclass(frozen=True)
class AlignmentContactConstraints:
    min_base_pairs: int = 0
    min_contact_pairs: int = 1
    min_site_residues: int = 1


@dataclass(frozen=True)
class AlignmentSelectionConfig:
    score_mode: str = "ic_log_likelihood"
    contact_policy: str = "require_contact"
    contact_cutoffs: ContactCutoffs = ContactCutoffs()
    contact_constraints: AlignmentContactConstraints = AlignmentContactConstraints()


def select_pwm_dna_alignment(
    pwm_target: np.ndarray,
    protein: list[ResidueRecord],
    dna: list[ResidueRecord],
    config: AlignmentSelectionConfig,
    manual_slot_to_dna_index: str | None = None,
    manual_dna_start_index: int | None = None,
) -> tuple[np.ndarray, dict]:
    motif_len = pwm_target.shape[0]
    manual_value = (manual_slot_to_dna_index or "").strip()

    if manual_value:
        indices = np.array(
            [int(item.strip()) for item in manual_value.split(",")],
            dtype=np.int64,
        )
        if indices.shape[0] != motif_len:
            raise ValueError(
                f"--slot-to-dna-index length {indices.shape[0]} != PWM length {motif_len}."
            )
        return indices, _manual_alignment_meta("manual_slot_to_dna_index", start=-1)

    if manual_dna_start_index is not None:
        indices = np.arange(
            manual_dna_start_index,
            manual_dna_start_index + motif_len,
            dtype=np.int64,
        )
        return indices, _manual_alignment_meta(
            "manual_dna_start_index", start=manual_dna_start_index
        )

    if config.contact_policy == "sequence_only":
        alignment = align_pwm_to_dna(pwm_target, dna, score_mode=config.score_mode)
        mode = "partial_auto_pwm_dna" if (alignment.slot_to_dna_index < 0).any() else "auto_pwm_dna"
        return alignment.slot_to_dna_index, _alignment_meta(
            alignment=alignment,
            mode=mode,
            candidate_count=0,
            contact_candidate_count=0,
        )

    alignment, candidate_count, contact_candidate_count, used_partial = (
        select_contact_constrained_alignment(
            pwm_target=pwm_target,
            protein=protein,
            dna=dna,
            config=config,
        )
    )
    return alignment.slot_to_dna_index, _alignment_meta(
        alignment=alignment,
        mode=(
            "partial_contact_constrained_pwm_dna"
            if used_partial
            else "contact_constrained_pwm_dna"
        ),
        candidate_count=candidate_count,
        contact_candidate_count=contact_candidate_count,
    )


def select_contact_constrained_alignment(
    pwm_target: np.ndarray,
    protein: list[ResidueRecord],
    dna: list[ResidueRecord],
    config: AlignmentSelectionConfig,
) -> tuple[PWMAlignment, int, int, bool]:
    candidates = enumerate_pwm_to_dna_alignments(
        pwm_target, dna, score_mode=config.score_mode
    )
    used_partial = False
    if not candidates:
        candidates = enumerate_partial_pwm_to_dna_alignments(
            pwm_target, dna, score_mode=config.score_mode
        )
        used_partial = True
    if not candidates:
        motif_len = pwm_target.shape[0]
        raise ValueError(
            f"No selected DNA chain can be aligned to PWM length {motif_len}."
        )

    valid = []
    for candidate in candidates:
        labels = compute_contact_labels(
            protein=protein,
            dna=dna,
            slot_to_dna_index=candidate.slot_to_dna_index,
            cutoffs=config.contact_cutoffs,
        )
        counts = labels.counts
        if contact_counts_pass(counts, config.contact_constraints):
            valid.append((candidate, counts))

    if not valid:
        best_sequence = max(candidates, key=lambda candidate: candidate.score)
        constraints = config.contact_constraints
        raise ValueError(
            "No PWM-DNA alignment candidate passed contact constraints. "
            f"candidate_count={len(candidates)} "
            f"min_contact_pairs={constraints.min_contact_pairs} "
            f"min_site_residues={constraints.min_site_residues} "
            f"min_base_pairs={constraints.min_base_pairs} "
            f"best_sequence_only=chain:{best_sequence.chain} "
            f"start:{best_sequence.start} rc:{best_sequence.reverse_complement} "
            f"score:{best_sequence.score:.6f}"
        )

    best, _ = max(valid, key=lambda item: item[0].score)
    return best, len(candidates), len(valid), used_partial


def contact_counts_pass(
    counts: dict[str, int], constraints: AlignmentContactConstraints
) -> bool:
    return (
        counts["A_contact_pos"] >= constraints.min_contact_pairs
        and counts["site_pos"] >= constraints.min_site_residues
        and counts["A_base_pos"] >= constraints.min_base_pairs
    )


def _manual_alignment_meta(mode: str, start: int) -> dict:
    return {
        "alignment_mode": mode,
        "alignment_score_mode": "manual",
        "alignment_chain": "",
        "alignment_start": start,
        "alignment_reverse_complement": False,
        "alignment_score": np.nan,
        "alignment_sequence": "",
        "alignment_candidate_count": 0,
        "alignment_contact_candidate_count": 0,
    }


def _alignment_meta(
    alignment: PWMAlignment,
    mode: str,
    candidate_count: int,
    contact_candidate_count: int,
) -> dict:
    return {
        "alignment_mode": mode,
        "alignment_score_mode": alignment.score_mode,
        "alignment_chain": alignment.chain,
        "alignment_start": alignment.start,
        "alignment_reverse_complement": alignment.reverse_complement,
        "alignment_score": alignment.score,
        "alignment_sequence": alignment.aligned_sequence,
        "alignment_candidate_count": candidate_count,
        "alignment_contact_candidate_count": contact_candidate_count,
    }
