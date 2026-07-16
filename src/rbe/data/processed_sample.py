from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rbe.data.alignment import reverse_complement_sequence
from rbe.data.alignment_selection import AlignmentSelectionConfig, select_pwm_dna_alignment
from rbe.data.atom_geometry import ca_or_centroid
from rbe.data.contact_labels import ContactCutoffs, compute_contact_labels
from rbe.data.esm import extract_esm2_t33_hidden
from rbe.data.features import build_residue_graph
from rbe.data.pwm import canonicalize_pwm, read_pwm
from rbe.data.residue_select import (
    parse_chain_list,
    residue_one_letter,
    residue_sequence,
    select_dna_residues,
    select_protein_residues,
)
from rbe.data.structure_io import parse_structure


@dataclass(frozen=True)
class ComplexProcessingConfig:
    structure_path: str | Path
    pwm_path: str | Path
    protein_chains: str | None = None
    dna_chains: str | None = None
    manual_dna_start_index: int | None = None
    manual_slot_to_dna_index: str | None = None
    alignment: AlignmentSelectionConfig = AlignmentSelectionConfig()
    esm_npy: str | Path | None = None
    device: str = "cpu"
    ca_cutoff: float = 14.0
    num_rbf: int = 16
    rbf_max_distance: float = 20.0


@dataclass(frozen=True)
class ProcessedComplexSample:
    arrays: dict[str, np.ndarray]
    n_protein_residues: int
    motif_len: int
    n_edges: int
    label_counts: dict[str, int]
    alignment_meta: dict


def build_processed_complex_sample(
    config: ComplexProcessingConfig,
) -> ProcessedComplexSample:
    residues = parse_structure(config.structure_path)
    protein = select_protein_residues(
        residues, parse_chain_list(config.protein_chains)
    )
    dna = select_dna_residues(residues, parse_chain_list(config.dna_chains))
    if not protein:
        raise ValueError("No protein residues found. Check --protein-chains.")
    if not dna:
        raise ValueError("No DNA residues found. Check --dna-chains.")

    pwm_target = read_pwm(config.pwm_path)
    motif_len = pwm_target.shape[0]
    slot_to_dna_index, alignment_meta = select_pwm_dna_alignment(
        pwm_target=pwm_target,
        protein=protein,
        dna=dna,
        config=config.alignment,
        manual_slot_to_dna_index=config.manual_slot_to_dna_index,
        manual_dna_start_index=config.manual_dna_start_index,
    )
    _validate_slot_indices(slot_to_dna_index, len(dna))

    residue_ids = np.asarray([residue.residue_id for residue in protein])
    residue_aa = np.asarray([residue_one_letter(residue) for residue in protein])
    residue_xyz = np.stack([ca_or_centroid(residue) for residue in protein]).astype(
        np.float32
    )
    residue_edges, edge_attr = build_residue_graph(
        residue_xyz,
        cutoff=config.ca_cutoff,
        num_rbf=config.num_rbf,
        max_distance=config.rbf_max_distance,
    )

    sequence = residue_sequence(protein)
    if config.esm_npy:
        esm2_repr = np.load(config.esm_npy).astype(np.float32)
    else:
        esm2_repr = extract_esm2_t33_hidden(sequence, device=config.device)
    if esm2_repr.shape != (len(protein), 1280):
        raise ValueError(
            f"esm2_repr must have shape {(len(protein), 1280)}, got {esm2_repr.shape}."
        )

    labels = compute_contact_labels(
        protein=protein,
        dna=dna,
        slot_to_dna_index=slot_to_dna_index,
        cutoffs=ContactCutoffs(
            base=config.alignment.contact_cutoffs.base,
            backbone=config.alignment.contact_cutoffs.backbone,
        ),
    )

    pwm_target, slot_to_dna_index, slot_arrays, canonical_rc = (
        _canonicalize_sample_slots(
            pwm_target,
            slot_to_dna_index,
            {
                "pwm_mask": labels.pwm_mask,
                "A_base_label": labels.A_base_label,
                "A_base_mask": labels.A_base_mask,
                "A_backbone_label": labels.A_backbone_label,
                "A_contact_label": labels.A_contact_label,
            },
        )
    )
    alignment_meta = _canonical_alignment_meta(alignment_meta, canonical_rc)

    arrays = {
        "residue_ids": residue_ids,
        "residue_aa": residue_aa,
        "residue_xyz": residue_xyz,
        "residue_edges": residue_edges,
        "edge_attr": edge_attr,
        "esm2_repr": esm2_repr,
        "pwm_target": pwm_target,
        "pwm_mask": slot_arrays["pwm_mask"],
        "A_label": slot_arrays["A_base_label"],
        "A_base_label": slot_arrays["A_base_label"],
        "A_base_mask": slot_arrays["A_base_mask"],
        "A_backbone_label": slot_arrays["A_backbone_label"],
        "A_contact_label": slot_arrays["A_contact_label"],
        "site_label": labels.site_label,
        "slot_to_dna_index": slot_to_dna_index,
        "pwm_orientation": np.asarray("canonical"),
        **_alignment_arrays(alignment_meta),
    }
    return ProcessedComplexSample(
        arrays=arrays,
        n_protein_residues=len(protein),
        motif_len=motif_len,
        n_edges=residue_edges.shape[1],
        label_counts=labels.counts,
        alignment_meta=alignment_meta,
    )


def write_processed_complex_sample(
    output_path: str | Path, sample: ProcessedComplexSample
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **sample.arrays)


def format_processed_complex_summary(
    output_path: str | Path, sample: ProcessedComplexSample
) -> str:
    counts = sample.label_counts
    meta = sample.alignment_meta
    return (
        f"wrote {Path(output_path)} "
        f"N={sample.n_protein_residues} M={sample.motif_len} E={sample.n_edges} "
        f"A_base_pos={counts['A_base_pos']} "
        f"A_backbone_pos={counts['A_backbone_pos']} "
        f"A_contact_pos={counts['A_contact_pos']} "
        f"site_pos={counts['site_pos']} "
        f"alignment={meta['alignment_mode']} chain={meta['alignment_chain']} "
        f"start={meta['alignment_start']} rc={meta['alignment_reverse_complement']} "
        f"canonical_rc={meta['canonical_reverse_complement']} "
        f"score_mode={meta['alignment_score_mode']} "
        f"contact_candidates={meta['alignment_contact_candidate_count']}/"
        f"{meta['alignment_candidate_count']}"
    )


def _validate_slot_indices(slot_to_dna_index: np.ndarray, dna_len: int) -> None:
    if slot_to_dna_index.size == 0:
        raise ValueError("slot_to_dna_index is empty.")
    visible = slot_to_dna_index[slot_to_dna_index >= 0]
    if visible.size == 0:
        raise ValueError("slot_to_dna_index has no visible DNA residues.")
    if visible.max() >= dna_len:
        raise ValueError(
            f"slot_to_dna_index out of range for selected DNA residues: 0..{dna_len - 1}."
        )


def _alignment_arrays(meta: dict) -> dict[str, np.ndarray]:
    return {
        "alignment_mode": np.asarray(meta["alignment_mode"]),
        "alignment_score_mode": np.asarray(meta["alignment_score_mode"]),
        "alignment_chain": np.asarray(meta["alignment_chain"]),
        "alignment_start": np.asarray(meta["alignment_start"], dtype=np.int64),
        "alignment_reverse_complement": np.asarray(
            meta["alignment_reverse_complement"], dtype=bool
        ),
        "alignment_score": np.asarray(meta["alignment_score"], dtype=np.float32),
        "alignment_sequence": np.asarray(meta["alignment_sequence"]),
        "alignment_candidate_count": np.asarray(
            meta["alignment_candidate_count"], dtype=np.int64
        ),
        "alignment_contact_candidate_count": np.asarray(
            meta["alignment_contact_candidate_count"], dtype=np.int64
        ),
        "canonical_reverse_complement": np.asarray(
            meta["canonical_reverse_complement"], dtype=bool
        ),
    }


def _canonical_alignment_meta(meta: dict, reverse: bool) -> dict:
    result = {**meta, "canonical_reverse_complement": bool(reverse)}
    if reverse:
        result["alignment_sequence"] = reverse_complement_sequence(
            str(meta["alignment_sequence"])
        )
    return result


def _canonicalize_sample_slots(
    pwm: np.ndarray,
    slot_to_dna_index: np.ndarray,
    slot_arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], bool]:
    canonical_pwm, reverse = canonicalize_pwm(pwm)
    motif_len = canonical_pwm.shape[0]
    if slot_to_dna_index.shape != (motif_len,):
        raise ValueError(
            f"slot_to_dna_index shape {slot_to_dna_index.shape} does not match "
            f"motif length {motif_len}."
        )
    for key, value in slot_arrays.items():
        if value.shape[-1] != motif_len:
            raise ValueError(
                f"{key} last axis {value.shape[-1]} does not match motif length "
                f"{motif_len}."
            )
    if not reverse:
        return canonical_pwm, slot_to_dna_index, slot_arrays, False

    transformed = {}
    for key, value in slot_arrays.items():
        transformed[key] = value[..., ::-1].copy()
    return canonical_pwm, slot_to_dna_index[::-1].copy(), transformed, True
