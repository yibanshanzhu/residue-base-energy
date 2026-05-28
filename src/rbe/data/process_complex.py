from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rbe.data.alignment import align_pwm_to_dna, enumerate_pwm_to_dna_alignments
from rbe.data.esm import extract_esm2_t33_hidden
from rbe.data.features import build_residue_graph, min_pairwise_distance
from rbe.data.pdb import (
    MissingDnaAtomsError,
    base_heavy_atom_coords,
    backbone_heavy_atom_coords,
    ca_or_centroid,
    heavy_atom_coords,
    parse_chain_list,
    parse_pdb,
    residue_one_letter,
    residue_sequence,
    select_dna_residues,
    select_protein_residues,
)
from rbe.data.pwm import read_pwm


def _parse_slot_to_dna_index(
    value: str | None,
    motif_len: int,
    start: int | None,
    pwm_target: np.ndarray,
    protein: list,
    dna: list,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    if value is None or value.strip() == "":
        if start is not None:
            return np.arange(start, start + motif_len, dtype=np.int64), {
                "alignment_mode": "manual_dna_start_index",
                "alignment_score_mode": "manual",
                "alignment_chain": "",
                "alignment_start": start,
                "alignment_reverse_complement": False,
                "alignment_score": np.nan,
                "alignment_sequence": "",
                "alignment_candidate_count": 0,
                "alignment_contact_candidate_count": 0,
            }
        if args.alignment_contact_policy == "sequence_only":
            alignment = align_pwm_to_dna(
                pwm_target, dna, score_mode=args.alignment_score
            )
            contact_candidate_count = 0
            candidate_count = 0
            mode = "auto_pwm_dna"
        else:
            alignment, candidate_count, contact_candidate_count = (
                _select_contact_constrained_alignment(
                    pwm_target=pwm_target,
                    protein=protein,
                    dna=dna,
                    args=args,
                )
            )
            mode = "contact_constrained_pwm_dna"
        return alignment.slot_to_dna_index, {
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
    indices = np.array([int(item.strip()) for item in value.split(",")], dtype=np.int64)
    if indices.shape[0] != motif_len:
        raise ValueError(
            f"--slot-to-dna-index length {indices.shape[0]} != PWM length {motif_len}."
        )
    return indices, {
        "alignment_mode": "manual_slot_to_dna_index",
        "alignment_score_mode": "manual",
        "alignment_chain": "",
        "alignment_start": -1,
        "alignment_reverse_complement": False,
        "alignment_score": np.nan,
        "alignment_sequence": "",
        "alignment_candidate_count": 0,
        "alignment_contact_candidate_count": 0,
    }


def _compute_contact_labels(
    protein: list,
    dna: list,
    slot_to_dna_index: np.ndarray,
    base_contact_cutoff: float,
    backbone_contact_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A_base_label = np.zeros((len(protein), len(slot_to_dna_index)), dtype=np.float32)
    A_backbone_label = np.zeros_like(A_base_label)

    slot_base_coords = [base_heavy_atom_coords(dna[idx]) for idx in slot_to_dna_index]
    slot_backbone_coords = [
        backbone_heavy_atom_coords(dna[idx]) for idx in slot_to_dna_index
    ]
    for i, residue in enumerate(protein):
        protein_heavy = heavy_atom_coords(residue)
        for j, base_coords in enumerate(slot_base_coords):
            if min_pairwise_distance(protein_heavy, base_coords) <= base_contact_cutoff:
                A_base_label[i, j] = 1.0
            if (
                min_pairwise_distance(protein_heavy, slot_backbone_coords[j])
                <= backbone_contact_cutoff
            ):
                A_backbone_label[i, j] = 1.0

    A_contact_label = np.maximum(A_base_label, A_backbone_label).astype(np.float32)
    site_label = A_contact_label.max(axis=1).astype(np.float32)
    return A_base_label, A_backbone_label, A_contact_label, site_label


def _contact_counts(
    A_base_label: np.ndarray,
    A_backbone_label: np.ndarray,
    A_contact_label: np.ndarray,
    site_label: np.ndarray,
) -> dict[str, int]:
    return {
        "A_base_pos": int(A_base_label.sum()),
        "A_backbone_pos": int(A_backbone_label.sum()),
        "A_contact_pos": int(A_contact_label.sum()),
        "site_pos": int(site_label.sum()),
    }


def _passes_alignment_contact_constraints(
    counts: dict[str, int], args: argparse.Namespace
) -> bool:
    return (
        counts["A_contact_pos"] >= args.alignment_min_contact_pairs
        and counts["site_pos"] >= args.alignment_min_site_residues
        and counts["A_base_pos"] >= args.alignment_min_base_pairs
    )


def _select_contact_constrained_alignment(
    pwm_target: np.ndarray,
    protein: list,
    dna: list,
    args: argparse.Namespace,
):
    candidates = enumerate_pwm_to_dna_alignments(
        pwm_target, dna, score_mode=args.alignment_score
    )
    if not candidates:
        motif_len = pwm_target.shape[0]
        raise ValueError(f"No selected DNA chain is long enough for PWM length {motif_len}.")

    valid = []
    invalid_geometry_count = 0
    for candidate in candidates:
        try:
            labels = _compute_contact_labels(
                protein=protein,
                dna=dna,
                slot_to_dna_index=candidate.slot_to_dna_index,
                base_contact_cutoff=args.base_contact_cutoff,
                backbone_contact_cutoff=args.backbone_contact_cutoff,
            )
        except MissingDnaAtomsError:
            invalid_geometry_count += 1
            continue
        counts = _contact_counts(*labels)
        if _passes_alignment_contact_constraints(counts, args):
            valid.append((candidate, counts))

    if not valid:
        best_sequence = max(candidates, key=lambda candidate: candidate.score)
        raise ValueError(
            "No PWM-DNA alignment candidate passed contact constraints. "
            f"candidate_count={len(candidates)} "
            f"min_contact_pairs={args.alignment_min_contact_pairs} "
            f"min_site_residues={args.alignment_min_site_residues} "
            f"min_base_pairs={args.alignment_min_base_pairs} "
            f"invalid_geometry_candidates={invalid_geometry_count} "
            f"best_sequence_only=chain:{best_sequence.chain} "
            f"start:{best_sequence.start} rc:{best_sequence.reverse_complement} "
            f"score:{best_sequence.score:.6f}"
        )

    best, _ = max(valid, key=lambda item: item[0].score)
    return best, len(candidates), len(valid)


def process_complex(args: argparse.Namespace) -> None:
    residues = parse_pdb(args.pdb)
    protein_chains = parse_chain_list(args.protein_chains)
    dna_chains = parse_chain_list(args.dna_chains)
    protein = select_protein_residues(residues, protein_chains)
    dna = select_dna_residues(residues, dna_chains)
    if not protein:
        raise ValueError("No protein residues found. Check --protein-chains.")
    if not dna:
        raise ValueError("No DNA residues found. Check --dna-chains.")

    pwm_target = read_pwm(args.pwm)
    motif_len = pwm_target.shape[0]
    slot_to_dna_index, alignment_meta = _parse_slot_to_dna_index(
        args.slot_to_dna_index,
        motif_len,
        args.dna_start_index,
        pwm_target,
        protein,
        dna,
        args,
    )
    if slot_to_dna_index.min() < 0 or slot_to_dna_index.max() >= len(dna):
        raise ValueError(
            f"slot_to_dna_index out of range for selected DNA residues: 0..{len(dna) - 1}."
        )

    residue_ids = np.asarray([residue.residue_id for residue in protein])
    residue_aa = np.asarray([residue_one_letter(residue) for residue in protein])
    residue_xyz = np.stack([ca_or_centroid(residue) for residue in protein]).astype(
        np.float32
    )
    residue_edges, edge_attr = build_residue_graph(
        residue_xyz,
        cutoff=args.ca_cutoff,
        num_rbf=args.num_rbf,
        max_distance=args.rbf_max_distance,
    )

    sequence = residue_sequence(protein)
    if args.esm_npy:
        esm2_repr = np.load(args.esm_npy).astype(np.float32)
    else:
        esm2_repr = extract_esm2_t33_hidden(sequence, device=args.device)
    if esm2_repr.shape != (len(protein), 1280):
        raise ValueError(
            f"esm2_repr must have shape {(len(protein), 1280)}, got {esm2_repr.shape}."
        )

    A_base_label, A_backbone_label, A_contact_label, site_label = (
        _compute_contact_labels(
            protein=protein,
            dna=dna,
            slot_to_dna_index=slot_to_dna_index,
            base_contact_cutoff=args.base_contact_cutoff,
            backbone_contact_cutoff=args.backbone_contact_cutoff,
        )
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        residue_ids=residue_ids,
        residue_aa=residue_aa,
        residue_xyz=residue_xyz,
        residue_edges=residue_edges,
        edge_attr=edge_attr,
        esm2_repr=esm2_repr,
        pwm_target=pwm_target,
        A_label=A_base_label,
        A_base_label=A_base_label,
        A_backbone_label=A_backbone_label,
        A_contact_label=A_contact_label,
        site_label=site_label,
        slot_to_dna_index=slot_to_dna_index,
        alignment_mode=np.asarray(alignment_meta["alignment_mode"]),
        alignment_score_mode=np.asarray(alignment_meta["alignment_score_mode"]),
        alignment_chain=np.asarray(alignment_meta["alignment_chain"]),
        alignment_start=np.asarray(alignment_meta["alignment_start"], dtype=np.int64),
        alignment_reverse_complement=np.asarray(
            alignment_meta["alignment_reverse_complement"], dtype=bool
        ),
        alignment_score=np.asarray(alignment_meta["alignment_score"], dtype=np.float32),
        alignment_sequence=np.asarray(alignment_meta["alignment_sequence"]),
        alignment_candidate_count=np.asarray(
            alignment_meta["alignment_candidate_count"], dtype=np.int64
        ),
        alignment_contact_candidate_count=np.asarray(
            alignment_meta["alignment_contact_candidate_count"], dtype=np.int64
        ),
    )
    print(
        f"wrote {output} N={len(protein)} M={motif_len} E={residue_edges.shape[1]} "
        f"A_base_pos={int(A_base_label.sum())} "
        f"A_backbone_pos={int(A_backbone_label.sum())} "
        f"A_contact_pos={int(A_contact_label.sum())} "
        f"site_pos={int(site_label.sum())} "
        f"alignment={alignment_meta['alignment_mode']} chain={alignment_meta['alignment_chain']} "
        f"start={alignment_meta['alignment_start']} rc={alignment_meta['alignment_reverse_complement']} "
        f"score_mode={alignment_meta['alignment_score_mode']} "
        f"contact_candidates={alignment_meta['alignment_contact_candidate_count']}/"
        f"{alignment_meta['alignment_candidate_count']}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one RBE training npz from a protein-DNA complex and PWM."
    )
    parser.add_argument("--pdb", required=True, help="Protein-DNA complex PDB.")
    parser.add_argument("--pwm", required=True, help="PWM file with A/C/G/T columns.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--protein-chains", default=None, help="Comma-separated protein chains.")
    parser.add_argument("--dna-chains", default=None, help="Comma-separated DNA chains.")
    parser.add_argument(
        "--dna-start-index",
        type=int,
        default=None,
        help="Manual contiguous DNA start index. If omitted, PWM-DNA alignment is automatic.",
    )
    parser.add_argument(
        "--slot-to-dna-index",
        default=None,
        help="Comma-separated selected DNA residue indices, one per PWM row.",
    )
    parser.add_argument(
        "--alignment-score",
        choices=["ic_log_likelihood", "log_likelihood", "deeppbs_ic_pcc"],
        default="ic_log_likelihood",
        help="Score used for automatic PWM-DNA alignment.",
    )
    parser.add_argument(
        "--alignment-contact-policy",
        choices=["require_contact", "sequence_only"],
        default="require_contact",
        help="For automatic alignment, require motif window contact before scoring.",
    )
    parser.add_argument("--alignment-min-base-pairs", type=int, default=0)
    parser.add_argument("--alignment-min-contact-pairs", type=int, default=1)
    parser.add_argument("--alignment-min-site-residues", type=int, default=1)
    parser.add_argument("--esm-npy", default=None, help="Precomputed [N,1280] ESM2 hidden .npy.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ca-cutoff", type=float, default=14.0)
    parser.add_argument(
        "--base-contact-cutoff",
        "--contact-cutoff",
        dest="base_contact_cutoff",
        type=float,
        default=4.5,
        help="Residue heavy atom to DNA base heavy atom cutoff for A_base_label.",
    )
    parser.add_argument(
        "--backbone-contact-cutoff",
        type=float,
        default=5.0,
        help="Residue heavy atom to DNA sugar/phosphate heavy atom cutoff for A_backbone_label.",
    )
    parser.add_argument(
        "--site-cutoff",
        type=float,
        default=5.0,
        help="Deprecated; site_label is derived from A_contact_label.",
    )
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--rbf-max-distance", type=float, default=20.0)
    return parser


def main() -> None:
    process_complex(build_argparser().parse_args())


if __name__ == "__main__":
    main()
