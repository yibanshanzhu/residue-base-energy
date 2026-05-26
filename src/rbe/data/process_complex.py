from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rbe.data.alignment import align_pwm_to_dna
from rbe.data.esm import extract_esm2_t33_hidden
from rbe.data.features import build_residue_graph, min_pairwise_distance
from rbe.data.pdb import (
    base_heavy_atom_coords,
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
    dna: list,
    alignment_score_mode: str,
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
            }
        alignment = align_pwm_to_dna(
            pwm_target, dna, score_mode=alignment_score_mode
        )
        return alignment.slot_to_dna_index, {
            "alignment_mode": "auto_pwm_dna",
            "alignment_score_mode": alignment.score_mode,
            "alignment_chain": alignment.chain,
            "alignment_start": alignment.start,
            "alignment_reverse_complement": alignment.reverse_complement,
            "alignment_score": alignment.score,
            "alignment_sequence": alignment.aligned_sequence,
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
    }


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
        dna,
        args.alignment_score,
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

    dna_heavy_all = np.concatenate([heavy_atom_coords(residue) for residue in dna], axis=0)
    A_label = np.zeros((len(protein), motif_len), dtype=np.float32)
    site_label = np.zeros((len(protein),), dtype=np.float32)

    slot_base_coords = [base_heavy_atom_coords(dna[idx]) for idx in slot_to_dna_index]
    for i, residue in enumerate(protein):
        protein_heavy = heavy_atom_coords(residue)
        if min_pairwise_distance(protein_heavy, dna_heavy_all) <= args.site_cutoff:
            site_label[i] = 1.0
        for j, base_coords in enumerate(slot_base_coords):
            if min_pairwise_distance(protein_heavy, base_coords) <= args.contact_cutoff:
                A_label[i, j] = 1.0

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
        A_label=A_label,
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
    )
    print(
        f"wrote {output} N={len(protein)} M={motif_len} E={residue_edges.shape[1]} "
        f"A_pos={int(A_label.sum())} site_pos={int(site_label.sum())} "
        f"alignment={alignment_meta['alignment_mode']} chain={alignment_meta['alignment_chain']} "
        f"start={alignment_meta['alignment_start']} rc={alignment_meta['alignment_reverse_complement']} "
        f"score_mode={alignment_meta['alignment_score_mode']}"
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
        choices=["ic_log_likelihood", "log_likelihood"],
        default="ic_log_likelihood",
        help="Score used for automatic PWM-DNA alignment.",
    )
    parser.add_argument("--esm-npy", default=None, help="Precomputed [N,1280] ESM2 hidden .npy.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ca-cutoff", type=float, default=14.0)
    parser.add_argument("--contact-cutoff", type=float, default=4.5)
    parser.add_argument("--site-cutoff", type=float, default=5.0)
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--rbf-max-distance", type=float, default=20.0)
    return parser


def main() -> None:
    process_complex(build_argparser().parse_args())


if __name__ == "__main__":
    main()
