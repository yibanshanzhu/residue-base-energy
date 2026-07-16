from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from rbe.data.esm import extract_esm2_t33_hidden
from rbe.data.features import build_residue_graph
from rbe.data.pdb import (
    ca_or_centroid,
    parse_chain_list,
    parse_pdb,
    residue_one_letter,
    residue_sequence,
    select_dna_residues,
    select_protein_residues,
)
from rbe.eval.prediction import orient_prediction_arrays
from rbe.models import build_model_from_config
from rbe.utils import resolve_device


def predict(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    residues = parse_pdb(args.pdb)
    if select_dna_residues(residues, None):
        raise ValueError("predict.py accepts monomer protein PDB only; DNA residues were found.")

    protein_chains = parse_chain_list(args.protein_chains)
    protein = select_protein_residues(residues, protein_chains)
    if not protein:
        raise ValueError("No protein residues found. Check --protein-chains.")

    residue_ids = np.asarray([residue.residue_id for residue in protein])
    residue_aa = np.asarray([residue_one_letter(residue) for residue in protein])
    residue_xyz = np.stack([ca_or_centroid(residue) for residue in protein]).astype(
        np.float32
    )
    edge_index, edge_attr = build_residue_graph(residue_xyz)
    if args.esm_npy:
        esm2_repr = np.load(args.esm_npy).astype(np.float32)
    else:
        esm2_repr = extract_esm2_t33_hidden(residue_sequence(protein), device=str(device))
    if esm2_repr.shape != (len(protein), 1280):
        raise ValueError(
            f"esm2_repr must have shape {(len(protein), 1280)}, got {esm2_repr.shape}."
        )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_config(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    aa_idx = np.array(
        ["ACDEFGHIKLMNPQRSTVWY".index(aa) for aa in residue_aa], dtype=np.int64
    )
    with torch.no_grad():
        outputs = model(
            esm2_repr=torch.from_numpy(esm2_repr).to(device),
            aa_idx=torch.from_numpy(aa_idx).to(device),
            residue_xyz=torch.from_numpy(residue_xyz).to(device),
            edge_index=torch.from_numpy(edge_index).to(device),
            edge_attr=torch.from_numpy(edge_attr).to(device),
            motif_len=int(args.motif_length),
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "residue_ids": residue_ids,
        "residue_aa": residue_aa,
        "residue_xyz": residue_xyz,
        "residue_edges": edge_index,
        "edge_attr": edge_attr,
        **{
            key: value.cpu().numpy()
            for key, value in outputs.items()
            if key in (
                "pwm",
                "pwm_logits",
                "A",
                "A_base",
                "A_base_logits",
                "A_backbone",
                "A_backbone_logits",
                "A_contact",
                "A_contact_logits",
                "E",
                "site_prob",
                "site_score",
            )
        },
    }
    np.savez_compressed(
        output, **orient_prediction_arrays(arrays, args.pwm_orientation)
    )
    print(
        f"wrote {output} N={len(protein)} M={int(args.motif_length)} "
        f"E_edges={edge_index.shape[1]}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict PWM/contact map from a protein monomer PDB and motif length."
    )
    parser.add_argument("--pdb", required=True, help="Protein monomer PDB.")
    parser.add_argument("--motif-length", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protein-chains", default=None)
    parser.add_argument("--esm-npy", default=None, help="Precomputed [N,1280] ESM2 hidden .npy.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--pwm-orientation",
        required=True,
        help="canonical or an explicit family_reference:FAMILY:VERSION contract.",
    )
    return parser


def main() -> None:
    predict(build_argparser().parse_args())


if __name__ == "__main__":
    main()
