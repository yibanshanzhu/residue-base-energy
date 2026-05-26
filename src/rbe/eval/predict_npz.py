from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from rbe.data.dataset import RBEDataset, to_device
from rbe.models import build_model_from_config
from rbe.utils import resolve_device


def predict_npz(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    sample = RBEDataset([args.sample])[0]
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_config(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    sample = to_device(sample, device)
    motif_len = int(sample["pwm_target"].shape[0])
    with torch.no_grad():
        outputs = model(
            esm2_repr=sample["esm2_repr"],
            aa_idx=sample["aa_idx"],
            residue_xyz=sample["residue_xyz"],
            edge_index=sample["edge_index"],
            edge_attr=sample["edge_attr"],
            motif_len=motif_len,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        residue_ids=np.asarray(sample["residue_ids"]),
        residue_aa=np.asarray(sample["residue_aa"]),
        residue_xyz=sample["residue_xyz"].detach().cpu().numpy(),
        pwm=outputs["pwm"].detach().cpu().numpy(),
        pwm_logits=outputs["pwm_logits"].detach().cpu().numpy(),
        A=outputs["A"].detach().cpu().numpy(),
        A_base=outputs["A_base"].detach().cpu().numpy(),
        A_base_logits=outputs["A_base_logits"].detach().cpu().numpy(),
        A_backbone=outputs["A_backbone"].detach().cpu().numpy(),
        A_backbone_logits=outputs["A_backbone_logits"].detach().cpu().numpy(),
        A_contact=outputs["A_contact"].detach().cpu().numpy(),
        A_contact_logits=outputs["A_contact_logits"].detach().cpu().numpy(),
        E=outputs["E"].detach().cpu().numpy(),
        site_prob=outputs["site_prob"].detach().cpu().numpy(),
        site_score=outputs["site_score"].detach().cpu().numpy(),
    )
    print(f"wrote {output} from checkpoint={args.checkpoint} sample={args.sample}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a trained checkpoint on one processed RBE npz sample."
    )
    parser.add_argument("--sample", required=True, help="Processed target .npz sample.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    predict_npz(build_argparser().parse_args())


if __name__ == "__main__":
    main()
