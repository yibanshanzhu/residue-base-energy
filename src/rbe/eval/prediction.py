from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rbe.data.pwm import canonicalize_pwm

if TYPE_CHECKING:
    import torch


PREDICTION_KEYS = (
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


def load_model(checkpoint_path: str | Path, device: "torch.device") -> "torch.nn.Module":
    import torch

    from rbe.models import build_model_from_config

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model_from_config(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict_sample_npz(
    sample_path: str | Path,
    pred_path: str | Path,
    model: "torch.nn.Module",
    device: "torch.device",
) -> None:
    from rbe.data.dataset import RBEDataset, to_device

    sample = to_device(RBEDataset([sample_path])[0], device)
    outputs = run_model_on_sample(sample, model)
    arrays = prediction_arrays(sample, outputs)

    output = Path(pred_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def run_model_on_sample(sample: dict, model: "torch.nn.Module") -> dict:
    import torch

    motif_len = int(sample["pwm_target"].shape[0])
    with torch.no_grad():
        return model(
            esm2_repr=sample["esm2_repr"],
            aa_idx=sample["aa_idx"],
            residue_xyz=sample["residue_xyz"],
            edge_index=sample["edge_index"],
            edge_attr=sample["edge_attr"],
            motif_len=motif_len,
        )


def prediction_arrays(sample: dict, outputs: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = sample_metadata_arrays(sample)
    for key in PREDICTION_KEYS:
        if key in outputs:
            arrays[key] = outputs[key].detach().cpu().numpy()
    return canonicalize_prediction_arrays(arrays)


def canonicalize_prediction_arrays(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if "pwm" not in arrays:
        raise ValueError("Prediction arrays must contain pwm for canonicalization.")

    result = dict(arrays)
    result["pwm"], reverse = canonicalize_pwm(result["pwm"])
    if reverse:
        for key in ("pwm_logits",):
            if key in result:
                result[key] = result[key][::-1, ::-1].copy()
        for key in (
            "A",
            "A_base",
            "A_base_logits",
            "A_backbone",
            "A_backbone_logits",
            "A_contact",
            "A_contact_logits",
        ):
            if key in result:
                result[key] = result[key][:, ::-1].copy()
        if "E" in result:
            result["E"] = result["E"][:, ::-1, ::-1].copy()
    result["canonical_reverse_complement"] = np.asarray(reverse, dtype=bool)
    return result


def sample_metadata_arrays(sample: dict) -> dict[str, np.ndarray]:
    return {
        "residue_ids": np.asarray(sample["residue_ids"]),
        "residue_aa": np.asarray(sample["residue_aa"]),
        "residue_xyz": sample["residue_xyz"].detach().cpu().numpy(),
    }
