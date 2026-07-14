from __future__ import annotations

from pathlib import Path

import numpy as np

from rbe.eval.io import get_pwm, load_npz
from rbe.eval.metrics import (
    average_precision,
    binary_metrics,
    pwm_metrics,
    top_l_precision,
)


def evaluate_pair(target_path: str | Path, pred_path: str | Path) -> dict:
    target = load_npz(target_path)
    pred = load_npz(pred_path)
    row = {
        "sample": Path(target_path).stem,
        "target_path": str(target_path),
        "pred_path": str(pred_path),
    }
    if "site_label" in target:
        row["n_residue"] = float(target["site_label"].size)
        row["site_pos"] = float(target["site_label"].sum())
    for label_key in ("A_base_label", "A_backbone_label", "A_contact_label"):
        if label_key in target:
            row[f"{label_key[:-6]}_pos"] = float(target[label_key].sum())

    if "pwm_mask" not in target:
        raise ValueError(f"{target_path}: missing pwm_mask required for masked PWM MAE.")
    for key, value in pwm_metrics(
        get_pwm(target),
        get_pwm(pred),
        mae_mask=target["pwm_mask"],
    ).items():
        row[f"pwm_{key}"] = value

    for prefix, label_key, pred_key, mask_key in _map_specs(target, pred):
        metrics = map_metrics(
            target[label_key],
            pred[pred_key],
            target[mask_key] if mask_key and mask_key in target else None,
        )
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value

    if "site_label" in target and "site_prob" in pred:
        site_metrics = binary_metrics(target["site_label"], pred["site_prob"])
        for key, value in site_metrics.items():
            row[f"site_{key}"] = value

    return row


def map_metrics(
    y_true: np.ndarray, y_score: np.ndarray, y_mask: np.ndarray | None = None
) -> dict:
    if y_mask is not None:
        mask = np.asarray(y_mask).astype(bool)
        if mask.ndim == 1 and np.asarray(y_true).ndim == 2:
            mask = np.broadcast_to(mask[None, :], np.asarray(y_true).shape)
        y_true = np.asarray(y_true)[mask]
        y_score = np.asarray(y_score)[mask]
    top_l = int(y_true.sum())
    if top_l <= 0:
        return {
            "ap": 0.0,
            "top_l_precision": 0.0,
            "top_l": 0.0,
        }
    return {
        "ap": average_precision(y_true, y_score),
        "top_l_precision": top_l_precision(y_true, y_score, top_l=top_l),
        "top_l": float(top_l),
    }


def _map_specs(target: dict, pred: dict) -> list[tuple[str, str, str, str | None]]:
    specs = [
        ("A_base", "A_base_label", "A_base", "A_base_mask"),
        ("A_backbone", "A_backbone_label", "A_backbone", "pwm_mask"),
        ("A_contact", "A_contact_label", "A_contact", "pwm_mask"),
    ]
    if "A_base_label" not in target and "A_label" in target and "A" in pred:
        specs[0] = ("A_base", "A_label", "A", None)
    return [
        (prefix, label_key, pred_key, mask_key)
        for prefix, label_key, pred_key, mask_key in specs
        if label_key in target and pred_key in pred
    ]
