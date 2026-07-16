from __future__ import annotations

from pathlib import Path

import numpy as np

from rbe.data.pwm import canonicalize_pwm
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

    target_orientation = _require_orientation(target, target_path)
    pred_orientation = _require_orientation(pred, pred_path)
    if target_orientation != pred_orientation:
        raise ValueError(
            f"PWM orientation mismatch: target={target_orientation!r}, "
            f"prediction={pred_orientation!r}."
        )
    for key, value in pwm_metrics(get_pwm(target), get_pwm(pred)).items():
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


def _require_orientation(data: dict, path: str | Path) -> str:
    if "pwm_orientation" not in data:
        raise ValueError(f"{path}: missing pwm_orientation metadata.")
    orientation = str(data["pwm_orientation"])
    if orientation != "canonical" and not orientation.startswith("family_reference:"):
        raise ValueError(f"{path}: unsupported PWM orientation {orientation!r}.")
    if orientation == "canonical" and canonicalize_pwm(get_pwm(data))[1]:
        raise ValueError(f"{path}: PWM is not in canonical orientation.")
    return orientation


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
