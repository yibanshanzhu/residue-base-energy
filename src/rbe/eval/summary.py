from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from rbe.data.pwm import normalize_pwm
from rbe.eval.io import get_pwm, load_npz, pred_path_for_sample
from rbe.eval.metrics import best_threshold_metrics


def numeric_keys(rows: Iterable[dict]) -> list[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                keys.add(key)

    ordered = [key for key in _preferred_metric_order() if key in keys]
    ordered.extend(sorted(keys - set(ordered)))
    return ordered


def summarize_rows(rows: list[dict]) -> list[dict]:
    summary = []
    for key in numeric_keys(rows):
        values = np.asarray(
            [
                float(row[key])
                for row in rows
                if key in row and np.isfinite(float(row[key]))
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        summary.append(
            {
                "metric": key,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "n": int(values.size),
            }
        )
    return summary


def global_pwm_mae_summary_row(
    samples: list[Path], pred_dir: Path, suffix: str
) -> dict:
    errors = []
    for sample_path in samples:
        target = load_npz(sample_path)
        pred = load_npz(pred_path_for_sample(sample_path, pred_dir, suffix))
        target_pwm = normalize_pwm(get_pwm(target))
        pred_pwm = normalize_pwm(get_pwm(pred))
        valid = np.asarray(
            target.get("pwm_mask", np.ones(target_pwm.shape[0])), dtype=bool
        )
        errors.append(np.abs(pred_pwm[valid] - target_pwm[valid]).sum(axis=1))

    values = np.concatenate(errors).astype(np.float64)
    return {
        "metric": "pwm_mae",
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "n": int(values.size),
    }


def global_site_summary_rows(samples: list[Path], pred_dir: Path, suffix: str) -> list[dict]:
    labels = []
    scores = []
    for sample_path in samples:
        target = load_npz(sample_path)
        pred = load_npz(pred_path_for_sample(sample_path, pred_dir, suffix))
        if "site_label" in target and "site_prob" in pred:
            labels.append(target["site_label"].reshape(-1))
            scores.append(pred["site_prob"].reshape(-1))
    if not labels:
        return []

    metrics = best_threshold_metrics(np.concatenate(labels), np.concatenate(scores))
    n = int(sum(label.size for label in labels))
    return [
        {"metric": _global_metric_name(key), "mean": float(value), "std": 0.0, "n": n}
        for key, value in metrics.items()
    ]


def print_summary(summary: list[dict], n_samples: int) -> None:
    print(f"samples\t{n_samples}")
    print("metric\tmean\tstd\tn")
    for row in summary:
        print(f"{row['metric']}\t{row['mean']:.6f}\t{row['std']:.6f}\t{row['n']}")


def _global_metric_name(metric: str) -> str:
    name_map = {
        "ap": "site_global_ap_diagnostic",
        "f1_at_0.5": "site_global_f1_at_0.5_diagnostic",
        "mcc_at_0.5": "site_global_mcc_at_0.5_diagnostic",
        "best_f1_diagnostic": "site_global_best_f1_diagnostic",
        "best_f1_threshold_diagnostic": "site_global_best_f1_threshold_diagnostic",
        "best_mcc_diagnostic": "site_global_best_mcc_diagnostic",
        "best_mcc_threshold_diagnostic": "site_global_best_mcc_threshold_diagnostic",
    }
    return name_map[metric]


def _preferred_metric_order() -> list[str]:
    return [
        "pwm_mae",
        "pwm_mae_sample",
        "pwm_kl",
        "pwm_ic_pcc",
        "pwm_rc_aware_kl",
        "A_base_ap",
        "A_base_top_l_precision",
        "A_backbone_ap",
        "A_backbone_top_l_precision",
        "A_contact_ap",
        "A_contact_top_l_precision",
        "site_ap",
        "site_mcc",
        "site_f1",
        "site_global_ap_diagnostic",
        "site_global_f1_at_0.5_diagnostic",
        "site_global_mcc_at_0.5_diagnostic",
        "site_global_best_f1_diagnostic",
        "site_global_best_f1_threshold_diagnostic",
        "site_global_best_mcc_diagnostic",
        "site_global_best_mcc_threshold_diagnostic",
        "n_residue",
        "site_pos",
        "A_base_pos",
        "A_backbone_pos",
        "A_contact_pos",
    ]
