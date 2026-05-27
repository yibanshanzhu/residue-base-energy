from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from rbe.data.dataset import RBEDataset, to_device
from rbe.eval.metrics import (
    average_precision,
    best_threshold_metrics,
    binary_metrics,
    pwm_metrics,
    top_l_precision,
)
from rbe.models import build_model_from_config
from rbe.utils import resolve_device


def load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def get_pwm(data: dict) -> np.ndarray:
    for key in ("pwm", "PWM", "pwm_target", "P"):
        if key in data:
            pwm = data[key]
            if pwm.ndim == 2 and pwm.shape[1] == 4:
                return pwm.astype(np.float32)
    raise ValueError("No [M,4] PWM found. Tried keys: pwm, PWM, pwm_target, P.")


def read_manifest(path: str | Path, limit: int = 0) -> list[Path]:
    manifest = Path(path)
    root = manifest.resolve().parent
    samples = []
    with manifest.open() as handle:
        for line in handle:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            sample = Path(item)
            samples.append(sample if sample.is_absolute() else root / sample)
            if limit and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"No samples found in manifest: {manifest}")
    return samples


def pred_path_for_sample(
    sample_path: str | Path, pred_dir: str | Path, suffix: str
) -> Path:
    return Path(pred_dir) / f"{Path(sample_path).stem}{suffix}"


def map_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
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

    for key, value in pwm_metrics(get_pwm(target), get_pwm(pred)).items():
        row[f"pwm_{key}"] = value

    map_specs = [
        ("A_base", "A_base_label", "A_base"),
        ("A_backbone", "A_backbone_label", "A_backbone"),
        ("A_contact", "A_contact_label", "A_contact"),
    ]
    if "A_base_label" not in target and "A_label" in target and "A" in pred:
        map_specs[0] = ("A_base", "A_label", "A")

    for prefix, label_key, pred_key in map_specs:
        if label_key in target and pred_key in pred:
            metrics = map_metrics(target[label_key], pred[pred_key])
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value

    if "site_label" in target and "site_prob" in pred:
        site_metrics = binary_metrics(target["site_label"], pred["site_prob"])
        for key, value in site_metrics.items():
            row[f"site_{key}"] = value

    return row


def numeric_keys(rows: Iterable[dict]) -> list[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                keys.add(key)
    preferred = [
        "pwm_mae",
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
    ordered = [key for key in preferred if key in keys]
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
    name_map = {
        "ap": "site_global_ap_diagnostic",
        "f1_at_0.5": "site_global_f1_at_0.5_diagnostic",
        "mcc_at_0.5": "site_global_mcc_at_0.5_diagnostic",
        "best_f1_diagnostic": "site_global_best_f1_diagnostic",
        "best_f1_threshold_diagnostic": "site_global_best_f1_threshold_diagnostic",
        "best_mcc_diagnostic": "site_global_best_mcc_diagnostic",
        "best_mcc_threshold_diagnostic": "site_global_best_mcc_threshold_diagnostic",
    }
    n = int(sum(label.size for label in labels))
    return [
        {"metric": name_map[key], "mean": float(value), "std": 0.0, "n": n}
        for key, value in metrics.items()
    ]


def write_rows_tsv(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = ["sample", "target_path", "pred_path"] + numeric_keys(rows)
    with output.open("w") as handle:
        handle.write("\t".join(keys) + "\n")
        for row in rows:
            values = []
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (float, np.floating)):
                    values.append(f"{float(value):.6f}")
                else:
                    values.append(str(value))
            handle.write("\t".join(values) + "\n")


def write_summary_tsv(path: str | Path, summary: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        handle.write("metric\tmean\tstd\tn\n")
        for row in summary:
            handle.write(
                f"{row['metric']}\t{row['mean']:.6f}\t{row['std']:.6f}\t{row['n']}\n"
            )


def load_model(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model_from_config(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict_sample_npz(
    sample_path: str | Path,
    pred_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    sample = to_device(RBEDataset([sample_path])[0], device)
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

    output = Path(pred_path)
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


def print_summary(summary: list[dict], n_samples: int) -> None:
    print(f"samples\t{n_samples}")
    print("metric\tmean\tstd\tn")
    for row in summary:
        print(f"{row['metric']}\t{row['mean']:.6f}\t{row['std']:.6f}\t{row['n']}")


def evaluate_manifest(args: argparse.Namespace) -> None:
    samples = read_manifest(args.manifest, limit=args.limit)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = None
    device = resolve_device(args.device)
    if args.checkpoint:
        model = load_model(args.checkpoint, device)

    rows = []
    for sample_path in samples:
        pred_path = pred_path_for_sample(sample_path, pred_dir, args.pred_suffix)
        if args.overwrite_pred or not pred_path.exists():
            if model is None:
                raise FileNotFoundError(
                    f"Missing prediction {pred_path}. Provide --checkpoint to generate it."
                )
            predict_sample_npz(sample_path, pred_path, model, device)
            print(f"wrote {pred_path}")
        rows.append(evaluate_pair(sample_path, pred_path))

    summary = summarize_rows(rows)
    summary.extend(global_site_summary_rows(samples, pred_dir, args.pred_suffix))
    per_sample_tsv = (
        Path(args.per_sample_tsv)
        if args.per_sample_tsv
        else pred_dir / "eval_per_sample.tsv"
    )
    summary_tsv = (
        Path(args.summary_tsv)
        if args.summary_tsv
        else pred_dir / "eval_summary.tsv"
    )
    write_rows_tsv(per_sample_tsv, rows)
    write_summary_tsv(summary_tsv, summary)

    if args.summary_json:
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"samples": rows, "summary": summary}, indent=2),
            encoding="utf-8",
        )

    print_summary(summary, n_samples=len(rows))
    print(f"wrote {per_sample_tsv}")
    print(f"wrote {summary_tsv}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a manifest of processed RBE samples and report mean metrics."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pred-suffix", default=".pred.npz")
    parser.add_argument("--overwrite-pred", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-sample-tsv", default=None)
    parser.add_argument("--summary-tsv", default=None)
    parser.add_argument("--summary-json", default=None)
    return parser


def main() -> None:
    evaluate_manifest(build_argparser().parse_args())


if __name__ == "__main__":
    main()
