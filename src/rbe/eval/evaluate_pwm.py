from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rbe.data.pwm import canonicalize_pwm
from rbe.eval.metrics import average_precision, binary_metrics, pwm_metrics, top_l_precision


def _load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _get_pwm(data: dict) -> np.ndarray:
    for key in ("pwm", "PWM", "pwm_target", "P"):
        if key in data:
            pwm = data[key]
            if pwm.ndim == 2 and pwm.shape[1] == 4:
                return pwm.astype(np.float32)
    raise ValueError("No [M,4] PWM found. Tried keys: pwm, PWM, pwm_target, P.")


def _parse_baseline(values: list[str] | None) -> list[tuple[str, str]]:
    out = []
    for value in values or []:
        if "=" not in value:
            raise ValueError("--baseline must be NAME=path.npz")
        name, path = value.split("=", 1)
        out.append((name, path))
    return out


def _print_map_metrics(
    title: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_mask: np.ndarray | None = None,
) -> dict:
    if y_mask is not None:
        mask = np.asarray(y_mask).astype(bool)
        if mask.ndim == 1 and np.asarray(y_true).ndim == 2:
            mask = np.broadcast_to(mask[None, :], np.asarray(y_true).shape)
        y_true = np.asarray(y_true)[mask]
        y_score = np.asarray(y_score)[mask]
    top_l = int(y_true.sum())
    if top_l <= 0:
        metrics = {
            "ap": 0.0,
            "top_l_precision": 0.0,
            "top_l": 0,
        }
        print(f"\n{title}")
        print("ap\ttop_l_precision\ttop_l")
        print("0.000000\t0.000000\t0")
        return metrics
    metrics = {
        "ap": average_precision(y_true, y_score),
        "top_l_precision": top_l_precision(y_true, y_score, top_l=top_l),
        "top_l": top_l,
    }
    print(f"\n{title}")
    print("ap\ttop_l_precision\ttop_l")
    print(
        f"{metrics['ap']:.6f}\t{metrics['top_l_precision']:.6f}\t{metrics['top_l']}"
    )
    return metrics


def evaluate(args: argparse.Namespace) -> None:
    target = _load_npz(args.target)
    pred = _load_npz(args.pred)
    target_pwm = _get_pwm(target)
    _require_canonical(target, args.target)
    _require_canonical(pred, args.pred)

    rows = []
    for method, path in [("ours", args.pred)] + _parse_baseline(args.baseline):
        prediction = _load_npz(path)
        _require_canonical(prediction, path)
        metrics = pwm_metrics(target_pwm, _get_pwm(prediction))
        rows.append({"method": method, **metrics})

    print("PWM")
    print("method\tmae\tkl\tic_pcc")
    for row in rows:
        print(
            f"{row['method']}\t{row['mae']:.6f}\t{row['kl']:.6f}\t"
            f"{row['ic_pcc']:.6f}"
        )

    extra = {"pwm": rows}
    if "A_base_label" in target and "A_base" in pred:
        extra["A_base_map"] = _print_map_metrics(
            "A_base_map",
            target["A_base_label"],
            pred["A_base"],
            target["A_base_mask"] if "A_base_mask" in target else None,
        )
    elif "A_label" in target and "A" in pred:
        extra["A_base_map"] = _print_map_metrics("A_base_map", target["A_label"], pred["A"])

    if "A_backbone_label" in target and "A_backbone" in pred:
        extra["A_backbone_map"] = _print_map_metrics(
            "A_backbone_map",
            target["A_backbone_label"],
            pred["A_backbone"],
            target["pwm_mask"] if "pwm_mask" in target else None,
        )

    if "A_contact_label" in target and "A_contact" in pred:
        extra["A_contact_map"] = _print_map_metrics(
            "A_contact_map",
            target["A_contact_label"],
            pred["A_contact"],
            target["pwm_mask"] if "pwm_mask" in target else None,
        )

    if "site_label" in target and "site_prob" in pred:
        site = binary_metrics(target["site_label"], pred["site_prob"])
        extra["protein_site"] = site
        print("\nprotein_site")
        print("ap\tmcc\tf1")
        print(f"{site['ap']:.6f}\t{site['mcc']:.6f}\t{site['f1']:.6f}")

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(extra, indent=2), encoding="utf-8")


def _require_canonical(data: dict, path: str | Path) -> None:
    if "canonical_reverse_complement" not in data:
        raise ValueError(f"{path}: missing canonical PWM orientation metadata.")
    if canonicalize_pwm(_get_pwm(data))[1]:
        raise ValueError(f"{path}: PWM is not in canonical orientation.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PWM, A map, and protein-site output.")
    parser.add_argument("--target", required=True, help="Target npz with pwm_target.")
    parser.add_argument("--pred", required=True, help="Prediction npz from rbe.predict.")
    parser.add_argument(
        "--baseline",
        action="append",
        default=None,
        help="Optional baseline prediction as NAME=path.npz.",
    )
    parser.add_argument("--json-output", default=None)
    return parser


def main() -> None:
    evaluate(build_argparser().parse_args())


if __name__ == "__main__":
    main()
