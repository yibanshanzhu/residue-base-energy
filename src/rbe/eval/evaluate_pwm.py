from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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


def evaluate(args: argparse.Namespace) -> None:
    target = _load_npz(args.target)
    pred = _load_npz(args.pred)
    target_pwm = _get_pwm(target)

    rows = []
    for method, path in [("ours", args.pred)] + _parse_baseline(args.baseline):
        metrics = pwm_metrics(target_pwm, _get_pwm(_load_npz(path)))
        rows.append({"method": method, **metrics})

    print("PWM")
    print("method\tmae\tkl\tic_pcc\trc_aware_kl")
    for row in rows:
        print(
            f"{row['method']}\t{row['mae']:.6f}\t{row['kl']:.6f}\t"
            f"{row['ic_pcc']:.6f}\t{row['rc_aware_kl']:.6f}"
        )

    extra = {"pwm": rows}
    if "A_label" in target and "A" in pred:
        y_true = target["A_label"]
        y_score = pred["A"]
        top_l = int(y_true.sum()) if int(y_true.sum()) > 0 else y_true.shape[1]
        amap = {
            "ap": average_precision(y_true, y_score),
            "top_l_precision": top_l_precision(y_true, y_score, top_l=top_l),
            "top_l": top_l,
        }
        extra["A_map"] = amap
        print("\nA_map")
        print("ap\ttop_l_precision\ttop_l")
        print(f"{amap['ap']:.6f}\t{amap['top_l_precision']:.6f}\t{amap['top_l']}")

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

