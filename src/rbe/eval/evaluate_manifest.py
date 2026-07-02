from __future__ import annotations

import argparse
from pathlib import Path

from rbe.eval.io import get_pwm, load_npz, pred_path_for_sample, read_manifest
from rbe.eval.pair_metrics import evaluate_pair, map_metrics
from rbe.eval.prediction import load_model, predict_sample_npz
from rbe.eval.reports import write_rows_tsv, write_summary_json, write_summary_tsv
from rbe.eval.summary import (
    global_site_summary_rows,
    numeric_keys,
    print_summary,
    summarize_rows,
)
from rbe.utils import resolve_device


def evaluate_manifest(args: argparse.Namespace) -> None:
    samples = read_manifest(args.manifest, limit=args.limit)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = None
    device = None
    if args.checkpoint:
        device = resolve_device(args.device)
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
        write_summary_json(args.summary_json, rows, summary)

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
