from __future__ import annotations

import argparse

from rbe.eval.family_evaluation import evaluate_family_methods


def _parse_method(value: str) -> tuple[str, str]:
    name, separator, template = value.partition("=")
    if not separator or not name.strip() or not template.strip():
        raise argparse.ArgumentTypeError(
            "--method must use NAME=PREDICTION_DIR_TEMPLATE."
        )
    return name.strip(), template.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate family predictions by averaging structures within UniProt, "
            "then UniProt groups within each method."
        )
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        type=_parse_method,
        metavar="NAME=TEMPLATE",
        help="Prediction directory; use {fold} where the fold number belongs.",
    )
    parser.add_argument("--reference-method", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--suffix", default=".pred.npz")
    args = parser.parse_args()

    methods = dict(args.method)
    if len(methods) != len(args.method):
        parser.error("--method names must be unique.")
    result = evaluate_family_methods(
        args.benchmark_root,
        methods,
        args.out_root,
        reference_method=args.reference_method,
        suffix=args.suffix,
    )
    print(f"wrote {result.per_sample_tsv}")
    print(f"wrote {result.per_group_tsv}")
    print(f"wrote {result.summary_tsv}")
    print(f"wrote {result.paired_pwm_mae_tsv}")
    print(f"wrote {result.paired_metrics_tsv}")


if __name__ == "__main__":
    main()
