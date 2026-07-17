from __future__ import annotations

import argparse

from rbe.eval.cluster_evaluation import evaluate_cluster_methods


def _parse_method(value: str) -> tuple[str, str]:
    name, separator, template = value.partition("=")
    if not separator or not name.strip() or not template.strip():
        raise argparse.ArgumentTypeError("--method must use NAME=PREDICTION_TEMPLATE.")
    return name.strip(), template.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions with held-out component-equal aggregation."
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--method", action="append", required=True, type=_parse_method)
    parser.add_argument("--reference-method", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    methods = dict(args.method)
    if len(methods) != len(args.method):
        parser.error("--method names must be unique.")
    result = evaluate_cluster_methods(
        args.benchmark_root,
        methods,
        args.out_root,
        reference_method=args.reference_method,
    )
    print(f"wrote {result.per_sample_tsv}")
    print(f"wrote {result.per_component_tsv}")
    print(f"wrote {result.summary_tsv}")
    print(f"wrote {result.paired_metrics_tsv}")


if __name__ == "__main__":
    main()
