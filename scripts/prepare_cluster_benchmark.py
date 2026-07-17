from __future__ import annotations

import argparse

from rbe.data.cluster_benchmark import prepare_cluster_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build sequence-cluster and PWM-target component-disjoint folds from "
            "canonical DeepPBS samples."
        )
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--mmseqs", required=True)
    parser.add_argument("--min-seq-id", type=float, default=0.3)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    result = prepare_cluster_benchmark(
        args.cache_root,
        args.source_root,
        args.out_root,
        mmseqs=args.mmseqs,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        n_folds=args.folds,
        threads=args.threads,
    )
    print(
        f"samples={result.sample_count} exact_sequences={result.sequence_count} "
        f"sequence_clusters={result.sequence_cluster_count} "
        f"components={result.component_count}"
    )


if __name__ == "__main__":
    main()
