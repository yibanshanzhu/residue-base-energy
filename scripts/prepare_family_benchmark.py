from __future__ import annotations

import argparse

from rbe.data.family_benchmark import prepare_family_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed-core, protein-group-disjoint family benchmark folds."
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--family-name", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    result = prepare_family_benchmark(
        args.cache_root,
        args.spec,
        args.out_root,
        family_name=args.family_name,
        version=args.version,
    )
    print(
        f"wrote {args.out_root} samples={result.included_samples} "
        f"protein_groups={result.protein_groups}"
    )


if __name__ == "__main__":
    main()
