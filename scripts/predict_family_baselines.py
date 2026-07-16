from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rbe.eval.family_baselines import predict_family_baselines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate group-balanced mean and nearest-ESM family baselines."
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    benchmark = Path(args.benchmark_root)
    with (benchmark / "fold_table.tsv").open(newline="") as handle:
        folds = list(csv.DictReader(handle, delimiter="\t"))
    for row in folds:
        fold = int(row["fold"])
        output = Path(args.out_root) / f"fold{fold}"
        predict_family_baselines(
            benchmark / "folds" / f"fold{fold}_train.txt",
            benchmark / "folds" / f"fold{fold}_test.txt",
            output,
        )
        print(f"wrote fold={fold} {output}")


if __name__ == "__main__":
    main()
