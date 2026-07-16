from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rbe.eval.mechanism_ablation import write_mechanism_ablations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Destroy residue-gate/energy pairing in family predictions."
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    benchmark = Path(args.benchmark_root)
    with (benchmark / "fold_table.tsv").open(newline="") as handle:
        folds = list(csv.DictReader(handle, delimiter="\t"))
    for row in folds:
        fold = int(row["fold"])
        output = Path(args.out_root) / f"fold{fold}"
        write_mechanism_ablations(
            benchmark / "folds" / f"fold{fold}_test.txt",
            Path(args.prediction_root) / f"fold{fold}" / "preds",
            output,
        )
        print(f"wrote fold={fold} {output}")


if __name__ == "__main__":
    main()
