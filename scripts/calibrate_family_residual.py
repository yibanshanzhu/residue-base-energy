from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rbe.eval.family_residual_calibration import calibrate_family_residual


def _parse_scales(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--residual-scales must be comma-separated numbers."
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select family residual scale on each fold's validation UniProt."
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--residual-scales",
        type=_parse_scales,
        default=_parse_scales("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"),
    )
    args = parser.parse_args()

    benchmark = Path(args.benchmark_root)
    prediction_root = Path(args.prediction_root)
    output_root = Path(args.out_root)
    with (benchmark / "fold_table.tsv").open(newline="") as handle:
        folds = list(csv.DictReader(handle, delimiter="\t"))

    rows = []
    for row in folds:
        fold = int(row["fold"])
        result = calibrate_family_residual(
            benchmark / "folds" / f"fold{fold}_valid.txt",
            prediction_root / f"fold{fold}" / "valid_preds",
            benchmark / "folds" / f"fold{fold}_test.txt",
            prediction_root / f"fold{fold}" / "preds",
            output_root / f"fold{fold}" / "preds",
            residual_scales=args.residual_scales,
        )
        rows.append(
            {
                "fold": fold,
                "valid_group": row["valid_group"],
                "residual_scale": result.residual_scale,
                "prior_valid_mae": result.prior_valid_mae,
                "calibrated_valid_mae": result.calibrated_valid_mae,
                **{
                    f"valid_mae_scale_{scale:.1f}": score
                    for scale, score in result.valid_mae_by_scale.items()
                },
            }
        )
        print(
            f"fold={fold} valid={row['valid_group']} "
            f"scale={result.residual_scale:.1f} "
            f"valid_mae={result.calibrated_valid_mae:.6f}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "residual_scales.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
