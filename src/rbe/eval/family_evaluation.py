from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rbe.eval.io import load_npz, pred_path_for_sample, read_manifest
from rbe.eval.pair_metrics import evaluate_pair
from rbe.eval.summary import numeric_keys


ROW_META_KEYS = {
    "fold",
    "method",
    "n_samples",
    "pred_path",
    "protein_group",
    "sample",
    "target_path",
}


@dataclass(frozen=True)
class FamilyEvaluationResult:
    per_sample_tsv: Path
    per_group_tsv: Path
    summary_tsv: Path
    paired_pwm_mae_tsv: Path
    paired_metrics_tsv: Path


def evaluate_family_methods(
    benchmark_root: str | Path,
    method_templates: dict[str, str | Path],
    out_root: str | Path,
    *,
    reference_method: str,
    suffix: str = ".pred.npz",
) -> FamilyEvaluationResult:
    if not method_templates:
        raise ValueError("At least one family prediction method is required.")
    if reference_method not in method_templates:
        raise ValueError(
            f"Reference method {reference_method!r} is not in method_templates."
        )

    benchmark = Path(benchmark_root)
    folds = _read_folds(benchmark / "fold_table.tsv")
    sample_rows: list[dict] = []
    method_order = list(method_templates)

    for fold, expected_group in folds:
        test_manifest = benchmark / "folds" / f"fold{fold}_test.txt"
        targets = read_manifest(test_manifest)
        if not targets:
            raise ValueError(f"{test_manifest} contains no test samples.")
        target_groups = {
            _required_scalar(load_npz(path), "protein_group", path) for path in targets
        }
        if target_groups != {expected_group}:
            raise ValueError(
                f"{test_manifest}: expected group {expected_group!r}, "
                f"got {sorted(target_groups)}."
            )

        for method in method_order:
            pred_dir = _prediction_dir(method_templates[method], fold)
            for target_path in targets:
                pred_path = pred_path_for_sample(target_path, pred_dir, suffix)
                if not pred_path.exists():
                    raise FileNotFoundError(
                        f"Missing prediction for method={method!r}, fold={fold}: "
                        f"{pred_path}"
                    )
                metrics = evaluate_pair(target_path, pred_path)
                sample_rows.append(
                    {
                        "method": method,
                        "fold": fold,
                        "protein_group": expected_group,
                        **metrics,
                    }
                )

    group_rows = _aggregate_groups(sample_rows, method_order)
    _validate_method_groups(group_rows, method_order, {group for _, group in folds})
    summary_rows = _summarize_methods(group_rows, method_order)
    paired_metric_rows = _paired_metrics(
        group_rows, method_order, reference_method
    )
    paired_rows = [
        row for row in paired_metric_rows if row["metric"] == "pwm_mae"
    ]

    output = Path(out_root)
    output.mkdir(parents=True, exist_ok=True)
    result = FamilyEvaluationResult(
        per_sample_tsv=output / "per_sample.tsv",
        per_group_tsv=output / "per_group.tsv",
        summary_tsv=output / "summary.tsv",
        paired_pwm_mae_tsv=output / "paired_pwm_mae.tsv",
        paired_metrics_tsv=output / "paired_metrics.tsv",
    )
    _write_tsv(
        result.per_sample_tsv,
        sample_rows,
        [
            "method",
            "fold",
            "protein_group",
            "sample",
            "target_path",
            "pred_path",
            *_metric_keys(sample_rows),
        ],
    )
    _write_tsv(
        result.per_group_tsv,
        group_rows,
        [
            "method",
            "fold",
            "protein_group",
            "n_samples",
            *_metric_keys(group_rows),
        ],
    )
    _write_tsv(
        result.summary_tsv,
        summary_rows,
        ["method", "metric", "mean", "std", "n_groups"],
    )
    _write_tsv(
        result.paired_pwm_mae_tsv,
        paired_rows,
        [
            "reference_method",
            "method",
            "reference_mean",
            "method_mean",
            "mean_delta",
            "mean_delta_ci95_low",
            "mean_delta_ci95_high",
            "median_delta",
            "std_delta",
            "sign_flip_pvalue",
            "n_groups",
            "reference_better",
            "method_better",
            "ties",
        ],
    )
    _write_tsv(
        result.paired_metrics_tsv,
        paired_metric_rows,
        [
            "reference_method",
            "method",
            "metric",
            "higher_is_better",
            "reference_mean",
            "method_mean",
            "mean_delta",
            "mean_delta_ci95_low",
            "mean_delta_ci95_high",
            "median_delta",
            "std_delta",
            "sign_flip_pvalue",
            "n_groups",
            "reference_better",
            "method_better",
            "ties",
        ],
    )
    return result


def _read_folds(path: Path) -> list[tuple[int, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path} contains no folds.")
    folds = [(int(row["fold"]), row["test_group"].strip()) for row in rows]
    if len({fold for fold, _ in folds}) != len(folds):
        raise ValueError(f"{path} contains duplicate fold values.")
    if len({group for _, group in folds}) != len(folds):
        raise ValueError(f"{path} must hold out every protein group exactly once.")
    return folds


def _prediction_dir(template: str | Path, fold: int) -> Path:
    try:
        rendered = str(template).format(fold=fold)
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError(f"Invalid prediction template {template!s}: {error}") from error
    return Path(rendered)


def _required_scalar(data: dict[str, np.ndarray], key: str, path: Path) -> str:
    if key not in data:
        raise ValueError(f"{path}: missing {key} metadata.")
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"{path}: {key} must be scalar, got shape {value.shape}.")
    return str(value)


def _aggregate_groups(rows: list[dict], method_order: list[str]) -> list[dict]:
    result = []
    metric_keys = _metric_keys(rows)
    for method in method_order:
        method_rows = [row for row in rows if row["method"] == method]
        groups = sorted({row["protein_group"] for row in method_rows})
        for group in groups:
            members = [row for row in method_rows if row["protein_group"] == group]
            folds = {int(row["fold"]) for row in members}
            if len(folds) != 1:
                raise ValueError(
                    f"method={method!r}, group={group!r} spans folds {sorted(folds)}."
                )
            aggregate = {
                "method": method,
                "fold": next(iter(folds)),
                "protein_group": group,
                "n_samples": len(members),
            }
            for key in metric_keys:
                values = [
                    float(row[key])
                    for row in members
                    if key in row and np.isfinite(float(row[key]))
                ]
                if values:
                    aggregate[key] = float(np.mean(values))
            result.append(aggregate)
    return result


def _validate_method_groups(
    rows: list[dict], method_order: list[str], expected_groups: set[str]
) -> None:
    for method in method_order:
        groups = {row["protein_group"] for row in rows if row["method"] == method}
        if groups != expected_groups:
            raise ValueError(
                f"method={method!r} groups differ from benchmark: "
                f"expected={sorted(expected_groups)}, got={sorted(groups)}."
            )


def _summarize_methods(rows: list[dict], method_order: list[str]) -> list[dict]:
    result = []
    for method in method_order:
        method_rows = [row for row in rows if row["method"] == method]
        for key in _metric_keys(method_rows):
            values = np.asarray(
                [float(row[key]) for row in method_rows if key in row],
                dtype=np.float64,
            )
            result.append(
                {
                    "method": method,
                    "metric": key,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "n_groups": int(values.size),
                }
            )
    return result


def _paired_metrics(
    rows: list[dict], method_order: list[str], reference_method: str
) -> list[dict]:
    metric_directions = {
        "pwm_mae": False,
        "pwm_kl": False,
        "pwm_ic_pcc": True,
        "A_base_ap": True,
    }
    result = []
    for metric, higher_is_better in metric_directions.items():
        by_method = {
            method: {
                row["protein_group"]: float(row[metric])
                for row in rows
                if row["method"] == method and metric in row
            }
            for method in method_order
        }
        reference = by_method[reference_method]
        if not reference:
            continue
        for method in method_order:
            if method == reference_method:
                continue
            if not by_method[method]:
                continue
            groups = sorted(reference)
            if set(by_method[method]) != set(groups):
                raise ValueError(
                    f"Cannot pair {reference_method!r} and {method!r} "
                    f"by group for {metric}."
                )
            reference_values = np.asarray([reference[group] for group in groups])
            method_values = np.asarray([by_method[method][group] for group in groups])
            delta = method_values - reference_values
            ci_low, ci_high = _bootstrap_mean_ci(delta)
            reference_wins = delta < -1e-12 if higher_is_better else delta > 1e-12
            method_wins = delta > 1e-12 if higher_is_better else delta < -1e-12
            result.append(
                {
                    "reference_method": reference_method,
                    "method": method,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "reference_mean": float(reference_values.mean()),
                    "method_mean": float(method_values.mean()),
                    "mean_delta": float(delta.mean()),
                    "mean_delta_ci95_low": ci_low,
                    "mean_delta_ci95_high": ci_high,
                    "median_delta": float(np.median(delta)),
                    "std_delta": float(delta.std(ddof=0)),
                    "sign_flip_pvalue": _exact_sign_flip_pvalue(delta),
                    "n_groups": len(groups),
                    "reference_better": int(np.sum(reference_wins)),
                    "method_better": int(np.sum(method_wins)),
                    "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                }
            )
    return result


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 20_000,
    seed: int = 7,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def _exact_sign_flip_pvalue(values: np.ndarray) -> float:
    if len(values) > 20:
        raise ValueError("Exact paired sign-flip test supports at most 20 groups.")
    observed = abs(float(values.mean()))
    permuted = np.fromiter(
        (
            abs(float(np.mean(values * signs)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ),
        dtype=np.float64,
        count=2 ** len(values),
    )
    return float(np.mean(permuted >= observed - 1e-15))


def _metric_keys(rows: list[dict]) -> list[str]:
    return [key for key in numeric_keys(rows) if key not in ROW_META_KEYS]


def _write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{float(value):.6f}"
                    if isinstance(value, (float, np.floating))
                    else value
                    for key, value in row.items()
                }
            )
