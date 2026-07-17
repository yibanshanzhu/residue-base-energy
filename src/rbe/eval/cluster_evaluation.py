from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rbe.eval.family_evaluation import paired_metrics
from rbe.eval.io import load_npz, pred_path_for_sample, read_manifest
from rbe.eval.pair_metrics import evaluate_pair
from rbe.eval.summary import numeric_keys


META_KEYS = {
    "component",
    "fold",
    "method",
    "n_samples",
    "pred_path",
    "sample",
    "target_path",
}


@dataclass(frozen=True)
class ClusterEvaluationResult:
    per_sample_tsv: Path
    per_component_tsv: Path
    summary_tsv: Path
    paired_metrics_tsv: Path


def evaluate_cluster_methods(
    benchmark_root: str | Path,
    method_templates: dict[str, str | Path],
    out_root: str | Path,
    *,
    reference_method: str,
    suffix: str = ".pred.npz",
) -> ClusterEvaluationResult:
    if reference_method not in method_templates:
        raise ValueError(f"Unknown reference method {reference_method!r}.")
    benchmark = Path(benchmark_root)
    sample_metadata = _read_sample_metadata(benchmark / "sample_table.tsv")
    folds = _read_fold_ids(benchmark / "fold_table.tsv")
    method_order = list(method_templates)
    sample_rows = []

    for fold in folds:
        targets = read_manifest(benchmark / "folds" / f"fold{fold}_test.txt")
        for method, template in method_templates.items():
            pred_dir = Path(str(template).format(fold=fold))
            for target_path in targets:
                sample_id = target_path.stem
                metadata = sample_metadata.get(sample_id)
                if metadata is None or int(metadata["fold"]) != fold:
                    raise ValueError(
                        f"{target_path}: sample table fold does not match test fold {fold}."
                    )
                pred_path = pred_path_for_sample(target_path, pred_dir, suffix)
                if not pred_path.exists():
                    raise FileNotFoundError(f"Missing prediction: {pred_path}")
                metrics = evaluate_pair(target_path, pred_path)
                sample_rows.append(
                    {
                        "method": method,
                        "fold": fold,
                        "component": metadata["component"],
                        **metrics,
                    }
                )

    component_rows = _aggregate_components(sample_rows, method_order)
    _validate_components(component_rows, method_order)
    summary_rows = _summarize(component_rows, method_order)
    paired_rows = paired_metrics(
        component_rows,
        method_order,
        reference_method,
        group_key="component",
    )

    output = Path(out_root)
    output.mkdir(parents=True, exist_ok=True)
    result = ClusterEvaluationResult(
        per_sample_tsv=output / "per_sample.tsv",
        per_component_tsv=output / "per_component.tsv",
        summary_tsv=output / "summary.tsv",
        paired_metrics_tsv=output / "paired_metrics.tsv",
    )
    _write_tsv(
        result.per_sample_tsv,
        sample_rows,
        [
            "method",
            "fold",
            "component",
            "sample",
            "target_path",
            "pred_path",
            *_metric_keys(sample_rows),
        ],
    )
    _write_tsv(
        result.per_component_tsv,
        component_rows,
        ["method", "fold", "component", "n_samples", *_metric_keys(component_rows)],
    )
    _write_tsv(
        result.summary_tsv,
        summary_rows,
        ["method", "metric", "mean", "std", "n_components"],
    )
    _write_tsv(
        result.paired_metrics_tsv,
        paired_rows,
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


def _read_sample_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {row["sample_id"]: row for row in rows}


def _read_fold_ids(path: Path) -> list[int]:
    with path.open(newline="") as handle:
        return [int(row["fold"]) for row in csv.DictReader(handle, delimiter="\t")]


def _aggregate_components(rows: list[dict], method_order: list[str]) -> list[dict]:
    metrics = _metric_keys(rows)
    result = []
    for method in method_order:
        method_rows = [row for row in rows if row["method"] == method]
        for component in sorted({row["component"] for row in method_rows}):
            members = [row for row in method_rows if row["component"] == component]
            folds = {int(row["fold"]) for row in members}
            if len(folds) != 1:
                raise ValueError(f"Component {component} spans test folds {sorted(folds)}.")
            aggregate = {
                "method": method,
                "fold": next(iter(folds)),
                "component": component,
                "n_samples": len(members),
            }
            for metric in metrics:
                values = [
                    float(row[metric])
                    for row in members
                    if metric in row and np.isfinite(float(row[metric]))
                ]
                if values:
                    aggregate[metric] = float(np.mean(values))
            result.append(aggregate)
    return result


def _validate_components(rows: list[dict], method_order: list[str]) -> None:
    expected = {row["component"] for row in rows if row["method"] == method_order[0]}
    for method in method_order[1:]:
        observed = {row["component"] for row in rows if row["method"] == method}
        if observed != expected:
            raise ValueError(f"Method {method!r} has different test components.")


def _summarize(rows: list[dict], method_order: list[str]) -> list[dict]:
    result = []
    for method in method_order:
        method_rows = [row for row in rows if row["method"] == method]
        for metric in _metric_keys(method_rows):
            values = np.asarray([float(row[metric]) for row in method_rows if metric in row])
            result.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "n_components": int(values.size),
                }
            )
    return result


def _metric_keys(rows: list[dict]) -> list[str]:
    return [key for key in numeric_keys(rows) if key not in META_KEYS]


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
