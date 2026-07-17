from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rbe.eval.cluster_evaluation import evaluate_cluster_methods
from rbe.eval.io import pred_path_for_sample


def _write_pwm(path: Path, base: int, *, prediction: bool) -> None:
    pwm = np.zeros((1, 4), dtype=np.float32)
    pwm[0, base] = 1.0
    arrays = {"pwm_orientation": np.asarray("canonical")}
    arrays["pwm" if prediction else "pwm_target"] = pwm
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_cluster_evaluation_weights_components_equally(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    folds = benchmark / "folds"
    folds.mkdir(parents=True)
    targets = [tmp_path / name for name in ("a1.npz", "a2.npz", "b.npz", "c.npz")]
    for target in targets:
        _write_pwm(target, 0, prediction=False)
    (benchmark / "sample_table.tsv").write_text(
        "sample_id\tcomponent\tfold\n"
        "a1\tcomponent_a\t0\n"
        "a2\tcomponent_a\t0\n"
        "b\tcomponent_b\t1\n"
        "c\tcomponent_c\t2\n"
    )
    (benchmark / "fold_table.tsv").write_text(
        "fold\tvalid_fold\n0\t1\n1\t2\n2\t0\n"
    )
    for fold, members in enumerate((targets[:2], targets[2:3], targets[3:])):
        (folds / f"fold{fold}_test.txt").write_text(
            "".join(f"{target.resolve()}\n" for target in members)
        )

    full_template = tmp_path / "runs/full/fold{fold}/preds"
    alt_template = tmp_path / "runs/alt/fold{fold}/preds"
    for fold, members in enumerate((targets[:2], targets[2:3], targets[3:])):
        for target in members:
            full_base = 0 if fold == 0 else 1
            _write_pwm(
                pred_path_for_sample(
                    target, Path(str(full_template).format(fold=fold)), ".pred.npz"
                ),
                full_base,
                prediction=True,
            )
            _write_pwm(
                pred_path_for_sample(
                    target, Path(str(alt_template).format(fold=fold)), ".pred.npz"
                ),
                1,
                prediction=True,
            )

    result = evaluate_cluster_methods(
        benchmark,
        {"full": full_template, "alt": alt_template},
        tmp_path / "evaluation",
        reference_method="full",
    )

    components = _read_rows(result.per_component_tsv)
    full_components = [row for row in components if row["method"] == "full"]
    assert len(full_components) == 3
    assert next(row for row in full_components if row["component"] == "component_a")[
        "n_samples"
    ] == "2"
    expected = np.mean([float(row["pwm_mae"]) for row in full_components])
    summary = _read_rows(result.summary_tsv)
    full_mae = next(
        row for row in summary if row["method"] == "full" and row["metric"] == "pwm_mae"
    )
    assert np.isclose(float(full_mae["mean"]), expected)
    paired = _read_rows(result.paired_metrics_tsv)
    mae = next(row for row in paired if row["metric"] == "pwm_mae")
    assert mae["n_groups"] == "3"
