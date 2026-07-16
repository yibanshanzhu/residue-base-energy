from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rbe.eval.family_evaluation import evaluate_family_methods
from rbe.eval.io import pred_path_for_sample


ORIENTATION = "family_reference:ETS:v1"


def _write_target(path: Path, group: str) -> None:
    np.savez_compressed(
        path,
        pwm_target=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        pwm_orientation=np.asarray(ORIENTATION),
        protein_group=np.asarray(group),
    )


def _write_prediction(path: Path, base: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pwm = np.zeros((1, 4), dtype=np.float32)
    pwm[0, base] = 1.0
    np.savez_compressed(
        path,
        pwm=pwm,
        pwm_orientation=np.asarray(ORIENTATION),
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_family_evaluation_weights_uniprot_groups_equally(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    folds = benchmark / "folds"
    folds.mkdir(parents=True)
    targets = [tmp_path / "g1a.npz", tmp_path / "g1b.npz", tmp_path / "g2.npz"]
    _write_target(targets[0], "G1")
    _write_target(targets[1], "G1")
    _write_target(targets[2], "G2")
    (benchmark / "fold_table.tsv").write_text(
        "fold\ttest_group\tvalid_group\ttrain_groups\n"
        "0\tG1\tG2\t\n"
        "1\tG2\tG1\t\n"
    )
    (folds / "fold0_test.txt").write_text(
        "".join(f"{path.resolve()}\n" for path in targets[:2])
    )
    (folds / "fold1_test.txt").write_text(f"{targets[2].resolve()}\n")

    full_template = tmp_path / "runs/full/fold{fold}/preds"
    alt_template = tmp_path / "runs/alt/fold{fold}/preds"
    for target in targets[:2]:
        _write_prediction(
            pred_path_for_sample(target, Path(str(full_template).format(fold=0)), ".pred.npz"),
            0,
        )
        _write_prediction(
            pred_path_for_sample(target, Path(str(alt_template).format(fold=0)), ".pred.npz"),
            1,
        )
    _write_prediction(
        pred_path_for_sample(
            targets[2], Path(str(full_template).format(fold=1)), ".pred.npz"
        ),
        1,
    )
    _write_prediction(
        pred_path_for_sample(
            targets[2], Path(str(alt_template).format(fold=1)), ".pred.npz"
        ),
        1,
    )

    result = evaluate_family_methods(
        benchmark,
        {"full": full_template, "alt": alt_template},
        tmp_path / "evaluation",
        reference_method="full",
    )

    summary = _read_tsv(result.summary_tsv)
    full_mae = next(
        row for row in summary if row["method"] == "full" and row["metric"] == "pwm_mae"
    )
    assert np.isclose(float(full_mae["mean"]), 1.0)
    assert full_mae["n_groups"] == "2"

    groups = _read_tsv(result.per_group_tsv)
    g1 = next(
        row
        for row in groups
        if row["method"] == "full" and row["protein_group"] == "G1"
    )
    assert g1["n_samples"] == "2"

    paired = _read_tsv(result.paired_pwm_mae_tsv)[0]
    assert paired["method"] == "alt"
    assert np.isclose(float(paired["mean_delta"]), 1.0)
    assert paired["reference_better"] == "1"
    assert paired["ties"] == "1"
