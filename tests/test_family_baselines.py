from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rbe.eval.family_baselines import predict_family_baselines
from rbe.eval.io import load_npz, pred_path_for_sample


ORIENTATION = "family_reference:ETS:v1"


def _write_sample(
    path: Path,
    *,
    pwm: list[float],
    embedding: list[float],
    group: str,
    orientation: str = ORIENTATION,
) -> None:
    np.savez_compressed(
        path,
        pwm_target=np.asarray([pwm], dtype=np.float32),
        esm2_repr=np.asarray([embedding], dtype=np.float32),
        protein_group=np.asarray(group),
        pwm_orientation=np.asarray(orientation),
    )


def _write_manifest(path: Path, samples: list[Path]) -> None:
    path.write_text("".join(f"{sample.resolve()}\n" for sample in samples))


def test_family_baselines_balance_groups_and_select_nearest_esm(tmp_path: Path):
    train = [tmp_path / f"train{i}.npz" for i in range(3)]
    test = tmp_path / "test.npz"
    _write_sample(train[0], pwm=[1, 0, 0, 0], embedding=[1, 0], group="G1")
    _write_sample(train[1], pwm=[1, 0, 0, 0], embedding=[1, 0], group="G1")
    _write_sample(train[2], pwm=[0, 1, 0, 0], embedding=[0, 1], group="G2")
    _write_sample(test, pwm=[0, 1, 0, 0], embedding=[0, 1], group="G3")
    train_manifest = tmp_path / "train.txt"
    test_manifest = tmp_path / "test.txt"
    _write_manifest(train_manifest, train)
    _write_manifest(test_manifest, [test])

    outputs = predict_family_baselines(
        train_manifest, test_manifest, tmp_path / "predictions"
    )

    mean = load_npz(pred_path_for_sample(test, outputs["family_mean"], ".pred.npz"))
    nearest = load_npz(
        pred_path_for_sample(test, outputs["nearest_esm"], ".pred.npz")
    )
    np.testing.assert_allclose(mean["pwm"], [[0.5, 0.5, 0.0, 0.0]], atol=1e-7)
    np.testing.assert_allclose(nearest["pwm"], [[0.0, 1.0, 0.0, 0.0]])
    assert str(nearest["baseline_source_group"]) == "G2"


def test_family_baselines_reject_train_test_orientation_mismatch(tmp_path: Path):
    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    _write_sample(train, pwm=[1, 0, 0, 0], embedding=[1, 0], group="G1")
    _write_sample(
        test,
        pwm=[1, 0, 0, 0],
        embedding=[1, 0],
        group="G2",
        orientation="family_reference:ETS:v2",
    )
    train_manifest = tmp_path / "train.txt"
    test_manifest = tmp_path / "test.txt"
    _write_manifest(train_manifest, [train])
    _write_manifest(test_manifest, [test])

    with pytest.raises(ValueError, match="train/test family contract mismatch"):
        predict_family_baselines(train_manifest, test_manifest, tmp_path / "out")
