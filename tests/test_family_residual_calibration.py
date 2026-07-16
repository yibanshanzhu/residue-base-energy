from __future__ import annotations

from pathlib import Path

import numpy as np

from rbe.eval.family_residual_calibration import calibrate_family_residual
from rbe.eval.io import load_npz, pred_path_for_sample


ORIENTATION = "family_reference:ETS:v1"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _write_manifest(path: Path, sample: Path) -> None:
    path.write_text(f"{sample.resolve()}\n")


def test_family_residual_scale_is_selected_only_from_validation(tmp_path: Path):
    valid_target = tmp_path / "valid.npz"
    test_target = tmp_path / "test.npz"
    valid_manifest = tmp_path / "valid.txt"
    test_manifest = tmp_path / "test.txt"
    valid_pred_dir = tmp_path / "valid_preds"
    test_pred_dir = tmp_path / "test_preds"
    valid_pred_dir.mkdir()
    test_pred_dir.mkdir()
    _write_manifest(valid_manifest, valid_target)
    _write_manifest(test_manifest, test_target)

    prior_logits = np.zeros((1, 4), dtype=np.float32)
    residual_logits = np.asarray([[2.0, -2.0, 0.0, 0.0]], dtype=np.float32)
    valid_pwm = _softmax(prior_logits + 0.5 * residual_logits)
    for path, pwm in ((valid_target, valid_pwm), (test_target, valid_pwm)):
        np.savez_compressed(
            path,
            pwm_target=pwm,
            pwm_orientation=np.asarray(ORIENTATION),
        )
    for target, pred_dir in (
        (valid_target, valid_pred_dir),
        (test_target, test_pred_dir),
    ):
        np.savez_compressed(
            pred_path_for_sample(target, pred_dir, ".pred.npz"),
            pwm=_softmax(prior_logits + residual_logits),
            pwm_logits=prior_logits + residual_logits,
            pwm_prior_logits=prior_logits,
            pwm_residual_logits=residual_logits,
            pwm_orientation=np.asarray(ORIENTATION),
        )

    result = calibrate_family_residual(
        valid_manifest,
        valid_pred_dir,
        test_manifest,
        test_pred_dir,
        tmp_path / "calibrated",
        residual_scales=(0.0, 0.5, 1.0),
    )

    calibrated = load_npz(
        pred_path_for_sample(
            test_target, tmp_path / "calibrated", ".pred.npz"
        )
    )
    assert result.residual_scale == 0.5
    assert np.isclose(result.calibrated_valid_mae, 0.0, atol=1e-6)
    np.testing.assert_allclose(calibrated["pwm"], valid_pwm)
    assert float(calibrated["residual_scale"]) == 0.5
