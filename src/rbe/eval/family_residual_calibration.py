from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rbe.eval.io import get_pwm, load_npz, pred_path_for_sample, read_manifest
from rbe.eval.metrics import pwm_mae


DEFAULT_RESIDUAL_SCALES = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))


@dataclass(frozen=True)
class ResidualCalibrationResult:
    residual_scale: float
    prior_valid_mae: float
    calibrated_valid_mae: float
    valid_mae_by_scale: dict[float, float]
    output_dir: Path


def calibrate_family_residual(
    valid_manifest: str | Path,
    valid_prediction_dir: str | Path,
    test_manifest: str | Path,
    test_prediction_dir: str | Path,
    output_dir: str | Path,
    *,
    residual_scales: tuple[float, ...] = DEFAULT_RESIDUAL_SCALES,
    suffix: str = ".pred.npz",
) -> ResidualCalibrationResult:
    scales = _validate_scales(residual_scales)
    valid_pairs = _load_pairs(valid_manifest, valid_prediction_dir, suffix)
    scores = {
        scale: float(
            np.mean(
                [
                    pwm_mae(get_pwm(target), _scaled_pwm(prediction, scale))
                    for target, prediction in valid_pairs
                ]
            )
        )
        for scale in scales
    }
    selected_scale = min(scales, key=lambda scale: (scores[scale], scale))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for target_path in read_manifest(test_manifest):
        target = load_npz(target_path)
        source_path = pred_path_for_sample(
            target_path, test_prediction_dir, suffix
        )
        prediction = load_npz(source_path)
        _validate_pair_orientation(target, prediction, target_path, source_path)
        logits = _scaled_logits(prediction, selected_scale)
        arrays = {
            **prediction,
            "pwm": _softmax(logits),
            "pwm_logits": logits.astype(np.float32),
            "residual_scale": np.asarray(selected_scale, dtype=np.float32),
        }
        np.savez_compressed(
            pred_path_for_sample(target_path, destination, suffix),
            **arrays,
        )

    return ResidualCalibrationResult(
        residual_scale=selected_scale,
        prior_valid_mae=scores[0.0],
        calibrated_valid_mae=scores[selected_scale],
        valid_mae_by_scale=scores,
        output_dir=destination,
    )


def _load_pairs(
    manifest: str | Path,
    prediction_dir: str | Path,
    suffix: str,
) -> list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    pairs = []
    for target_path in read_manifest(manifest):
        prediction_path = pred_path_for_sample(
            target_path, prediction_dir, suffix
        )
        target = load_npz(target_path)
        prediction = load_npz(prediction_path)
        _validate_pair_orientation(
            target, prediction, target_path, prediction_path
        )
        pairs.append((target, prediction))
    if not pairs:
        raise ValueError(f"Residual calibration manifest is empty: {manifest}")
    return pairs


def _validate_pair_orientation(
    target: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    target_path: str | Path,
    prediction_path: str | Path,
) -> None:
    target_orientation = str(target["pwm_orientation"])
    prediction_orientation = str(prediction["pwm_orientation"])
    if not target_orientation.startswith("family_reference:"):
        raise ValueError(
            f"{target_path}: calibration requires family_reference orientation."
        )
    if prediction_orientation != target_orientation:
        raise ValueError(
            f"{prediction_path}: orientation {prediction_orientation!r} does not "
            f"match target {target_orientation!r}."
        )


def _validate_scales(values: tuple[float, ...]) -> tuple[float, ...]:
    scales = tuple(sorted({float(value) for value in values}))
    if not scales or scales[0] != 0.0 or any(value < 0.0 for value in scales):
        raise ValueError("Residual scales must be non-negative and include 0.0.")
    return scales


def _scaled_pwm(prediction: dict[str, np.ndarray], scale: float) -> np.ndarray:
    return _softmax(_scaled_logits(prediction, scale))


def _scaled_logits(
    prediction: dict[str, np.ndarray], scale: float
) -> np.ndarray:
    prior = np.asarray(prediction["pwm_prior_logits"], dtype=np.float32)
    residual = np.asarray(prediction["pwm_residual_logits"], dtype=np.float32)
    if prior.shape != residual.shape or prior.ndim != 2 or prior.shape[1] != 4:
        raise ValueError(
            f"PWM prior/residual shapes must match [M, 4], got "
            f"{prior.shape} and {residual.shape}."
        )
    return prior + float(scale) * residual


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
