from __future__ import annotations

from pathlib import Path

import numpy as np

from rbe.eval.io import load_npz, pred_path_for_sample, read_manifest


def predict_family_baselines(
    train_manifest: str | Path,
    test_manifest: str | Path,
    out_root: str | Path,
) -> dict[str, Path]:
    train_paths = read_manifest(train_manifest)
    test_paths = read_manifest(test_manifest)
    train = [(path, load_npz(path)) for path in train_paths]
    orientation, pwm_shape = _validate_family_contract([data for _, data in train])

    mean_pwm = _group_balanced_mean_pwm([data for _, data in train])
    train_embeddings = np.stack([_pooled_esm(data) for _, data in train])
    output_dirs = {
        "nearest_esm": Path(out_root) / "nearest_esm" / "preds",
        "family_mean": Path(out_root) / "family_mean" / "preds",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for target_path in test_paths:
        target = load_npz(target_path)
        target_orientation, target_shape = _validate_family_contract([target])
        if (target_orientation, target_shape) != (orientation, pwm_shape):
            raise ValueError(
                f"{target_path}: train/test family contract mismatch: "
                f"train={(orientation, pwm_shape)}, "
                f"test={(target_orientation, target_shape)}."
            )
        embedding = _pooled_esm(target)
        nearest_index = int(np.argmax(train_embeddings @ embedding))
        nearest_path, nearest = train[nearest_index]

        nearest_output = pred_path_for_sample(
            target_path, output_dirs["nearest_esm"], ".pred.npz"
        )
        np.savez_compressed(
            nearest_output,
            pwm=nearest["pwm_target"],
            pwm_orientation=np.asarray(orientation),
            baseline_source_sample=np.asarray(nearest_path.stem),
            baseline_source_group=np.asarray(str(nearest["protein_group"])),
        )
        mean_output = pred_path_for_sample(
            target_path, output_dirs["family_mean"], ".pred.npz"
        )
        np.savez_compressed(
            mean_output,
            pwm=mean_pwm,
            pwm_orientation=np.asarray(orientation),
        )
    return output_dirs


def _pooled_esm(data: dict[str, np.ndarray]) -> np.ndarray:
    embedding = np.asarray(data["esm2_repr"], dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        raise ValueError("Encountered a zero pooled ESM embedding.")
    return embedding / norm


def _group_balanced_mean_pwm(data: list[dict[str, np.ndarray]]) -> np.ndarray:
    by_group: dict[str, list[np.ndarray]] = {}
    for sample in data:
        by_group.setdefault(str(sample["protein_group"]), []).append(sample["pwm_target"])
    group_pwms = [np.mean(values, axis=0) for values in by_group.values()]
    return np.mean(group_pwms, axis=0).astype(np.float32)


def _validate_family_contract(
    data: list[dict[str, np.ndarray]],
) -> tuple[str, tuple[int, ...]]:
    if not data:
        raise ValueError("Family baseline received no samples.")
    orientations = {str(sample["pwm_orientation"]) for sample in data}
    shapes = {sample["pwm_target"].shape for sample in data}
    if len(orientations) != 1 or not next(iter(orientations)).startswith(
        "family_reference:"
    ):
        raise ValueError(f"Expected one family_reference orientation, got {orientations}.")
    if len(shapes) != 1:
        raise ValueError(f"Family PWM shapes are inconsistent: {sorted(shapes)}.")
    return next(iter(orientations)), next(iter(shapes))
