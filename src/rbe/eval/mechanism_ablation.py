from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from rbe.eval.io import load_npz, pred_path_for_sample, read_manifest


def write_mechanism_ablations(
    manifest: str | Path,
    prediction_dir: str | Path,
    out_root: str | Path,
    suffix: str = ".pred.npz",
) -> dict[str, Path]:
    output_dirs = {
        "uniform_gate": Path(out_root) / "uniform_gate" / "preds",
        "shuffled_energy": Path(out_root) / "shuffled_energy" / "preds",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for sample_path in read_manifest(manifest):
        prediction = load_npz(
            pred_path_for_sample(sample_path, prediction_dir, suffix)
        )
        A = np.asarray(prediction["A_base"], dtype=np.float32)
        E = np.asarray(prediction["E"], dtype=np.float32)
        if E.shape != (*A.shape, 4):
            raise ValueError(
                f"{sample_path.stem}: E shape {E.shape} does not match A {A.shape}."
            )
        orientation = str(prediction["pwm_orientation"])
        prior_logits = np.asarray(prediction["pwm_prior_logits"], dtype=np.float32)
        if prior_logits.shape != (A.shape[1], 4):
            raise ValueError(
                f"{sample_path.stem}: invalid PWM prior shape {prior_logits.shape}."
            )

        gate_mass = A.sum(axis=0, keepdims=True)
        uniform_gate = np.broadcast_to(gate_mass / A.shape[0], A.shape)
        uniform_logits = prior_logits + np.sum(uniform_gate[..., None] * E, axis=0)
        _write_pwm_prediction(
            pred_path_for_sample(sample_path, output_dirs["uniform_gate"], suffix),
            uniform_logits,
            orientation,
        )

        seed = int.from_bytes(
            hashlib.sha256(sample_path.stem.encode()).digest()[:8], "little"
        )
        permutation = np.random.default_rng(seed).permutation(A.shape[0])
        shuffled_logits = prior_logits + np.sum(
            A[..., None] * E[permutation], axis=0
        )
        _write_pwm_prediction(
            pred_path_for_sample(sample_path, output_dirs["shuffled_energy"], suffix),
            shuffled_logits,
            orientation,
        )
    return output_dirs


def _write_pwm_prediction(path: Path, logits: np.ndarray, orientation: str) -> None:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    pwm = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
    np.savez_compressed(
        path,
        pwm=pwm,
        pwm_logits=logits.astype(np.float32),
        pwm_orientation=np.asarray(orientation),
    )
