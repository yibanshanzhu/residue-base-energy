from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from rbe.eval.io import load_npz, pred_path_for_sample
from rbe.eval.mechanism_ablation import write_mechanism_ablations


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def test_mechanism_ablations_preserve_mass_and_shuffle_residue_pairing(
    tmp_path: Path,
):
    sample = tmp_path / "sample.npz"
    sample.touch()
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{sample}\n")
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    A = np.asarray([[0.8, 0.1], [0.1, 0.3], [0.2, 0.6]], dtype=np.float32)
    E = np.arange(24, dtype=np.float32).reshape(3, 2, 4) / 10.0
    prior_logits = np.asarray(
        [[0.2, 0.1, 0.0, -0.1], [-0.2, 0.0, 0.2, 0.4]],
        dtype=np.float32,
    )
    np.savez_compressed(
        pred_path_for_sample(sample, pred_dir, ".pred.npz"),
        A_base=A,
        E=E,
        pwm_prior_logits=prior_logits,
        residual_scale=np.asarray(0.25, dtype=np.float32),
        pwm_orientation=np.asarray("family_reference:ETS:v1"),
    )

    outputs = write_mechanism_ablations(
        manifest, pred_dir, tmp_path / "ablations"
    )

    uniform = load_npz(
        pred_path_for_sample(sample, outputs["uniform_gate"], ".pred.npz")
    )
    expected_uniform_gate = np.broadcast_to(A.sum(axis=0) / A.shape[0], A.shape)
    expected_uniform_logits = prior_logits + 0.25 * np.sum(
        expected_uniform_gate[..., None] * E, axis=0
    )
    np.testing.assert_allclose(uniform["pwm_logits"], expected_uniform_logits)
    np.testing.assert_allclose(uniform["pwm"], _softmax(expected_uniform_logits))

    shuffled = load_npz(
        pred_path_for_sample(sample, outputs["shuffled_energy"], ".pred.npz")
    )
    seed = int.from_bytes(hashlib.sha256(b"sample").digest()[:8], "little")
    permutation = np.random.default_rng(seed).permutation(A.shape[0])
    expected_shuffled_logits = prior_logits + 0.25 * np.sum(
        A[..., None] * E[permutation], axis=0
    )
    np.testing.assert_allclose(shuffled["pwm_logits"], expected_shuffled_logits)
    assert str(shuffled["pwm_orientation"]) == "family_reference:ETS:v1"
