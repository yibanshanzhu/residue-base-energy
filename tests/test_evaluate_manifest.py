from __future__ import annotations

import numpy as np

from rbe.eval.evaluate_manifest import evaluate_pair, summarize_rows
from rbe.eval.metrics import best_threshold_metrics
from rbe.eval.summary import global_pwm_mae_summary_row


def _write_npz_pair(tmp_path, name: str):
    target = tmp_path / f"{name}.npz"
    pred = tmp_path / f"{name}.pred.npz"
    pwm = np.asarray(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.1, 0.7, 0.1],
        ],
        dtype=np.float32,
    )
    A_base = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    A_backbone = np.asarray([[1, 1, 0], [0, 0, 1]], dtype=np.float32)
    A_contact = np.maximum(A_base, A_backbone)
    site = A_contact.max(axis=1)
    np.savez_compressed(
        target,
        pwm_target=pwm,
        A_base_label=A_base,
        A_backbone_label=A_backbone,
        A_contact_label=A_contact,
        site_label=site,
    )
    np.savez_compressed(
        pred,
        pwm=pwm,
        A_base=A_base,
        A_backbone=A_backbone,
        A_contact=A_contact,
        site_prob=site,
    )
    return target, pred


def test_evaluate_pair_and_summary(tmp_path):
    target, pred = _write_npz_pair(tmp_path, "sample")
    row = evaluate_pair(target, pred)
    assert row["pwm_mae_sample"] < 1e-6
    assert "pwm_mae" not in row
    assert row["pwm_kl"] < 1e-6
    assert row["A_base_ap"] == 1.0
    assert row["A_backbone_top_l_precision"] == 1.0
    assert row["site_f1"] == 1.0

    summary = summarize_rows([row, row])
    by_metric = {item["metric"]: item for item in summary}
    assert by_metric["pwm_kl"]["n"] == 2
    assert by_metric["site_f1"]["mean"] == 1.0


def test_best_threshold_metrics_are_global_diagnostics():
    y_true = np.asarray([1, 1, 0, 0], dtype=np.int64)
    y_score = np.asarray([0.42, 0.38, 0.10, 0.05], dtype=np.float32)

    metrics = best_threshold_metrics(y_true, y_score)

    assert metrics["f1_at_0.5"] == 0.0
    assert metrics["best_f1_diagnostic"] == 1.0
    assert metrics["best_f1_threshold_diagnostic"] < 0.5


def test_global_pwm_mae_pools_valid_columns_instead_of_samples(tmp_path):
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    sample1 = tmp_path / "sample1.npz"
    sample2 = tmp_path / "sample2.npz"

    np.savez_compressed(
        sample1,
        pwm_target=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        pwm_mask=np.asarray([1], dtype=np.float32),
    )
    np.savez_compressed(
        pred_dir / "sample1.pred.npz",
        pwm=np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.savez_compressed(
        sample2,
        pwm_target=np.asarray(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=np.float32,
        ),
        pwm_mask=np.asarray([1, 1, 0], dtype=np.float32),
    )
    np.savez_compressed(
        pred_dir / "sample2.pred.npz",
        pwm=np.asarray(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=np.float32,
        ),
    )

    row = global_pwm_mae_summary_row([sample1, sample2], pred_dir, ".pred.npz")

    assert row["metric"] == "pwm_mae"
    assert row["n"] == 3
    assert np.isclose(row["mean"], 2.0 / 3.0)


def test_evaluate_pair_masks_unknown_A_base_positions(tmp_path):
    target = tmp_path / "sample.npz"
    pred = tmp_path / "sample.pred.npz"
    pwm = np.asarray([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
    A_base = np.asarray([[1, 1]], dtype=np.float32)
    A_base_mask = np.asarray([[1, 0]], dtype=np.float32)
    A_backbone = np.zeros_like(A_base)
    site = A_base.max(axis=1)
    np.savez_compressed(
        target,
        pwm_target=pwm,
        A_base_label=A_base,
        A_base_mask=A_base_mask,
        A_backbone_label=A_backbone,
        A_contact_label=A_base,
        site_label=site,
    )
    np.savez_compressed(
        pred,
        pwm=pwm,
        A_base=np.asarray([[0.9, 0.1]], dtype=np.float32),
        A_backbone=A_backbone,
        A_contact=A_base,
        site_prob=site,
    )

    row = evaluate_pair(target, pred)

    assert row["A_base_top_l"] == 1.0
    assert row["A_base_top_l_precision"] == 1.0


def test_evaluate_pair_masks_unobserved_pwm_columns_for_contact_maps(tmp_path):
    target = tmp_path / "sample.npz"
    pred = tmp_path / "sample.pred.npz"
    pwm = np.asarray([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
    pwm_mask = np.asarray([1, 0], dtype=np.float32)
    label = np.asarray([[1, 1]], dtype=np.float32)
    np.savez_compressed(
        target,
        pwm_target=pwm,
        pwm_mask=pwm_mask,
        A_base_label=label,
        A_base_mask=np.asarray([[1, 0]], dtype=np.float32),
        A_backbone_label=label,
        A_contact_label=label,
        site_label=np.ones((1,), dtype=np.float32),
    )
    np.savez_compressed(
        pred,
        pwm=pwm,
        A_base=np.asarray([[0.9, 0.1]], dtype=np.float32),
        A_backbone=np.asarray([[0.9, 0.1]], dtype=np.float32),
        A_contact=np.asarray([[0.9, 0.1]], dtype=np.float32),
        site_prob=np.ones((1,), dtype=np.float32),
    )

    row = evaluate_pair(target, pred)

    assert row["A_backbone_top_l"] == 1.0
    assert row["A_contact_top_l"] == 1.0
