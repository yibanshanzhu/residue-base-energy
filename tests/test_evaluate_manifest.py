from __future__ import annotations

import numpy as np

from rbe.eval.evaluate_manifest import evaluate_pair, summarize_rows


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
    assert row["pwm_kl"] < 1e-6
    assert row["A_base_ap"] == 1.0
    assert row["A_backbone_top_l_precision"] == 1.0
    assert row["site_f1"] == 1.0

    summary = summarize_rows([row, row])
    by_metric = {item["metric"]: item for item in summary}
    assert by_metric["pwm_kl"]["n"] == 2
    assert by_metric["site_f1"]["mean"] == 1.0
