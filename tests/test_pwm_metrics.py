from __future__ import annotations

import numpy as np

from rbe.eval.metrics import pwm_mae


def test_pwm_mae_matches_per_sample_position_l1_definition():
    target = np.asarray(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float32,
    )
    pred = np.asarray(
        [
            [0.1, 0.7, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float32,
    )

    assert np.isclose(pwm_mae(target, pred, mask=np.ones(2)), 0.6)


def test_pwm_mae_only_uses_masked_columns():
    target = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]],
        dtype=np.float32,
    )
    pred = np.asarray(
        [[0.0, 1.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]],
        dtype=np.float32,
    )

    assert np.isclose(pwm_mae(target, pred, mask=np.asarray([0, 1])), 0.0)
