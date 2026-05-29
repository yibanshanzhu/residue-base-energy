from __future__ import annotations

import numpy as np

from rbe.eval.metrics import pwm_mae


def test_pwm_mae_matches_deeppbs_position_l1_definition():
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

    assert np.isclose(pwm_mae(target, pred), 0.6)
