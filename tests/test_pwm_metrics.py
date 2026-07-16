from __future__ import annotations

import numpy as np

from rbe.data.pwm import canonicalize_pwm, reverse_complement_pwm
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

    assert np.isclose(pwm_mae(target, pred), 0.6)


def test_pwm_mae_uses_all_columns():
    target = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]],
        dtype=np.float32,
    )
    pred = np.asarray(
        [[0.0, 1.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]],
        dtype=np.float32,
    )

    assert np.isclose(pwm_mae(target, pred), 1.0)


def test_canonical_pwm_is_identical_for_both_strand_orientations():
    pwm = np.asarray(
        [[0.1, 0.2, 0.3, 0.4], [0.6, 0.1, 0.2, 0.1]], dtype=np.float32
    )

    direct, _ = canonicalize_pwm(pwm)
    reverse, _ = canonicalize_pwm(reverse_complement_pwm(pwm))

    np.testing.assert_allclose(direct, reverse)
    assert not canonicalize_pwm(direct)[1]
