from __future__ import annotations

import numpy as np

from rbe.data.pwm import canonicalize_pwm
from rbe.eval.prediction import canonicalize_prediction_arrays, orient_prediction_arrays


def test_prediction_canonicalization_transforms_every_slot_axis():
    pwm = np.asarray(
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.7]], dtype=np.float32
    )
    _, reverse = canonicalize_pwm(pwm)
    assert reverse

    arrays = {
        "pwm": pwm,
        "pwm_logits": np.arange(8, dtype=np.float32).reshape(2, 4),
        "pwm_prior_logits": np.arange(8, dtype=np.float32).reshape(2, 4),
        "pwm_residual_logits": np.arange(8, dtype=np.float32).reshape(2, 4),
        "A_base": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "E": np.arange(16, dtype=np.float32).reshape(2, 2, 4),
        "site_prob": np.asarray([0.2, 0.8], dtype=np.float32),
    }

    result = canonicalize_prediction_arrays(arrays)

    np.testing.assert_array_equal(result["pwm_logits"], arrays["pwm_logits"][::-1, ::-1])
    np.testing.assert_array_equal(
        result["pwm_prior_logits"], arrays["pwm_prior_logits"][::-1, ::-1]
    )
    np.testing.assert_array_equal(
        result["pwm_residual_logits"], arrays["pwm_residual_logits"][::-1, ::-1]
    )
    np.testing.assert_array_equal(result["A_base"], arrays["A_base"][:, ::-1])
    np.testing.assert_array_equal(result["E"], arrays["E"][:, ::-1, ::-1])
    np.testing.assert_array_equal(result["site_prob"], arrays["site_prob"])
    assert bool(result["canonical_reverse_complement"])
    assert not canonicalize_pwm(result["pwm"])[1]


def test_family_reference_prediction_keeps_learned_slot_orientation():
    pwm = np.asarray(
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.7]], dtype=np.float32
    )

    result = orient_prediction_arrays({"pwm": pwm}, "family_reference:ETS:v1")

    np.testing.assert_allclose(result["pwm"], pwm)
    assert str(result["pwm_orientation"]) == "family_reference:ETS:v1"
    assert not bool(result["canonical_reverse_complement"])
