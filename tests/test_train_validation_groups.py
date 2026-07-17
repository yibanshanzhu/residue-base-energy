from __future__ import annotations

import numpy as np

from rbe.train import _mean_validation_values


def test_validation_metric_weights_groups_equally():
    values = [("a1", 0.0), ("a2", 0.0), ("b", 3.0)]
    groups = {"a1": "A", "a2": "A", "b": "B"}

    assert np.isclose(_mean_validation_values(values, None), 1.0)
    assert np.isclose(_mean_validation_values(values, groups), 1.5)
