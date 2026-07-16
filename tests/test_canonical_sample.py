from __future__ import annotations

import numpy as np

from rbe.data.processed_sample import _canonicalize_sample_slots


def test_canonical_sample_reverses_all_slot_supervision():
    pwm = np.asarray(
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.7]], dtype=np.float32
    )
    slot_to_dna_index = np.asarray([5, 6], dtype=np.int64)
    slot_arrays = {
        "pwm_mask": np.asarray([0, 1], dtype=np.float32),
        "A_base_label": np.asarray([[1, 2], [3, 4]], dtype=np.float32),
    }

    _, canonical_slots, canonical_arrays, reverse = _canonicalize_sample_slots(
        pwm, slot_to_dna_index, slot_arrays
    )

    assert reverse
    np.testing.assert_array_equal(canonical_slots, [6, 5])
    np.testing.assert_array_equal(canonical_arrays["pwm_mask"], [1, 0])
    np.testing.assert_array_equal(
        canonical_arrays["A_base_label"], [[2, 1], [4, 3]]
    )
