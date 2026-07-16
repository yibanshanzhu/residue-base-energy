from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from rbe.data.pwm import normalize_pwm


def load_group_balanced_pwm_prior(paths: Iterable[str | Path]) -> np.ndarray:
    samples = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            samples.append({key: data[key] for key in data.files})
    return group_balanced_pwm_prior(samples)


def group_balanced_pwm_prior(samples: list[dict[str, np.ndarray]]) -> np.ndarray:
    if not samples:
        raise ValueError("Cannot calculate a family PWM prior from zero samples.")
    orientations = {str(sample["pwm_orientation"]) for sample in samples}
    shapes = {np.asarray(sample["pwm_target"]).shape for sample in samples}
    if len(orientations) != 1 or not next(iter(orientations)).startswith(
        "family_reference:"
    ):
        raise ValueError(
            f"Family PWM prior requires one family_reference orientation, "
            f"got {orientations}."
        )
    if len(shapes) != 1:
        raise ValueError(f"Family PWM shapes are inconsistent: {sorted(shapes)}.")

    by_group: dict[str, list[np.ndarray]] = {}
    for sample in samples:
        if "protein_group" not in sample:
            raise ValueError("Family PWM prior sample is missing protein_group.")
        group = str(sample["protein_group"])
        by_group.setdefault(group, []).append(normalize_pwm(sample["pwm_target"]))
    group_means = [np.mean(group_pwms, axis=0) for group_pwms in by_group.values()]
    return normalize_pwm(np.mean(group_means, axis=0))
