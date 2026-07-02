from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def get_pwm(data: dict) -> np.ndarray:
    for key in ("pwm", "PWM", "pwm_target", "P"):
        if key in data:
            pwm = data[key]
            if pwm.ndim == 2 and pwm.shape[1] == 4:
                return pwm.astype(np.float32)
    raise ValueError("No [M,4] PWM found. Tried keys: pwm, PWM, pwm_target, P.")


def read_manifest(path: str | Path, limit: int = 0) -> list[Path]:
    manifest = Path(path)
    root = manifest.resolve().parent
    samples = []
    with manifest.open() as handle:
        for line in handle:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            sample = Path(item)
            samples.append(sample if sample.is_absolute() else root / sample)
            if limit and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"No samples found in manifest: {manifest}")
    return samples


def pred_path_for_sample(
    sample_path: str | Path, pred_dir: str | Path, suffix: str
) -> Path:
    return Path(pred_dir) / f"{Path(sample_path).stem}{suffix}"
