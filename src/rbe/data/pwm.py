from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _numeric_tokens(line: str) -> list[float]:
    cleaned = line.replace("[", " ").replace("]", " ").replace(",", " ")
    tokens = []
    for token in cleaned.split():
        try:
            tokens.append(float(token))
        except ValueError:
            continue
    return tokens


def normalize_pwm(pwm: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    pwm = np.asarray(pwm, dtype=np.float32)
    if pwm.ndim != 2 or pwm.shape[1] != 4:
        raise ValueError(f"PWM must have shape [M, 4], got {pwm.shape}.")
    pwm = np.clip(pwm, eps, None)
    return (pwm / pwm.sum(axis=1, keepdims=True)).astype(np.float32)


def reverse_complement_pwm(pwm: np.ndarray) -> np.ndarray:
    pwm = normalize_pwm(pwm)
    return pwm[::-1, ::-1].copy()


def canonicalize_pwm(pwm: np.ndarray) -> tuple[np.ndarray, bool]:
    """Choose one deterministic representative of a PWM/RC pair.

    The orientation with the lexicographically larger flattened [M, 4] matrix is
    canonical. Exact RC-palindromes keep their input orientation.
    """
    direct = normalize_pwm(pwm)
    reverse = reverse_complement_pwm(direct)
    use_reverse = tuple(reverse.ravel()) > tuple(direct.ravel())
    return (reverse if use_reverse else direct), use_reverse


def read_pwm(path: str | Path) -> np.ndarray:
    rows: list[list[float]] = []
    in_meme_matrix = False
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith(">"):
                continue
            lower = line.lower()
            if lower.startswith("letter-probability matrix"):
                in_meme_matrix = True
                continue
            if re.fullmatch(r"[acgtACGT\s]+", line):
                continue
            nums = _numeric_tokens(line)
            if len(nums) >= 4:
                rows.append(nums[-4:])
            elif in_meme_matrix and rows:
                break
    if not rows:
        raise ValueError(f"No PWM rows found in {path}.")
    return normalize_pwm(np.asarray(rows, dtype=np.float32))
