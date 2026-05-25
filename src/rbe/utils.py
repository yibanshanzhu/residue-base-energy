from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

