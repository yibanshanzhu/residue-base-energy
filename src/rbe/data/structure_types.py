from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtomRecord:
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    coord: np.ndarray
    element: str


@dataclass
class ResidueRecord:
    resname: str
    chain: str
    resseq: int
    icode: str
    atoms: list[AtomRecord]

    @property
    def residue_id(self) -> str:
        suffix = self.icode if self.icode else ""
        return f"{self.chain}:{self.resseq}{suffix}"
