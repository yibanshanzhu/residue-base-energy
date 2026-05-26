from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from rbe.constants import AA_TO_IDX
from rbe.data.pwm import normalize_pwm


class RBEDataset(Dataset):
    def __init__(self, paths: Iterable[str | Path]):
        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError("RBEDataset received no npz files.")

    @classmethod
    def from_dir(cls, data_dir: str | Path) -> "RBEDataset":
        paths = sorted(Path(data_dir).glob("*.npz"))
        return cls(paths)

    @classmethod
    def from_manifest(cls, manifest: str | Path) -> "RBEDataset":
        root = Path(manifest).resolve().parent
        paths = []
        with Path(manifest).open() as handle:
            for line in handle:
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                path = Path(item)
                paths.append(path if path.is_absolute() else root / path)
        return cls(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        with np.load(path, allow_pickle=False) as data:
            residue_aa = data["residue_aa"].astype(str)
            aa_idx = np.array([AA_TO_IDX[aa] for aa in residue_aa], dtype=np.int64)
            A_base_label = data["A_base_label"] if "A_base_label" in data else data["A_label"]
            if "A_backbone_label" in data:
                A_backbone_label = data["A_backbone_label"]
            else:
                A_backbone_label = np.zeros_like(A_base_label)
            if "A_contact_label" in data:
                A_contact_label = data["A_contact_label"]
            else:
                A_contact_label = np.maximum(A_base_label, A_backbone_label)
            sample = {
                "path": str(path),
                "residue_ids": data["residue_ids"].astype(str),
                "residue_aa": residue_aa,
                "aa_idx": torch.from_numpy(aa_idx),
                "residue_xyz": torch.from_numpy(data["residue_xyz"].astype(np.float32)),
                "edge_index": torch.from_numpy(data["residue_edges"].astype(np.int64)),
                "edge_attr": torch.from_numpy(data["edge_attr"].astype(np.float32)),
                "esm2_repr": torch.from_numpy(data["esm2_repr"].astype(np.float32)),
                "pwm_target": torch.from_numpy(normalize_pwm(data["pwm_target"])),
                "A_label": torch.from_numpy(A_base_label.astype(np.float32)),
                "A_base_label": torch.from_numpy(A_base_label.astype(np.float32)),
                "A_backbone_label": torch.from_numpy(
                    A_backbone_label.astype(np.float32)
                ),
                "A_contact_label": torch.from_numpy(A_contact_label.astype(np.float32)),
                "site_label": torch.from_numpy(data["site_label"].astype(np.float32)),
                "slot_to_dna_index": torch.from_numpy(
                    data["slot_to_dna_index"].astype(np.int64)
                ),
            }
        return sample


def rbe_collate(batch: list[dict]) -> list[dict]:
    return batch


def to_device(sample: dict, device: torch.device | str) -> dict:
    out = {}
    for key, value in sample.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out
