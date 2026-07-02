from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rbe.data.atom_geometry import (
    backbone_heavy_atom_coords,
    heavy_atom_coords,
    try_base_heavy_atom_coords,
)
from rbe.data.features import min_pairwise_distance
from rbe.data.structure_types import ResidueRecord


@dataclass(frozen=True)
class ContactCutoffs:
    base: float = 4.5
    backbone: float = 5.0


@dataclass(frozen=True)
class ContactLabels:
    pwm_mask: np.ndarray
    A_base_label: np.ndarray
    A_base_mask: np.ndarray
    A_backbone_label: np.ndarray
    A_contact_label: np.ndarray
    site_label: np.ndarray

    @property
    def counts(self) -> dict[str, int]:
        return contact_counts(
            self.A_base_label,
            self.A_backbone_label,
            self.A_contact_label,
            self.site_label,
        )


def compute_contact_labels(
    protein: list[ResidueRecord],
    dna: list[ResidueRecord],
    slot_to_dna_index: np.ndarray,
    cutoffs: ContactCutoffs,
) -> ContactLabels:
    A_base_label = np.zeros((len(protein), len(slot_to_dna_index)), dtype=np.float32)
    A_base_mask = np.zeros_like(A_base_label)
    A_backbone_label = np.zeros_like(A_base_label)
    pwm_mask = (slot_to_dna_index >= 0).astype(np.float32)

    slot_base_coords = [
        try_base_heavy_atom_coords(dna[idx]) if idx >= 0 else None
        for idx in slot_to_dna_index
    ]
    slot_backbone_coords = [
        backbone_heavy_atom_coords(dna[idx]) if idx >= 0 else None
        for idx in slot_to_dna_index
    ]
    for i, residue in enumerate(protein):
        protein_heavy = heavy_atom_coords(residue)
        for j, base_coords in enumerate(slot_base_coords):
            if base_coords is not None:
                A_base_mask[i, j] = 1.0
            if (
                base_coords is not None
                and min_pairwise_distance(protein_heavy, base_coords) <= cutoffs.base
            ):
                A_base_label[i, j] = 1.0
            if (
                slot_backbone_coords[j] is not None
                and min_pairwise_distance(protein_heavy, slot_backbone_coords[j])
                <= cutoffs.backbone
            ):
                A_backbone_label[i, j] = 1.0

    A_contact_label = np.maximum(A_base_label, A_backbone_label).astype(np.float32)
    site_label = A_contact_label.max(axis=1).astype(np.float32)
    return ContactLabels(
        pwm_mask=pwm_mask,
        A_base_label=A_base_label,
        A_base_mask=A_base_mask,
        A_backbone_label=A_backbone_label,
        A_contact_label=A_contact_label,
        site_label=site_label,
    )


def contact_counts(
    A_base_label: np.ndarray,
    A_backbone_label: np.ndarray,
    A_contact_label: np.ndarray,
    site_label: np.ndarray,
) -> dict[str, int]:
    return {
        "A_base_pos": int(A_base_label.sum()),
        "A_backbone_pos": int(A_backbone_label.sum()),
        "A_contact_pos": int(A_contact_label.sum()),
        "site_pos": int(site_label.sum()),
    }
