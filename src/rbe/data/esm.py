from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=2)
def _load_esm2_t33(device: str):
    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    return model.eval().to(device), alphabet


def extract_esm2_t33_hidden(sequence: str, device: str = "cpu") -> np.ndarray:
    if len(sequence) > 1022:
        raise ValueError(
            "ESM2-t33 has a practical single-sequence length limit around 1022 residues; "
            f"got {len(sequence)}."
        )
    import torch

    model, alphabet = _load_esm2_t33(str(device))
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("protein", sequence)])
    tokens = tokens.to(device)

    with torch.no_grad():
        result = model(tokens, repr_layers=[33], return_contacts=False)
    hidden = result["representations"][33][0, 1 : len(sequence) + 1]
    return hidden.detach().cpu().numpy().astype(np.float32)
