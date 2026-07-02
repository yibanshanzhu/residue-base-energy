from __future__ import annotations

import numpy as np


def extract_esm2_t33_hidden(sequence: str, device: str = "cpu") -> np.ndarray:
    if len(sequence) > 1022:
        raise ValueError(
            "ESM2-t33 has a practical single-sequence length limit around 1022 residues; "
            f"got {len(sequence)}."
        )
    try:
        import torch
        import esm
    except ImportError as exc:
        raise RuntimeError(
            "torch and fair-esm are required for ESM2 hidden extraction. "
            "Install fair-esm with `pip install fair-esm`."
        ) from exc

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("protein", sequence)])
    tokens = tokens.to(device)

    with torch.no_grad():
        result = model(tokens, repr_layers=[33], return_contacts=False)
    hidden = result["representations"][33][0, 1 : len(sequence) + 1]
    return hidden.detach().cpu().numpy().astype(np.float32)
