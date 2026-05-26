from __future__ import annotations

import torch

from rbe.losses import compute_rbe_losses
from rbe.models.model import ResidueBaseEnergyModel


def test_forward_and_loss_shapes():
    n_res, motif_len = 6, 4
    model = ResidueBaseEnergyModel(
        esm_dim=1280,
        hidden_dim=32,
        num_egnn_layers=2,
        edge_attr_dim=17,
        pair_hidden_dim=32,
        max_motif_len=8,
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 2], [1, 2, 3, 4, 5, 0, 0, 1]], dtype=torch.long
    )
    sample = {
        "esm2_repr": torch.randn(n_res, 1280),
        "aa_idx": torch.arange(n_res) % 20,
        "residue_xyz": torch.randn(n_res, 3),
        "edge_index": edge_index,
        "edge_attr": torch.randn(edge_index.shape[1], 17),
        "pwm_target": torch.softmax(torch.randn(motif_len, 4), dim=-1),
        "A_label": torch.zeros(n_res, motif_len),
        "site_label": torch.zeros(n_res),
    }
    out = model(
        sample["esm2_repr"],
        sample["aa_idx"],
        sample["residue_xyz"],
        sample["edge_index"],
        sample["edge_attr"],
        motif_len=motif_len,
    )
    assert out["A"].shape == (n_res, motif_len)
    assert out["E"].shape == (n_res, motif_len, 4)
    assert out["pwm"].shape == (motif_len, 4)
    assert out["site_prob"].shape == (n_res,)
    losses = compute_rbe_losses(
        out,
        sample,
        {
            "lambda_pwm_teacher": 1.0,
            "lambda_A": 1.0,
            "lambda_site": 0.5,
            "lambda_sparse": 0.01,
            "lambda_noncontact": 0.05,
        },
    )
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["loss_pwm_teacher"])
    assert torch.isfinite(losses["loss_noncontact"])
