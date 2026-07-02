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
        "pwm_mask": torch.ones(motif_len),
        "A_label": torch.zeros(n_res, motif_len),
        "A_base_label": torch.zeros(n_res, motif_len),
        "A_base_mask": torch.ones(n_res, motif_len),
        "A_backbone_label": torch.zeros(n_res, motif_len),
        "A_contact_label": torch.zeros(n_res, motif_len),
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
    assert out["A_base"].shape == (n_res, motif_len)
    assert out["A_backbone"].shape == (n_res, motif_len)
    assert out["A_contact"].shape == (n_res, motif_len)
    assert out["E"].shape == (n_res, motif_len, 4)
    assert out["pwm"].shape == (motif_len, 4)
    assert out["site_prob"].shape == (n_res,)
    losses = compute_rbe_losses(
        out,
        sample,
        {
            "lambda_pwm_teacher": 1.0,
            "lambda_A_base": 1.0,
            "lambda_A_backbone": 1.0,
            "lambda_site": 0.5,
            "lambda_sparse": 0.01,
            "lambda_noncontact": 0.05,
        },
    )
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["loss_pwm_teacher"])
    assert torch.isfinite(losses["loss_A_base"])
    assert torch.isfinite(losses["loss_A_backbone"])
    assert torch.isfinite(losses["loss_noncontact"])


def test_A_base_mask_excludes_unknown_positions_from_base_losses():
    outputs = {
        "pwm_logits": torch.zeros(2, 4),
        "A_base": torch.full((2, 2), 0.5),
        "A_base_logits": torch.zeros(2, 2),
        "A_backbone_logits": torch.zeros(2, 2),
        "site_score": torch.zeros(2),
        "A_contact": torch.zeros(2, 2),
        "E": torch.ones(2, 2, 4),
    }
    sample = {
        "pwm_target": torch.full((2, 4), 0.25),
        "pwm_mask": torch.ones(2),
        "A_base_label": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        "A_base_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "A_backbone_label": torch.zeros(2, 2),
        "A_contact_label": torch.zeros(2, 2),
        "site_label": torch.zeros(2),
    }
    weights = {
        "lambda_pwm_teacher": 1.0,
        "lambda_A_base": 1.0,
        "lambda_A_backbone": 1.0,
        "lambda_site": 0.5,
        "lambda_sparse": 0.01,
        "lambda_noncontact": 0.05,
    }

    base_losses = compute_rbe_losses(outputs, sample, weights)
    sample_changed = {**sample, "A_base_label": sample["A_base_label"].clone()}
    sample_changed["A_base_label"][0, 1] = 1.0
    changed_losses = compute_rbe_losses(outputs, sample_changed, weights)

    assert torch.allclose(base_losses["loss_A_base"], changed_losses["loss_A_base"])
    assert torch.allclose(
        base_losses["loss_noncontact"], changed_losses["loss_noncontact"]
    )


def test_pwm_mask_excludes_unobserved_columns_from_structure_losses():
    outputs = {
        "pwm_logits": torch.zeros(2, 4),
        "A_base": torch.full((2, 2), 0.5),
        "A_base_logits": torch.zeros(2, 2),
        "A_backbone_logits": torch.zeros(2, 2),
        "site_score": torch.zeros(2),
        "A_contact": torch.tensor([[0.2, 0.9], [0.2, 0.9]]),
        "E": torch.ones(2, 2, 4),
    }
    sample = {
        "pwm_target": torch.full((2, 4), 0.25),
        "pwm_mask": torch.tensor([1.0, 0.0]),
        "A_base_label": torch.zeros(2, 2),
        "A_base_mask": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "A_backbone_label": torch.zeros(2, 2),
        "A_contact_label": torch.zeros(2, 2),
        "site_label": torch.zeros(2),
    }
    weights = {
        "lambda_pwm_teacher": 1.0,
        "lambda_A_base": 1.0,
        "lambda_A_backbone": 1.0,
        "lambda_site": 0.5,
        "lambda_sparse": 0.01,
        "lambda_noncontact": 0.05,
    }

    base_losses = compute_rbe_losses(outputs, sample, weights)
    changed = {**sample, "A_backbone_label": sample["A_backbone_label"].clone()}
    changed["A_backbone_label"][:, 1] = 1.0
    changed_losses = compute_rbe_losses(outputs, changed, weights)

    assert torch.allclose(
        base_losses["loss_A_backbone"], changed_losses["loss_A_backbone"]
    )
    assert torch.allclose(base_losses["loss_sparse"], torch.tensor(0.2))
