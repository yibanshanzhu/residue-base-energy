from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rbe.models.egnn import EGNN


class ResidueBaseEnergyModel(nn.Module):
    def __init__(
        self,
        esm_dim: int = 1280,
        hidden_dim: int = 256,
        num_egnn_layers: int = 4,
        edge_attr_dim: int = 17,
        pair_hidden_dim: int = 256,
        max_motif_len: int = 64,
        coord_update_scale: float = 0.1,
        use_pwm_prior: bool = False,
    ) -> None:
        super().__init__()
        self.max_motif_len = max_motif_len
        self.use_pwm_prior = use_pwm_prior
        node_in_dim = esm_dim + 20 + 1
        self.input_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.egnn = EGNN(
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim,
            num_layers=num_egnn_layers,
            coord_update_scale=coord_update_scale,
        )
        self.slot_embedding = nn.Embedding(max_motif_len, hidden_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, pair_hidden_dim),
            nn.SiLU(),
            nn.Linear(pair_hidden_dim, pair_hidden_dim),
            nn.SiLU(),
        )
        self.A_base_head = nn.Linear(pair_hidden_dim, 1)
        self.A_backbone_head = nn.Linear(pair_hidden_dim, 1)
        self.E_head = nn.Linear(pair_hidden_dim, 4)
        self.register_buffer(
            "_pwm_prior_logits",
            torch.empty((0, 4)),
            persistent=False,
        )
        if use_pwm_prior:
            nn.init.zeros_(self.E_head.weight)
            nn.init.zeros_(self.E_head.bias)

    def set_pwm_prior(self, pwm: torch.Tensor) -> None:
        pwm = torch.as_tensor(
            pwm,
            dtype=self.slot_embedding.weight.dtype,
            device=self.slot_embedding.weight.device,
        )
        if pwm.ndim != 2 or pwm.shape[1] != 4:
            raise ValueError(f"PWM prior must have shape [M, 4], got {tuple(pwm.shape)}.")
        if pwm.shape[0] > self.max_motif_len:
            raise ValueError(
                f"PWM prior length {pwm.shape[0]} exceeds max_motif_len="
                f"{self.max_motif_len}."
            )
        pwm = pwm.clamp_min(1e-8)
        pwm = pwm / pwm.sum(dim=-1, keepdim=True)
        self._pwm_prior_logits = pwm.log()

    def pwm_prior(self) -> torch.Tensor:
        if not self.use_pwm_prior or self._pwm_prior_logits.numel() == 0:
            raise ValueError("This model has no configured family PWM prior.")
        return torch.softmax(self._pwm_prior_logits, dim=-1)

    def forward(
        self,
        esm2_repr: torch.Tensor,
        aa_idx: torch.Tensor,
        residue_xyz: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        motif_len: int,
    ) -> dict:
        if motif_len > self.max_motif_len:
            raise ValueError(
                f"motif_len={motif_len} exceeds max_motif_len={self.max_motif_len}."
            )
        coord = residue_xyz - residue_xyz.mean(dim=0, keepdim=True)
        radius = coord.norm(dim=-1, keepdim=True) / 20.0
        aa_onehot = F.one_hot(aa_idx.long(), num_classes=20).float()
        node_input = torch.cat([esm2_repr.float(), aa_onehot, radius], dim=-1)

        h = self.input_proj(node_input)
        h, coord = self.egnn(h, coord, edge_index, edge_attr.float())

        slot_ids = torch.arange(motif_len, device=h.device)
        slot = self.slot_embedding(slot_ids)
        if motif_len == 1:
            slot_pos = torch.zeros((1, 1), dtype=h.dtype, device=h.device)
        else:
            slot_pos = torch.linspace(0.0, 1.0, motif_len, device=h.device)[:, None]

        n_res = h.shape[0]
        h_i = h[:, None, :].expand(n_res, motif_len, -1)
        s_j = slot[None, :, :].expand(n_res, motif_len, -1)
        pos_j = slot_pos[None, :, :].expand(n_res, motif_len, -1)
        pair_input = torch.cat([h_i, s_j, h_i * s_j, pos_j], dim=-1)
        z = self.pair_mlp(pair_input)

        A_base_logits = self.A_base_head(z).squeeze(-1)
        A_backbone_logits = self.A_backbone_head(z).squeeze(-1)
        A_base = torch.sigmoid(A_base_logits)
        A_backbone = torch.sigmoid(A_backbone_logits)
        A_contact = torch.maximum(A_base, A_backbone)
        A_contact_logits = torch.maximum(A_base_logits, A_backbone_logits)
        E = self.E_head(z)

        pwm_residual_logits = torch.sum(A_base.unsqueeze(-1) * E, dim=0)
        if self.use_pwm_prior:
            if self._pwm_prior_logits.shape != (motif_len, 4):
                raise ValueError(
                    "Family PWM prior is unset or does not match motif_len: "
                    f"prior={tuple(self._pwm_prior_logits.shape)}, motif_len={motif_len}."
                )
            pwm_prior_logits = self._pwm_prior_logits
        else:
            pwm_prior_logits = torch.zeros_like(pwm_residual_logits)
        pwm_logits = pwm_prior_logits + pwm_residual_logits
        pwm = torch.softmax(pwm_logits, dim=-1)
        site_score = torch.max(A_contact_logits, dim=1).values
        site_prob = torch.sigmoid(site_score)
        return {
            "h": h,
            "coord": coord,
            "A_logits": A_base_logits,
            "A": A_base,
            "A_base_logits": A_base_logits,
            "A_base": A_base,
            "A_backbone_logits": A_backbone_logits,
            "A_backbone": A_backbone,
            "A_contact_logits": A_contact_logits,
            "A_contact": A_contact,
            "E": E,
            "pwm_prior_logits": pwm_prior_logits,
            "pwm_residual_logits": pwm_residual_logits,
            "pwm_logits": pwm_logits,
            "pwm": pwm,
            "site_score": site_score,
            "site_prob": site_prob,
        }


def build_model_from_config(config: dict) -> ResidueBaseEnergyModel:
    model_cfg = config["model"] if "model" in config else config
    return ResidueBaseEnergyModel(**model_cfg)
