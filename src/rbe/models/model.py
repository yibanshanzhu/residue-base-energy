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
    ) -> None:
        super().__init__()
        self.max_motif_len = max_motif_len
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
        self.A_head = nn.Linear(pair_hidden_dim, 1)
        self.E_head = nn.Linear(pair_hidden_dim, 4)

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

        A_logits = self.A_head(z).squeeze(-1)
        A = torch.sigmoid(A_logits)
        E = self.E_head(z)

        pwm_logits = torch.sum(A.unsqueeze(-1) * E, dim=0)
        pwm = torch.softmax(pwm_logits, dim=-1)
        residue_energy = torch.logsumexp(E, dim=-1)
        site_score = torch.max(A * residue_energy, dim=1).values
        site_prob = torch.sigmoid(site_score)
        return {
            "h": h,
            "coord": coord,
            "A_logits": A_logits,
            "A": A,
            "E": E,
            "pwm_logits": pwm_logits,
            "pwm": pwm,
            "site_score": site_score,
            "site_prob": site_prob,
        }


def build_model_from_config(config: dict) -> ResidueBaseEnergyModel:
    model_cfg = config["model"] if "model" in config else config
    return ResidueBaseEnergyModel(**model_cfg)

