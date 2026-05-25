from __future__ import annotations

import torch
from torch import nn


class EGNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int,
        coord_update_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.coord_update_scale = coord_update_scale
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_attr_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        coord: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_res = h.shape[0]
        if edge_index.numel() == 0:
            agg = torch.zeros_like(h)
            return self.norm(h + self.node_mlp(torch.cat([h, agg], dim=-1))), coord

        src, dst = edge_index[0].long(), edge_index[1].long()
        coord_diff = coord[dst] - coord[src]
        radial = (coord_diff.square().sum(dim=-1, keepdim=True)) / 100.0
        message_in = torch.cat([h[src], h[dst], radial, edge_attr], dim=-1)
        message = self.edge_mlp(message_in)

        coord_gate = self.coord_mlp(message) * self.coord_update_scale
        coord_update = torch.zeros_like(coord)
        coord_update.index_add_(0, dst, coord_diff * coord_gate)
        degree = torch.zeros((n_res, 1), dtype=coord.dtype, device=coord.device)
        degree.index_add_(0, dst, torch.ones_like(coord_gate))
        coord = coord + coord_update / degree.clamp_min(1.0)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, message)
        h = self.norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))
        return h, coord


class EGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int,
        num_layers: int,
        coord_update_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    coord_update_scale=coord_update_scale,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        h: torch.Tensor,
        coord: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h, coord = layer(h, coord, edge_index, edge_attr)
        return h, coord

