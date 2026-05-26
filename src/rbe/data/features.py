from __future__ import annotations

import numpy as np


def rbf_encode(
    distances: np.ndarray, num_rbf: int = 16, max_distance: float = 20.0
) -> np.ndarray:
    centers = np.linspace(0.0, max_distance, num_rbf, dtype=np.float32)
    width = float(centers[1] - centers[0]) if num_rbf > 1 else max_distance
    return np.exp(-((distances[:, None] - centers[None, :]) ** 2) / (width**2)).astype(
        np.float32
    )


def build_residue_graph(
    coords: np.ndarray,
    cutoff: float = 14.0,
    num_rbf: int = 16,
    max_distance: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float32)
    n_res = coords.shape[0]
    if n_res == 0:
        raise ValueError("Cannot build residue graph for an empty protein.")

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    src, dst = np.where((dist < cutoff) & (dist > 0.0))
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    if edge_index.shape[1] == 0:
        edge_attr = np.zeros((0, num_rbf + 1), dtype=np.float32)
        return edge_index, edge_attr

    edge_dist = dist[src, dst].astype(np.float32)
    rbf = rbf_encode(edge_dist, num_rbf=num_rbf, max_distance=max_distance)
    seq_sep = np.log1p(np.abs(src - dst).astype(np.float32))
    seq_sep = seq_sep / np.log1p(max(n_res - 1, 1))
    edge_attr = np.concatenate([rbf, seq_sep[:, None].astype(np.float32)], axis=1)
    return edge_index, edge_attr.astype(np.float32)


def min_pairwise_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a[:, None, :] - b[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    return float(np.sqrt(dist2.min()))

