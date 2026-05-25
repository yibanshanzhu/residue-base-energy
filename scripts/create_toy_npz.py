from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rbe.data.features import build_residue_graph


def make_sample(path: Path, seed: int, n_res: int, motif_len: int) -> None:
    rng = np.random.default_rng(seed)
    residue_xyz = rng.normal(size=(n_res, 3)).astype(np.float32) * 8.0
    residue_edges, edge_attr = build_residue_graph(residue_xyz)
    residue_aa = np.asarray(list(("ACDEFGHIKLMNPQRSTVWY" * 10)[:n_res]))
    residue_ids = np.asarray([f"A:{i + 1}" for i in range(n_res)])
    esm2_repr = rng.normal(size=(n_res, 1280)).astype(np.float32) * 0.1

    A_label = np.zeros((n_res, motif_len), dtype=np.float32)
    for j in range(motif_len):
        A_label[(j * 3 + seed) % n_res, j] = 1.0
        A_label[(j * 3 + seed + 1) % n_res, j] = 1.0
    site_label = (A_label.max(axis=1) > 0).astype(np.float32)

    logits = rng.normal(size=(motif_len, 4)).astype(np.float32)
    logits[:, 0] += np.linspace(0.5, -0.5, motif_len)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    pwm_target = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
    slot_to_dna_index = np.arange(motif_len, dtype=np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        residue_ids=residue_ids,
        residue_aa=residue_aa,
        residue_xyz=residue_xyz,
        residue_edges=residue_edges,
        edge_attr=edge_attr,
        esm2_repr=esm2_repr,
        pwm_target=pwm_target,
        A_label=A_label,
        site_label=site_label,
        slot_to_dna_index=slot_to_dna_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create toy RBE npz files for smoke tests.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--n-res", type=int, default=12)
    parser.add_argument("--motif-len", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    paths = []
    for idx in range(args.num_samples):
        path = out_dir / f"toy_{idx}.npz"
        make_sample(path, seed=idx + 1, n_res=args.n_res, motif_len=args.motif_len)
        paths.append(path)
    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(path.name for path in paths) + "\n", encoding="utf-8")
    print(f"wrote {len(paths)} samples to {out_dir}")


if __name__ == "__main__":
    main()

