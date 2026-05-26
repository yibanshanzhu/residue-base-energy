from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from rbe.data.dataset import RBEDataset, rbe_collate, to_device
from rbe.losses import compute_rbe_losses
from rbe.models import build_model_from_config
from rbe.utils import ensure_dir, load_config, resolve_device, set_seed


def _build_dataset(args: argparse.Namespace) -> RBEDataset:
    if args.manifest:
        return RBEDataset.from_manifest(args.manifest)
    if args.data_dir:
        return RBEDataset.from_dir(args.data_dir)
    raise ValueError("Provide either --data-dir or --manifest.")


def _as_float_dict(losses: dict) -> dict:
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.epochs is not None:
        config["optim"]["epochs"] = args.epochs
    if args.device is not None:
        config["device"] = args.device
    set_seed(int(config.get("seed", 7)))

    batch_size = int(config["data"].get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("V1 supports batch_size=1 because N and M are variable.")

    device = resolve_device(config.get("device", "auto"))
    dataset = _build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=rbe_collate,
        num_workers=0,
    )
    model = build_model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optim"]["lr"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )
    out_dir = ensure_dir(args.out_dir)
    best_loss = float("inf")

    metric_keys = [
        "loss",
        "loss_pwm",
        "loss_pwm_teacher",
        "loss_A",
        "loss_A_base",
        "loss_A_backbone",
        "loss_site",
        "loss_sparse",
        "loss_noncontact",
    ]
    print("\t".join(["epoch"] + metric_keys))
    for epoch in range(1, int(config["optim"]["epochs"]) + 1):
        model.train()
        totals = {key: 0.0 for key in metric_keys}
        for batch in loader:
            sample = to_device(batch[0], device)
            motif_len = int(sample["pwm_target"].shape[0])
            outputs = model(
                esm2_repr=sample["esm2_repr"],
                aa_idx=sample["aa_idx"],
                residue_xyz=sample["residue_xyz"],
                edge_index=sample["edge_index"],
                edge_attr=sample["edge_attr"],
                motif_len=motif_len,
            )
            losses = compute_rbe_losses(outputs, sample, config["loss"])
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            for key, value in _as_float_dict(losses).items():
                totals[key] += value

        metrics = {key: value / len(dataset) for key, value in totals.items()}
        print("\t".join([str(epoch)] + [f"{metrics[key]:.6f}" for key in metric_keys]))
        ckpt = {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(ckpt, out_dir / "best.pt")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Residue-Base Energy V1.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--config", default="configs/dna_v1.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    train(build_argparser().parse_args())


if __name__ == "__main__":
    main()
