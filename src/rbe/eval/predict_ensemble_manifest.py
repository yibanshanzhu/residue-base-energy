from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from rbe.data.dataset import RBEDataset, to_device
from rbe.eval.io import pred_path_for_sample, read_manifest
from rbe.eval.prediction import (
    PREDICTION_KEYS,
    load_model,
    run_model_on_sample,
    sample_metadata_arrays,
)
from rbe.utils import resolve_device


def predict_sample_ensemble(
    sample_path: str | Path,
    pred_path: str | Path,
    models: list[torch.nn.Module],
    device: torch.device,
) -> None:
    sample = to_device(RBEDataset([sample_path])[0], device)
    sums: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for model in models:
            outputs = run_model_on_sample(sample, model)
            for key in PREDICTION_KEYS:
                if key not in outputs:
                    continue
                value = outputs[key].detach().cpu().numpy()
                sums[key] = value if key not in sums else sums[key] + value

    arrays = {
        key: (value / float(len(models))).astype(np.float32)
        for key, value in sums.items()
    }
    arrays.update(sample_metadata_arrays(sample))

    output = Path(pred_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def predict_ensemble_manifest(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoints = [Path(path) for path in args.checkpoints]
    if not checkpoints:
        raise ValueError("At least one checkpoint is required.")

    models = [load_model(path, device) for path in checkpoints]
    samples = read_manifest(args.manifest, limit=args.limit)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for sample_path in samples:
        pred_path = pred_path_for_sample(sample_path, pred_dir, args.pred_suffix)
        if pred_path.exists() and not args.overwrite_pred:
            continue
        predict_sample_ensemble(sample_path, pred_path, models, device)
        print(f"wrote {pred_path}")

    print(
        f"wrote ensemble predictions samples={len(samples)} "
        f"checkpoints={len(checkpoints)} pred_dir={pred_dir}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average predictions from multiple RBE checkpoints over a manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pred-suffix", default=".pred.npz")
    parser.add_argument("--overwrite-pred", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> None:
    predict_ensemble_manifest(build_argparser().parse_args())


if __name__ == "__main__":
    main()
