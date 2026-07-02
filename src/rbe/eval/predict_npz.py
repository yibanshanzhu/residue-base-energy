from __future__ import annotations

import argparse

from rbe.eval.prediction import load_model, predict_sample_npz
from rbe.utils import resolve_device


def predict_npz(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    predict_sample_npz(args.sample, args.output, model, device)
    print(f"wrote {args.output} from checkpoint={args.checkpoint} sample={args.sample}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a trained checkpoint on one processed RBE npz sample."
    )
    parser.add_argument("--sample", required=True, help="Processed target .npz sample.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    predict_npz(build_argparser().parse_args())


if __name__ == "__main__":
    main()
