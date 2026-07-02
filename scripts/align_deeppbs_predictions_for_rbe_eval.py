from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rbe.eval.deeppbs_alignment import (
    align_deeppbs_predictions,
    complement,
    one_hot_to_seq,
    reverse_complement,
    select_deeppbs_pwm,
)


def align_predictions(args: argparse.Namespace) -> None:
    result = align_deeppbs_predictions(
        manifest=args.manifest,
        deeppbs_pred_dir=args.deeppbs_pred_dir,
        out_dir=args.out_dir,
    )
    print(f"aligned={result.aligned_count} failures={result.failure_count}")
    print(f"wrote {result.aligned_dir}")
    print(f"wrote {result.aligned_manifest}")
    print(f"wrote {result.mode_table}")
    print(f"wrote {result.failure_table}")
    if args.strict and result.failure_count:
        raise SystemExit(1)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align DeepPBS DNA-position predictions to RBE PWM slots so they can "
            "be evaluated with rbe.eval.evaluate_manifest."
        )
    )
    parser.add_argument("--manifest", required=True, help="RBE manifest with target npz files.")
    parser.add_argument(
        "--deeppbs-pred-dir",
        required=True,
        help="Directory containing DeepPBS '*.npz_predict.npz' prediction files.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any manifest sample cannot be aligned.",
    )
    return parser


def main() -> None:
    align_predictions(build_argparser().parse_args())


if __name__ == "__main__":
    main()
