from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rbe.data.shared_cache import prepare_deeppbs_shared_cache


def prepare(args: argparse.Namespace) -> None:
    result = prepare_deeppbs_shared_cache(
        source_root=args.source_root,
        out_root=args.out_root,
        download_structures=args.download_structures,
        overwrite=args.overwrite,
        device=args.device,
    )
    prepared = result.prepare_result
    print(
        f"shared cache unique={result.unique_sample_count} "
        f"successes={prepared.success_count} failures={prepared.failure_count}"
    )
    for split, manifest in result.split_manifests.items():
        count = sum(1 for line in manifest.read_text().splitlines() if line.strip())
        print(f"wrote {manifest} samples={count} split={split}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare each unique DeepPBS sample once, then write per-fold processed manifests."
        )
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--download-structures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    prepare(build_argparser().parse_args())


if __name__ == "__main__":
    main()
