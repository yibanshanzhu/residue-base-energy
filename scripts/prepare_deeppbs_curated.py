from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rbe.data.deeppbs_curated import (
    DEFAULT_CURATED_ROOT,
    curated_pwm_path,
    download_pdb,
    parse_deeppbs_entry,
    read_entries,
    resolve_fold_file,
    write_pwm,
)
from rbe.data.deeppbs_prepare import (
    DeepPBSCuratedPrepareConfig,
    prepare_deeppbs_curated,
)
from rbe.data.source_prepare import LabelFilterConfig


def prepare(args: argparse.Namespace) -> None:
    result = prepare_deeppbs_curated(_config_from_args(args))
    print(
        f"wrote {result.manifest_path} successes={result.success_count} "
        f"attempts={result.attempt_count} failures={result.failure_count}"
    )
    print(f"wrote {result.sample_table}")
    print(f"wrote {result.failed_table}")


def _config_from_args(args: argparse.Namespace) -> DeepPBSCuratedPrepareConfig:
    return DeepPBSCuratedPrepareConfig(
        curated_root=args.curated_root,
        fold_file=args.fold_file,
        out_root=args.out_root,
        limit=args.limit,
        max_attempts=args.max_attempts,
        label_filter=LabelFilterConfig(
            min_base_pairs=args.min_base_pairs,
            min_contact_pairs=args.min_contact_pairs,
            min_site_residues=args.min_site_residues,
        ),
        device=args.device,
        alignment_score=args.alignment_score,
        allow_multi_char_chain=args.allow_multi_char_chain,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare RBE samples from vendored DeepPBS curated mappings."
    )
    parser.add_argument("--curated-root", default=str(DEFAULT_CURATED_ROOT))
    parser.add_argument("--fold-file", default="valid0.txt")
    parser.add_argument("--out-root", default="data/deeppbs_curated")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--min-base-pairs", type=int, default=0)
    parser.add_argument("--min-contact-pairs", type=int, default=1)
    parser.add_argument("--min-site-residues", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--alignment-score",
        choices=["deeppbs_ic_pcc", "ic_log_likelihood", "log_likelihood"],
        default="deeppbs_ic_pcc",
    )
    parser.add_argument(
        "--allow-multi-char-chain",
        action="store_true",
        help="Allow non-PDB-style multi-character protein chain ids.",
    )
    return parser


def main() -> None:
    prepare(build_argparser().parse_args())


if __name__ == "__main__":
    main()
