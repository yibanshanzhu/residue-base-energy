from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rbe.data.alignment_selection import (
    AlignmentContactConstraints,
    AlignmentSelectionConfig,
)
from rbe.data.contact_labels import ContactCutoffs
from rbe.data.source_prepare import (
    LabelFilterConfig,
    SourcePrepareConfig,
    prepare_source_manifest,
)


def prepare(args: argparse.Namespace) -> None:
    result = prepare_source_manifest(_config_from_args(args))
    print(
        f"wrote {result.processed_manifest} successes={result.success_count} "
        f"failures={result.failure_count}"
    )
    print(f"wrote {result.sample_table}")
    print(f"wrote {result.failed_table}")


def _config_from_args(args: argparse.Namespace) -> SourcePrepareConfig:
    return SourcePrepareConfig(
        source_manifest=args.source_manifest,
        out_root=args.out_root,
        split=args.split,
        limit=args.limit,
        structure_cache_dir=args.structure_cache_dir,
        download_structures=args.download_structures,
        overwrite=args.overwrite,
        drop_filtered_npz=args.drop_filtered_npz,
        label_filter=LabelFilterConfig(
            min_base_pairs=args.min_base_pairs,
            min_contact_pairs=args.min_contact_pairs,
            min_site_residues=args.min_site_residues,
        ),
        alignment=AlignmentSelectionConfig(
            score_mode=args.alignment_score,
            contact_policy=args.alignment_contact_policy,
            contact_cutoffs=ContactCutoffs(
                base=args.base_contact_cutoff,
                backbone=args.backbone_contact_cutoff,
            ),
            contact_constraints=AlignmentContactConstraints(
                min_base_pairs=args.alignment_min_base_pairs,
                min_contact_pairs=args.alignment_min_contact_pairs,
                min_site_residues=args.alignment_min_site_residues,
            ),
        ),
        device=args.device,
        ca_cutoff=args.ca_cutoff,
        num_rbf=args.num_rbf,
        rbf_max_distance=args.rbf_max_distance,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare RBE training cache from source PDB/mmCIF + motif database manifest."
        )
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--structure-cache-dir", default=None)
    parser.add_argument("--download-structures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--drop-filtered-npz", action="store_true")
    parser.add_argument("--min-base-pairs", type=int, default=0)
    parser.add_argument("--min-contact-pairs", type=int, default=1)
    parser.add_argument("--min-site-residues", type=int, default=1)
    parser.add_argument("--alignment-min-base-pairs", type=int, default=0)
    parser.add_argument("--alignment-min-contact-pairs", type=int, default=1)
    parser.add_argument("--alignment-min-site-residues", type=int, default=1)
    parser.add_argument(
        "--alignment-score",
        choices=["deeppbs_ic_pcc", "ic_log_likelihood", "log_likelihood"],
        default="deeppbs_ic_pcc",
    )
    parser.add_argument(
        "--alignment-contact-policy",
        choices=["require_contact", "sequence_only"],
        default="require_contact",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ca-cutoff", type=float, default=14.0)
    parser.add_argument("--base-contact-cutoff", type=float, default=4.5)
    parser.add_argument("--backbone-contact-cutoff", type=float, default=5.0)
    parser.add_argument("--site-cutoff", type=float, default=5.0)
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--rbf-max-distance", type=float, default=20.0)
    return parser


def main() -> None:
    prepare(build_argparser().parse_args())


if __name__ == "__main__":
    main()
