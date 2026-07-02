from __future__ import annotations

import argparse

from rbe.data.alignment_selection import (
    AlignmentContactConstraints,
    AlignmentSelectionConfig,
)
from rbe.data.contact_labels import ContactCutoffs
from rbe.data.processed_sample import (
    ComplexProcessingConfig,
    build_processed_complex_sample,
    format_processed_complex_summary,
    write_processed_complex_sample,
)


def process_complex(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    sample = build_processed_complex_sample(config)
    write_processed_complex_sample(args.output, sample)
    print(format_processed_complex_summary(args.output, sample))


def _config_from_args(args: argparse.Namespace) -> ComplexProcessingConfig:
    return ComplexProcessingConfig(
        structure_path=args.pdb,
        pwm_path=args.pwm,
        protein_chains=args.protein_chains,
        dna_chains=args.dna_chains,
        manual_dna_start_index=args.dna_start_index,
        manual_slot_to_dna_index=args.slot_to_dna_index,
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
        esm_npy=args.esm_npy,
        device=args.device,
        ca_cutoff=args.ca_cutoff,
        num_rbf=args.num_rbf,
        rbf_max_distance=args.rbf_max_distance,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one RBE training npz from a protein-DNA complex and PWM."
    )
    parser.add_argument("--pdb", required=True, help="Protein-DNA complex PDB/mmCIF.")
    parser.add_argument("--pwm", required=True, help="PWM file with A/C/G/T columns.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--protein-chains", default=None, help="Comma-separated protein chains.")
    parser.add_argument("--dna-chains", default=None, help="Comma-separated DNA chains.")
    parser.add_argument(
        "--dna-start-index",
        type=int,
        default=None,
        help="Manual contiguous DNA start index. If omitted, PWM-DNA alignment is automatic.",
    )
    parser.add_argument(
        "--slot-to-dna-index",
        default=None,
        help="Comma-separated selected DNA residue indices, one per PWM row.",
    )
    parser.add_argument(
        "--alignment-score",
        choices=["ic_log_likelihood", "log_likelihood", "deeppbs_ic_pcc"],
        default="ic_log_likelihood",
        help="Score used for automatic PWM-DNA alignment.",
    )
    parser.add_argument(
        "--alignment-contact-policy",
        choices=["require_contact", "sequence_only"],
        default="require_contact",
        help="For automatic alignment, require motif window contact before scoring.",
    )
    parser.add_argument("--alignment-min-base-pairs", type=int, default=0)
    parser.add_argument("--alignment-min-contact-pairs", type=int, default=1)
    parser.add_argument("--alignment-min-site-residues", type=int, default=1)
    parser.add_argument("--esm-npy", default=None, help="Precomputed [N,1280] ESM2 hidden .npy.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ca-cutoff", type=float, default=14.0)
    parser.add_argument(
        "--base-contact-cutoff",
        "--contact-cutoff",
        dest="base_contact_cutoff",
        type=float,
        default=4.5,
        help="Residue heavy atom to DNA base heavy atom cutoff for A_base_label.",
    )
    parser.add_argument(
        "--backbone-contact-cutoff",
        type=float,
        default=5.0,
        help="Residue heavy atom to DNA sugar/phosphate heavy atom cutoff.",
    )
    parser.add_argument(
        "--site-cutoff",
        type=float,
        default=5.0,
        help="Deprecated; site_label is derived from A_contact_label.",
    )
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--rbf-max-distance", type=float, default=20.0)
    return parser


def main() -> None:
    process_complex(build_argparser().parse_args())


if __name__ == "__main__":
    main()
