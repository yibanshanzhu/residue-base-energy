from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rbe.data.deeppbs_curated import (
    DEFAULT_CURATED_ROOT,
    curated_pwm_path,
    parse_deeppbs_entry,
    read_entries,
    resolve_fold_file,
)
from rbe.data.source_manifest import SourceSample, write_source_manifest


def infer_split_name(fold_file: Path) -> str:
    stem = fold_file.stem
    if stem.startswith("train"):
        return stem
    if stem.startswith("valid"):
        return stem
    if stem == "id":
        return "id"
    return stem


def import_deeppbs_sources(args: argparse.Namespace) -> None:
    curated_root = Path(args.curated_root).resolve()
    fold_file = resolve_fold_file(args.fold_file, curated_root)
    output = Path(args.output)
    split = args.split or infer_split_name(fold_file)
    structure_format = args.structure_format
    structure_ext = ".pdb" if structure_format == "pdb" else ".cif"

    samples = []
    for entry in read_entries(fold_file):
        pdb_id, protein_chain, pwm_id = parse_deeppbs_entry(entry)
        sample_id = f"{pdb_id}_{protein_chain}_{pwm_id}"
        structure_path = ""
        if args.structure_cache_dir:
            structure_path = str(
                (Path(args.structure_cache_dir) / f"{pdb_id}{structure_ext}").resolve()
            )
        motif_path = str(curated_pwm_path(curated_root, pwm_id).resolve())
        samples.append(
            SourceSample(
                sample_id=sample_id,
                split=split,
                structure_id=pdb_id,
                structure_path=structure_path,
                structure_format=structure_format,
                protein_chains=protein_chain,
                dna_chains=args.dna_chains,
                motif_id=pwm_id,
                motif_source="DeepPBS-vendored",
                motif_version="pwms.pickle-trimmed-ic>0.5",
                motif_path=motif_path,
                notes=f"imported_from={fold_file.name}",
                manifest_dir=Path("."),
            )
        )

    write_source_manifest(output, samples)
    print(f"wrote {output} samples={len(samples)} split={split}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert vendored DeepPBS fold entries into RBE source manifest rows. "
            "The generated manifest names PDB/mmCIF structures and motif files."
        )
    )
    parser.add_argument("--fold-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--curated-root", default=str(DEFAULT_CURATED_ROOT))
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--structure-format",
        choices=["pdb", "mmcif"],
        default="mmcif",
        help="Public structure format to use when preparing this manifest.",
    )
    parser.add_argument(
        "--structure-cache-dir",
        default="data/raw/structures",
        help=(
            "Optional structure cache path written into the manifest. "
            "Use an empty string to leave structure_path blank and let prepare "
            "choose its cache directory."
        ),
    )
    parser.add_argument(
        "--dna-chains",
        default="",
        help="DNA chains to use. Empty means all DNA chains in the structure.",
    )
    return parser


def main() -> None:
    import_deeppbs_sources(build_argparser().parse_args())


if __name__ == "__main__":
    main()
