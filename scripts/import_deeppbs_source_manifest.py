from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rbe.data.deeppbs_curated import (
    DEFAULT_CURATED_ROOT,
    parse_deeppbs_entry,
    read_entries,
    resolve_fold_file,
)
from rbe.data.source_manifest import SourceSample, write_source_manifest


@dataclass(frozen=True)
class MotifIndexEntry:
    motif_source: str
    motif_version: str
    motif_path: Path


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
    motif_index = read_motif_index(args.motif_index)

    samples = []
    for entry in read_entries(fold_file):
        pdb_id, protein_chain, pwm_id = parse_deeppbs_entry(entry)
        sample_id = f"{pdb_id}_{protein_chain}_{pwm_id}"
        structure_path = ""
        if args.structure_cache_dir:
            structure_path = str(
                (Path(args.structure_cache_dir) / f"{pdb_id}{structure_ext}").resolve()
            )
        if pwm_id not in motif_index:
            raise KeyError(f"{pwm_id} not found in motif index {args.motif_index}")
        motif = motif_index[pwm_id]
        motif_path = motif.motif_path.resolve()
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
                motif_source=motif.motif_source,
                motif_version=motif.motif_version,
                motif_path=str(motif_path),
                notes=f"imported_from={fold_file.name};motif_source=untrimmed_motif_index",
                manifest_dir=Path("."),
            )
        )

    write_source_manifest(output, samples)
    print(f"wrote {output} samples={len(samples)} split={split}")


def read_motif_index(path: str | Path) -> dict[str, MotifIndexEntry]:
    index_path = Path(path)
    required = {"motif_id", "motif_source", "motif_version", "motif_path"}
    entries = {}
    with index_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Motif index has no header: {index_path}")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{index_path} is missing columns: {', '.join(missing)}")
        for row in reader:
            motif_id = (row.get("motif_id") or "").strip()
            if not motif_id:
                continue
            motif_path = _resolve_path((row.get("motif_path") or "").strip(), index_path.parent)
            if not motif_path.exists():
                raise FileNotFoundError(f"{motif_id}: motif file not found: {motif_path}")
            entries[motif_id] = MotifIndexEntry(
                motif_source=(row.get("motif_source") or "").strip(),
                motif_version=(row.get("motif_version") or "").strip(),
                motif_path=motif_path,
            )
    if not entries:
        raise ValueError(f"No motif entries found in {index_path}")
    return entries


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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
    parser.add_argument(
        "--motif-index",
        required=True,
        help=(
            "TSV with motif_id, motif_source, motif_version, and motif_path columns. "
            "Every DeepPBS motif id must be present."
        ),
    )
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
