from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE_MANIFEST_COLUMNS = [
    "sample_id",
    "split",
    "structure_id",
    "structure_path",
    "structure_format",
    "protein_chains",
    "dna_chains",
    "motif_id",
    "motif_source",
    "motif_version",
    "motif_path",
    "notes",
]

REQUIRED_SOURCE_COLUMNS = {
    "sample_id",
    "structure_id",
    "structure_format",
    "protein_chains",
    "dna_chains",
    "motif_id",
    "motif_source",
    "motif_version",
    "motif_path",
}

STRUCTURE_FORMAT_EXTENSIONS = {
    "pdb": ".pdb",
    "cif": ".cif",
    "mmcif": ".cif",
}


@dataclass(frozen=True)
class SourceSample:
    sample_id: str
    split: str
    structure_id: str
    structure_path: str
    structure_format: str
    protein_chains: str
    dna_chains: str
    motif_id: str
    motif_source: str
    motif_version: str
    motif_path: str
    notes: str = ""
    manifest_dir: Path = Path(".")

    @classmethod
    def from_row(cls, row: dict[str, str], manifest_dir: str | Path) -> "SourceSample":
        missing = sorted(REQUIRED_SOURCE_COLUMNS - set(row))
        if missing:
            raise ValueError(f"Source manifest is missing columns: {', '.join(missing)}")

        values = {key: (row.get(key, "") or "").strip() for key in SOURCE_MANIFEST_COLUMNS}
        if not values["sample_id"]:
            raise ValueError("source manifest row has empty sample_id")
        if not values["structure_id"]:
            raise ValueError(f"{values['sample_id']}: empty structure_id")
        if not values["motif_id"]:
            raise ValueError(f"{values['sample_id']}: empty motif_id")
        if not values["motif_path"]:
            raise ValueError(f"{values['sample_id']}: empty motif_path")

        structure_format = normalize_structure_format(values["structure_format"])
        return cls(
            sample_id=values["sample_id"],
            split=values["split"],
            structure_id=values["structure_id"],
            structure_path=values["structure_path"],
            structure_format=structure_format,
            protein_chains=values["protein_chains"],
            dna_chains=values["dna_chains"],
            motif_id=values["motif_id"],
            motif_source=values["motif_source"],
            motif_version=values["motif_version"],
            motif_path=values["motif_path"],
            notes=values["notes"],
            manifest_dir=Path(manifest_dir),
        )

    @property
    def structure_extension(self) -> str:
        return STRUCTURE_FORMAT_EXTENSIONS[self.structure_format]

    def resolve_structure_path(self, cache_dir: str | Path | None = None) -> Path:
        if self.structure_path:
            return _resolve_path(self.structure_path, self.manifest_dir)
        if cache_dir is None:
            raise ValueError(
                f"{self.sample_id}: structure_path is empty and no cache_dir was provided"
            )
        return Path(cache_dir) / f"{self.structure_id.lower()}{self.structure_extension}"

    def resolve_motif_path(self) -> Path:
        return _resolve_path(self.motif_path, self.manifest_dir)

    def to_row(self, manifest_dir: str | Path | None = None) -> dict[str, str]:
        row = {key: getattr(self, key) for key in SOURCE_MANIFEST_COLUMNS}
        row.pop("manifest_dir", None)
        if manifest_dir is not None:
            root = Path(manifest_dir)
            if row["structure_path"]:
                row["structure_path"] = _display_path(
                    _resolve_path(row["structure_path"], self.manifest_dir), root
                )
            if row["motif_path"]:
                row["motif_path"] = _display_path(
                    _resolve_path(row["motif_path"], self.manifest_dir), root
                )
        return row


def normalize_structure_format(value: str) -> str:
    normalized = value.strip().lower().lstrip(".")
    if normalized == "pdbx":
        normalized = "cif"
    if normalized not in STRUCTURE_FORMAT_EXTENSIONS:
        raise ValueError(
            f"Unsupported structure_format={value!r}. "
            f"Use one of: {', '.join(sorted(STRUCTURE_FORMAT_EXTENSIONS))}."
        )
    return normalized


def infer_structure_format(path: str | Path, default: str = "mmcif") -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdb":
        return "pdb"
    if suffix in {".cif", ".mmcif"}:
        return "mmcif"
    return normalize_structure_format(default)


def read_source_manifest(
    path: str | Path,
    split: str | None = None,
    limit: int = 0,
) -> list[SourceSample]:
    manifest = Path(path)
    samples = []
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Source manifest has no header: {manifest}")
        missing = sorted(REQUIRED_SOURCE_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{manifest} is missing columns: {', '.join(missing)}")

        for row in reader:
            sample = SourceSample.from_row(row, manifest.parent)
            if split is not None and sample.split != split:
                continue
            samples.append(sample)
            if limit and len(samples) >= limit:
                break

    if not samples:
        split_msg = f" for split={split}" if split is not None else ""
        raise ValueError(f"No source samples found in {manifest}{split_msg}.")
    return samples


def write_source_manifest(
    path: str | Path,
    samples: Iterable[SourceSample],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SOURCE_MANIFEST_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.to_row(manifest_dir=output.parent))


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
