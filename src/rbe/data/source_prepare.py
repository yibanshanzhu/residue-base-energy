from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import urllib.request

import numpy as np

from rbe.data.alignment_selection import (
    AlignmentContactConstraints,
    AlignmentSelectionConfig,
)
from rbe.data.contact_labels import ContactCutoffs
from rbe.data.processed_sample import (
    ComplexProcessingConfig,
    build_processed_complex_sample,
    write_processed_complex_sample,
)
from rbe.data.source_manifest import SourceSample, read_source_manifest


RCSB_DOWNLOAD_URLS = {
    "pdb": "https://files.rcsb.org/download/{structure_id}.pdb",
    "cif": "https://files.rcsb.org/download/{structure_id}.cif",
    "mmcif": "https://files.rcsb.org/download/{structure_id}.cif",
}


@dataclass(frozen=True)
class LabelFilterConfig:
    min_base_pairs: int = 0
    min_contact_pairs: int = 1
    min_site_residues: int = 1


@dataclass(frozen=True)
class SourcePrepareConfig:
    source_manifest: str | Path
    out_root: str | Path
    split: str | None = None
    limit: int = 0
    structure_cache_dir: str | Path | None = None
    download_structures: bool = False
    overwrite: bool = False
    drop_filtered_npz: bool = False
    label_filter: LabelFilterConfig = LabelFilterConfig()
    alignment: AlignmentSelectionConfig = AlignmentSelectionConfig()
    device: str = "cuda"
    ca_cutoff: float = 14.0
    num_rbf: int = 16
    rbf_max_distance: float = 20.0


@dataclass(frozen=True)
class SourcePrepareResult:
    processed_manifest: Path
    sample_table: Path
    failed_table: Path
    success_count: int
    failure_count: int


def prepare_source_manifest(config: SourcePrepareConfig) -> SourcePrepareResult:
    paths = _prepare_paths(config)
    samples = read_source_manifest(
        config.source_manifest, split=config.split, limit=config.limit
    )
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    successes: list[Path] = []
    failures: list[tuple[str, str]] = []
    table_rows = [_sample_table_header()]

    for sample in samples:
        out_npz = paths.processed_dir / f"{sample.sample_id}.npz"
        try:
            structure_path = sample.resolve_structure_path(paths.structure_cache_dir)
            motif_path = sample.resolve_motif_path()
            if config.download_structures:
                download_structure(sample, structure_path)
            _require_input_files(structure_path, motif_path)

            if out_npz.exists() and not config.overwrite:
                counts = label_counts(out_npz)
            else:
                built = build_processed_complex_sample(
                    _complex_config(config, sample, structure_path, motif_path)
                )
                write_processed_complex_sample(out_npz, built)
                counts = built.label_counts

            if not passes_label_filters(counts, config.label_filter):
                reason = _low_contact_reason(counts)
                failures.append((sample.sample_id, reason))
                if config.drop_filtered_npz:
                    out_npz.unlink(missing_ok=True)
                print(f"SKIP {sample.sample_id}: {reason}")
                continue

            successes.append(out_npz.resolve())
            table_rows.append(
                _sample_table_row(sample, structure_path, motif_path, out_npz, counts)
            )
            print(_ok_message(sample.sample_id, counts))
        except Exception as exc:
            failures.append((sample.sample_id, repr(exc)))
            print(f"FAIL {sample.sample_id}: {exc}")

    paths.out_root.mkdir(parents=True, exist_ok=True)
    _write_processed_manifest(paths.processed_manifest, successes)
    _write_sample_table(paths.sample_table, table_rows)
    _write_failed_table(paths.failed_table, failures)

    return SourcePrepareResult(
        processed_manifest=paths.processed_manifest,
        sample_table=paths.sample_table,
        failed_table=paths.failed_table,
        success_count=len(successes),
        failure_count=len(failures),
    )


def download_structure(sample: SourceSample, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = RCSB_DOWNLOAD_URLS[sample.structure_format].format(
        structure_id=sample.structure_id.upper()
    )
    tmp_destination = destination.with_name(f"{destination.name}.tmp")
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            tmp_destination.unlink(missing_ok=True)
            urllib.request.urlretrieve(url, tmp_destination)
            tmp_destination.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            tmp_destination.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(2)
    assert last_error is not None
    raise last_error


def label_counts(npz_path: str | Path) -> dict[str, int]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {
            "A_base_pos": int(data["A_base_label"].sum()),
            "A_backbone_pos": int(data["A_backbone_label"].sum()),
            "A_contact_pos": int(data["A_contact_label"].sum()),
            "site_pos": int(data["site_label"].sum()),
        }


def passes_label_filters(
    counts: dict[str, int], label_filter: LabelFilterConfig
) -> bool:
    return (
        counts["A_contact_pos"] >= label_filter.min_contact_pairs
        and counts["site_pos"] >= label_filter.min_site_residues
        and counts["A_base_pos"] >= label_filter.min_base_pairs
    )


@dataclass(frozen=True)
class _PreparePaths:
    out_root: Path
    structure_cache_dir: Path
    processed_dir: Path
    processed_manifest: Path
    sample_table: Path
    failed_table: Path


def _prepare_paths(config: SourcePrepareConfig) -> _PreparePaths:
    out_root = Path(config.out_root)
    structure_cache_dir = Path(
        config.structure_cache_dir or out_root / "raw" / "structures"
    )
    return _PreparePaths(
        out_root=out_root,
        structure_cache_dir=structure_cache_dir,
        processed_dir=out_root / "processed",
        processed_manifest=out_root / "processed_manifest.txt",
        sample_table=out_root / "sample_table.tsv",
        failed_table=out_root / "failed.tsv",
    )


def _complex_config(
    config: SourcePrepareConfig,
    sample: SourceSample,
    structure_path: Path,
    motif_path: Path,
) -> ComplexProcessingConfig:
    return ComplexProcessingConfig(
        structure_path=structure_path,
        pwm_path=motif_path,
        protein_chains=sample.protein_chains or None,
        dna_chains=sample.dna_chains or None,
        alignment=config.alignment,
        device=config.device,
        ca_cutoff=config.ca_cutoff,
        num_rbf=config.num_rbf,
        rbf_max_distance=config.rbf_max_distance,
    )


def _require_input_files(structure_path: Path, motif_path: Path) -> None:
    if not structure_path.exists():
        raise FileNotFoundError(f"structure file not found: {structure_path}")
    if not motif_path.exists():
        raise FileNotFoundError(f"motif file not found: {motif_path}")


def _sample_table_header() -> str:
    return (
        "sample_id\tsplit\tstructure_id\tstructure_path\tmotif_id\tmotif_source\t"
        "motif_version\tmotif_path\tprotein_chains\tdna_chains\tprocessed_npz\t"
        "A_base_pos\tA_backbone_pos\tA_contact_pos\tsite_pos"
    )


def _sample_table_row(
    sample: SourceSample,
    structure_path: Path,
    motif_path: Path,
    out_npz: Path,
    counts: dict[str, int],
) -> str:
    return "\t".join(
        [
            sample.sample_id,
            sample.split,
            sample.structure_id,
            str(structure_path),
            sample.motif_id,
            sample.motif_source,
            sample.motif_version,
            str(motif_path),
            sample.protein_chains,
            sample.dna_chains,
            str(out_npz.resolve()),
            str(counts["A_base_pos"]),
            str(counts["A_backbone_pos"]),
            str(counts["A_contact_pos"]),
            str(counts["site_pos"]),
        ]
    )


def _low_contact_reason(counts: dict[str, int]) -> str:
    return (
        "filtered_low_contact "
        f"A_base_pos={counts['A_base_pos']} "
        f"A_backbone_pos={counts['A_backbone_pos']} "
        f"A_contact_pos={counts['A_contact_pos']} "
        f"site_pos={counts['site_pos']}"
    )


def _ok_message(sample_id: str, counts: dict[str, int]) -> str:
    return (
        f"OK {sample_id}: "
        f"A_base_pos={counts['A_base_pos']} "
        f"A_backbone_pos={counts['A_backbone_pos']} "
        f"A_contact_pos={counts['A_contact_pos']} "
        f"site_pos={counts['site_pos']}"
    )


def _write_processed_manifest(path: Path, successes: list[Path]) -> None:
    path.write_text("".join(f"{sample_path}\n" for sample_path in successes))


def _write_sample_table(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n")


def _write_failed_table(path: Path, failures: list[tuple[str, str]]) -> None:
    path.write_text(
        "sample_id\treason\n"
        + "\n".join(f"{sample_id}\t{reason}" for sample_id, reason in failures)
        + ("\n" if failures else "")
    )
