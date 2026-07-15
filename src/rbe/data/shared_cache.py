from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from rbe.data.source_manifest import (
    SourceSample,
    read_source_manifest,
    write_source_manifest,
)
from rbe.data.source_prepare import (
    SourcePrepareConfig,
    SourcePrepareResult,
    prepare_source_manifest,
)


DEEPPBS_SPLITS = tuple(
    [f"train{fold}" for fold in range(5)]
    + [f"valid{fold}" for fold in range(5)]
    + ["id"]
)


@dataclass(frozen=True)
class SharedSourceCollection:
    samples: tuple[SourceSample, ...]
    sample_ids_by_split: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class SharedCacheResult:
    prepare_result: SourcePrepareResult
    split_manifests: dict[str, Path]
    unique_sample_count: int


def prepare_deeppbs_shared_cache(
    source_root: str | Path,
    out_root: str | Path,
    *,
    download_structures: bool = False,
    overwrite: bool = False,
    device: str = "cuda",
) -> SharedCacheResult:
    source_root = Path(source_root)
    out_root = Path(out_root)
    source_manifests = {
        split: source_root / f"deeppbs_{split}_sources.tsv"
        for split in DEEPPBS_SPLITS
    }
    collection = collect_shared_sources(source_manifests)

    out_root.mkdir(parents=True, exist_ok=True)
    union_manifest = out_root / "source_manifest.tsv"
    write_source_manifest(union_manifest, collection.samples)

    prepare_result = prepare_source_manifest(
        SourcePrepareConfig(
            source_manifest=union_manifest,
            out_root=out_root,
            download_structures=download_structures,
            overwrite=overwrite,
            device=device,
        )
    )
    successful_paths = _read_processed_paths(prepare_result.processed_manifest)
    split_manifests = write_split_processed_manifests(
        collection.sample_ids_by_split,
        successful_paths,
        out_root / "manifests",
    )
    return SharedCacheResult(
        prepare_result=prepare_result,
        split_manifests=split_manifests,
        unique_sample_count=len(collection.samples),
    )


def collect_shared_sources(
    source_manifests: dict[str, str | Path],
) -> SharedSourceCollection:
    unique_samples: dict[str, SourceSample] = {}
    sample_ids_by_split: dict[str, tuple[str, ...]] = {}

    for split, manifest in source_manifests.items():
        samples = read_source_manifest(manifest)
        ids = []
        seen_in_split = set()
        for sample in samples:
            if sample.sample_id in seen_in_split:
                raise ValueError(f"{manifest}: duplicate sample_id {sample.sample_id}")
            seen_in_split.add(sample.sample_id)
            ids.append(sample.sample_id)

            existing = unique_samples.get(sample.sample_id)
            if existing is None:
                unique_samples[sample.sample_id] = replace(sample, split="shared")
            elif _sample_identity(existing) != _sample_identity(sample):
                raise ValueError(
                    f"Conflicting source definitions for sample_id {sample.sample_id}"
                )
        sample_ids_by_split[split] = tuple(ids)

    return SharedSourceCollection(
        samples=tuple(unique_samples.values()),
        sample_ids_by_split=sample_ids_by_split,
    )


def write_split_processed_manifests(
    sample_ids_by_split: dict[str, tuple[str, ...]],
    successful_paths: list[Path],
    output_dir: str | Path,
) -> dict[str, Path]:
    path_by_sample_id = {path.stem: path.resolve() for path in successful_paths}
    if len(path_by_sample_id) != len(successful_paths):
        raise ValueError("Processed cache contains duplicate sample ids.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split, sample_ids in sample_ids_by_split.items():
        output = output_dir / f"{split}.txt"
        paths = [
            path_by_sample_id[sample_id]
            for sample_id in sample_ids
            if sample_id in path_by_sample_id
        ]
        output.write_text("".join(f"{path}\n" for path in paths))
        manifests[split] = output
    return manifests


def _sample_identity(sample: SourceSample) -> tuple[str, ...]:
    structure_path = (
        str(sample.resolve_structure_path().resolve()) if sample.structure_path else ""
    )
    return (
        sample.structure_id,
        structure_path,
        sample.structure_format,
        sample.protein_chains,
        sample.dna_chains,
        sample.motif_id,
        sample.motif_source,
        sample.motif_version,
        str(sample.resolve_motif_path().resolve()),
    )


def _read_processed_paths(manifest: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in manifest.read_text().splitlines()
        if line.strip()
    ]
