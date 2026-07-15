from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import rbe.data.shared_cache as shared_cache
from rbe.data.shared_cache import (
    DEEPPBS_SPLITS,
    collect_shared_sources,
    prepare_deeppbs_shared_cache,
    write_split_processed_manifests,
)
from rbe.data.source_manifest import SourceSample, write_source_manifest
from rbe.data.source_prepare import SourcePrepareResult


def _sample(tmp_path: Path, sample_id: str, split: str) -> SourceSample:
    return SourceSample(
        sample_id=sample_id,
        split=split,
        structure_id="1abc",
        structure_path=str(tmp_path / "1abc.cif"),
        structure_format="mmcif",
        protein_chains="A",
        dna_chains="B,C",
        motif_id="MA0001.1",
        motif_source="JASPAR",
        motif_version="MA0001.1-untrimmed",
        motif_path=str(tmp_path / "MA0001.1.txt"),
    )


def test_shared_sources_deduplicate_cache_but_keep_fold_membership(tmp_path):
    shared = _sample(tmp_path, "shared", "train0")
    train_only = _sample(tmp_path, "train_only", "train0")
    valid_only = _sample(tmp_path, "valid_only", "valid0")
    train_manifest = tmp_path / "train0.tsv"
    valid_manifest = tmp_path / "valid0.tsv"
    write_source_manifest(train_manifest, [shared, train_only])
    write_source_manifest(valid_manifest, [replace(shared, split="valid0"), valid_only])

    collection = collect_shared_sources(
        {"train0": train_manifest, "valid0": valid_manifest}
    )

    assert [sample.sample_id for sample in collection.samples] == [
        "shared",
        "train_only",
        "valid_only",
    ]
    assert collection.sample_ids_by_split["train0"] == ("shared", "train_only")
    assert collection.sample_ids_by_split["valid0"] == ("shared", "valid_only")


def test_shared_sources_reject_conflicting_duplicate_definitions(tmp_path):
    sample = _sample(tmp_path, "shared", "train0")
    train_manifest = tmp_path / "train0.tsv"
    valid_manifest = tmp_path / "valid0.tsv"
    write_source_manifest(train_manifest, [sample])
    write_source_manifest(
        valid_manifest,
        [replace(sample, split="valid0", protein_chains="Z")],
    )

    with pytest.raises(ValueError, match="Conflicting source definitions"):
        collect_shared_sources({"train0": train_manifest, "valid0": valid_manifest})


def test_split_manifests_only_reference_successful_shared_cache(tmp_path):
    shared = tmp_path / "processed" / "shared.npz"
    train_only = tmp_path / "processed" / "train_only.npz"
    shared.parent.mkdir()
    shared.touch()
    train_only.touch()

    manifests = write_split_processed_manifests(
        {
            "train0": ("shared", "train_only"),
            "valid0": ("shared", "failed"),
        },
        [shared, train_only],
        tmp_path / "manifests",
    )

    assert manifests["train0"].read_text().splitlines() == [
        str(shared.resolve()),
        str(train_only.resolve()),
    ]
    assert manifests["valid0"].read_text().splitlines() == [str(shared.resolve())]


def test_prepare_shared_cache_writes_all_deeppbs_fold_manifests(tmp_path, monkeypatch):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    for split in DEEPPBS_SPLITS:
        write_source_manifest(
            source_root / f"deeppbs_{split}_sources.tsv",
            [_sample(tmp_path, "shared", split)],
        )

    def fake_prepare(config):
        out_root = Path(config.out_root)
        processed = out_root / "processed" / "shared.npz"
        processed.parent.mkdir(parents=True)
        processed.touch()
        processed_manifest = out_root / "processed_manifest.txt"
        processed_manifest.write_text(f"{processed.resolve()}\n")
        return SourcePrepareResult(
            processed_manifest=processed_manifest,
            sample_table=out_root / "sample_table.tsv",
            failed_table=out_root / "failed.tsv",
            success_count=1,
            failure_count=0,
        )

    monkeypatch.setattr(shared_cache, "prepare_source_manifest", fake_prepare)
    result = prepare_deeppbs_shared_cache(source_root, tmp_path / "cache")

    assert result.unique_sample_count == 1
    assert set(result.split_manifests) == set(DEEPPBS_SPLITS)
    assert all(
        len(manifest.read_text().splitlines()) == 1
        for manifest in result.split_manifests.values()
    )
