from __future__ import annotations

from pathlib import Path

from rbe.data.source_manifest import read_source_manifest, write_source_manifest
from scripts.import_deeppbs_source_manifest import import_deeppbs_sources


def test_source_manifest_roundtrip(tmp_path):
    manifest = tmp_path / "samples.tsv"
    manifest.write_text(
        "\t".join(
            [
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
        )
        + "\n"
        + "\t".join(
            [
                "toy_1",
                "train0",
                "1abc",
                "raw/1abc.cif",
                "mmcif",
                "A",
                "B,C",
                "MA0001.1",
                "JASPAR",
                "2026",
                "motifs/MA0001.1.jaspar",
                "example",
            ]
        )
        + "\n"
    )

    samples = read_source_manifest(manifest)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "toy_1"
    assert sample.structure_format == "mmcif"
    assert sample.resolve_structure_path() == tmp_path / "raw/1abc.cif"
    assert sample.resolve_motif_path() == tmp_path / "motifs/MA0001.1.jaspar"

    out = tmp_path / "copy.tsv"
    write_source_manifest(out, samples)
    copied = read_source_manifest(out)
    assert copied[0].sample_id == sample.sample_id
    assert copied[0].motif_version == "2026"


def test_import_deeppbs_sources_writes_public_source_manifest(tmp_path):
    output = tmp_path / "deeppbs_sources.tsv"
    args = type(
        "Args",
        (),
        {
            "fold_file": "valid0.txt",
            "output": str(output),
            "curated_root": "resources/deeppbs_curated",
            "split": "valid0",
            "structure_format": "mmcif",
            "structure_cache_dir": str(tmp_path / "raw" / "structures"),
            "dna_chains": "",
        },
    )()
    import_deeppbs_sources(args)

    samples = read_source_manifest(output, limit=3)
    assert samples[0].split == "valid0"
    assert samples[0].structure_id
    assert samples[0].structure_path.endswith(".cif")
    assert samples[0].motif_source == "DeepPBS-vendored"
    assert Path(samples[0].resolve_motif_path()).exists()
