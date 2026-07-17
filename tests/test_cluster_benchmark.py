from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rbe.data.cluster_benchmark import (
    ClusterSample,
    build_cluster_benchmark,
    collect_cluster_samples,
)


def _sample(
    tmp_path: Path,
    sample_id: str,
    sequence_id: str,
    pwm_hash: str,
) -> ClusterSample:
    path = tmp_path / f"{sample_id}.npz"
    path.touch()
    return ClusterSample(
        sample_id=sample_id,
        path=path,
        motif_id=f"motif_{pwm_hash}",
        sequence_id=sequence_id,
        sequence="ACDE",
        pwm_hash=pwm_hash,
        pwm_len=8,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_cluster_benchmark_keeps_homology_and_pwm_components_disjoint(tmp_path: Path):
    samples = [
        _sample(tmp_path, "a1", "seq1", "pwm1"),
        _sample(tmp_path, "a2", "seq2", "pwm2"),
        _sample(tmp_path, "a3", "seq3", "pwm2"),
        _sample(tmp_path, "b", "seq4", "pwm3"),
        _sample(tmp_path, "c", "seq5", "pwm4"),
        _sample(tmp_path, "d", "seq6", "pwm5"),
    ]
    sequence_clusters = {
        "seq1": "cluster_a",
        "seq2": "cluster_a",
        "seq3": "cluster_b",
        "seq4": "cluster_c",
        "seq5": "cluster_d",
        "seq6": "cluster_e",
    }

    result = build_cluster_benchmark(
        samples,
        sequence_clusters,
        tmp_path / "benchmark",
        n_folds=3,
    )

    rows = {row["sample_id"]: row for row in _read_rows(result.sample_table)}
    assert rows["a1"]["component"] == rows["a2"]["component"]
    assert rows["a2"]["component"] == rows["a3"]["component"]
    assert rows["a1"]["fold"] == rows["a3"]["fold"]
    assert result.component_count == 4

    for fold in range(3):
        partitions = {}
        for name in ("train", "valid", "test"):
            manifest = result.fold_dir / f"fold{fold}_{name}.txt"
            sample_ids = {Path(line).stem for line in manifest.read_text().splitlines()}
            partitions[name] = {rows[sample_id]["component"] for sample_id in sample_ids}
        assert partitions["train"].isdisjoint(partitions["valid"])
        assert partitions["train"].isdisjoint(partitions["test"])
        assert partitions["valid"].isdisjoint(partitions["test"])


def test_collect_cluster_samples_requires_canonical_contract(tmp_path: Path):
    cache = tmp_path / "cache/processed"
    source = tmp_path / "source"
    cache.mkdir(parents=True)
    source.mkdir()
    np.savez_compressed(
        cache / "sample.npz",
        pwm_orientation=np.asarray("canonical"),
        residue_aa=np.asarray(["A", "C", "D"]),
        pwm_target=np.asarray([[0.7, 0.1, 0.1, 0.1]], dtype=np.float32),
    )
    (source / "deeppbs_train0_sources.tsv").write_text(
        "sample_id\tmotif_id\nsample\tM1\n"
    )

    samples = collect_cluster_samples(tmp_path / "cache", source)

    assert len(samples) == 1
    assert samples[0].sequence == "ACD"
    assert samples[0].motif_id == "M1"
    assert samples[0].pwm_len == 1
