from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClusterSample:
    sample_id: str
    path: Path
    motif_id: str
    sequence_id: str
    sequence: str
    pwm_hash: str
    pwm_len: int


@dataclass(frozen=True)
class ClusterBenchmarkResult:
    sample_table: Path
    component_table: Path
    fold_table: Path
    fold_dir: Path
    sample_count: int
    sequence_count: int
    sequence_cluster_count: int
    component_count: int


def prepare_cluster_benchmark(
    cache_root: str | Path,
    source_root: str | Path,
    out_root: str | Path,
    *,
    mmseqs: str | Path,
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
    n_folds: int = 5,
    threads: int = 8,
) -> ClusterBenchmarkResult:
    samples = collect_cluster_samples(cache_root, source_root)
    sequence_by_id = {sample.sequence_id: sample.sequence for sample in samples}
    sequence_clusters = cluster_sequences_mmseqs(
        sequence_by_id,
        mmseqs=mmseqs,
        min_seq_id=min_seq_id,
        coverage=coverage,
        threads=threads,
    )
    result = build_cluster_benchmark(
        samples,
        sequence_clusters,
        out_root,
        n_folds=n_folds,
    )
    protocol = {
        "mmseqs": str(Path(mmseqs).resolve()),
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "coverage_mode": 0,
        "cluster_mode": 1,
        "folds": n_folds,
        "samples": result.sample_count,
        "exact_sequences": result.sequence_count,
        "sequence_clusters": result.sequence_cluster_count,
        "components": result.component_count,
    }
    (Path(out_root) / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    return result


def collect_cluster_samples(
    cache_root: str | Path,
    source_root: str | Path,
) -> list[ClusterSample]:
    motif_by_sample = _read_motif_ids(source_root)
    processed = sorted((Path(cache_root) / "processed").glob("*.npz"))
    if not processed:
        raise ValueError(f"No processed samples found under {cache_root}.")

    samples = []
    for path in processed:
        sample_id = path.stem
        if sample_id not in motif_by_sample:
            raise ValueError(f"{path}: sample is absent from source manifests.")
        with np.load(path, allow_pickle=False) as data:
            if str(data.get("pwm_orientation", "")) != "canonical":
                raise ValueError(f"{path}: expected canonical pwm_orientation.")
            sequence = "".join(data["residue_aa"].astype(str).tolist())
            pwm = np.asarray(data["pwm_target"], dtype=np.float64)
        sequence_id = "seq_" + _digest(sequence.encode())
        pwm_payload = np.asarray(np.round(pwm, 6), dtype="<f8").tobytes()
        pwm_hash = "pwm_" + _digest(str(pwm.shape).encode() + pwm_payload)
        samples.append(
            ClusterSample(
                sample_id=sample_id,
                path=path.resolve(),
                motif_id=motif_by_sample[sample_id],
                sequence_id=sequence_id,
                sequence=sequence,
                pwm_hash=pwm_hash,
                pwm_len=int(pwm.shape[0]),
            )
        )
    return samples


def cluster_sequences_mmseqs(
    sequence_by_id: dict[str, str],
    *,
    mmseqs: str | Path,
    min_seq_id: float,
    coverage: float,
    threads: int,
) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="rbe-mmseqs-") as temporary:
        root = Path(temporary)
        fasta = root / "sequences.fasta"
        fasta.write_text(
            "".join(
                f">{sequence_id}\n{sequence}\n"
                for sequence_id, sequence in sorted(sequence_by_id.items())
            )
        )
        prefix = root / "clusters"
        subprocess.run(
            [
                str(mmseqs),
                "easy-cluster",
                str(fasta),
                str(prefix),
                str(root / "tmp"),
                "--min-seq-id",
                str(min_seq_id),
                "-c",
                str(coverage),
                "--cov-mode",
                "0",
                "--cluster-mode",
                "1",
                "--threads",
                str(threads),
            ],
            check=True,
        )
        cluster_tsv = Path(f"{prefix}_cluster.tsv")
        members_by_representative: dict[str, list[str]] = defaultdict(list)
        with cluster_tsv.open() as handle:
            for line in handle:
                representative, member = line.rstrip("\n").split("\t")
                members_by_representative[representative].append(member)

    cluster_by_sequence = {}
    for members in members_by_representative.values():
        cluster_id = "sc_" + _digest("\n".join(sorted(members)).encode())
        for member in members:
            cluster_by_sequence[member] = cluster_id
    missing = set(sequence_by_id) - set(cluster_by_sequence)
    if missing:
        raise ValueError(f"MMseqs omitted sequence ids: {sorted(missing)[:5]}")
    return cluster_by_sequence


def build_cluster_benchmark(
    samples: list[ClusterSample],
    sequence_cluster_by_id: dict[str, str],
    out_root: str | Path,
    *,
    n_folds: int = 5,
) -> ClusterBenchmarkResult:
    if n_folds < 3:
        raise ValueError("Cluster benchmark requires at least three folds.")
    sequence_ids = {sample.sequence_id for sample in samples}
    if set(sequence_cluster_by_id) != sequence_ids:
        raise ValueError("Sequence cluster assignments do not match benchmark sequences.")

    union = _UnionFind()
    for sample in samples:
        sequence_node = f"sequence:{sequence_cluster_by_id[sample.sequence_id]}"
        pwm_node = f"pwm:{sample.pwm_hash}"
        union.union(sequence_node, pwm_node)

    samples_by_root: dict[str, list[ClusterSample]] = defaultdict(list)
    for sample in samples:
        node = f"sequence:{sequence_cluster_by_id[sample.sequence_id]}"
        samples_by_root[union.find(node)].append(sample)

    component_by_sample = {}
    samples_by_component = {}
    for members in samples_by_root.values():
        component_id = "component_" + _digest(
            "\n".join(sorted(sample.sample_id for sample in members)).encode()
        )
        samples_by_component[component_id] = members
        for sample in members:
            component_by_sample[sample.sample_id] = component_id

    fold_by_component = _balanced_folds(samples_by_component, n_folds)
    output = Path(out_root)
    fold_dir = output / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        component = component_by_sample[sample.sample_id]
        sample_rows.append(
            {
                "sample_id": sample.sample_id,
                "path": str(sample.path),
                "motif_id": sample.motif_id,
                "sequence_id": sample.sequence_id,
                "sequence_cluster": sequence_cluster_by_id[sample.sequence_id],
                "pwm_hash": sample.pwm_hash,
                "pwm_len": sample.pwm_len,
                "component": component,
                "fold": fold_by_component[component],
            }
        )

    component_rows = []
    for component, members in sorted(samples_by_component.items()):
        component_rows.append(
            {
                "component": component,
                "fold": fold_by_component[component],
                "n_samples": len(members),
                "n_exact_sequences": len({sample.sequence_id for sample in members}),
                "n_sequence_clusters": len(
                    {sequence_cluster_by_id[sample.sequence_id] for sample in members}
                ),
                "n_motifs": len({sample.motif_id for sample in members}),
                "n_pwm_targets": len({sample.pwm_hash for sample in members}),
            }
        )

    fold_rows = []
    for fold in range(n_folds):
        valid_fold = (fold + 1) % n_folds
        test_components = {
            component for component, assigned in fold_by_component.items() if assigned == fold
        }
        valid_components = {
            component
            for component, assigned in fold_by_component.items()
            if assigned == valid_fold
        }
        train_components = set(samples_by_component) - test_components - valid_components
        partition_components = {
            "train": train_components,
            "valid": valid_components,
            "test": test_components,
        }
        partition_samples = {
            name: [
                sample
                for component in sorted(components)
                for sample in samples_by_component[component]
            ]
            for name, components in partition_components.items()
        }
        _validate_partition(partition_samples, sequence_cluster_by_id)
        for name, members in partition_samples.items():
            _write_manifest(
                fold_dir / f"fold{fold}_{name}.txt",
                [sample.path for sample in members],
            )
        fold_rows.append(
            {
                "fold": fold,
                "valid_fold": valid_fold,
                **{
                    f"{name}_samples": len(members)
                    for name, members in partition_samples.items()
                },
                **{
                    f"{name}_components": len(components)
                    for name, components in partition_components.items()
                },
            }
        )

    sample_table = output / "sample_table.tsv"
    component_table = output / "component_table.tsv"
    fold_table = output / "fold_table.tsv"
    _write_rows(sample_table, sample_rows)
    _write_rows(component_table, component_rows)
    _write_rows(fold_table, fold_rows)
    _write_manifest(output / "all.txt", [sample.path for sample in samples])
    return ClusterBenchmarkResult(
        sample_table=sample_table,
        component_table=component_table,
        fold_table=fold_table,
        fold_dir=fold_dir,
        sample_count=len(samples),
        sequence_count=len(sequence_ids),
        sequence_cluster_count=len(set(sequence_cluster_by_id.values())),
        component_count=len(samples_by_component),
    )


def _validate_partition(
    partitions: dict[str, list[ClusterSample]],
    sequence_cluster_by_id: dict[str, str],
) -> None:
    for key in ("sequence_cluster", "pwm_hash"):
        values = {}
        for name, samples in partitions.items():
            if key == "sequence_cluster":
                values[name] = {
                    sequence_cluster_by_id[sample.sequence_id] for sample in samples
                }
            else:
                values[name] = {sample.pwm_hash for sample in samples}
        names = list(values)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = values[left] & values[right]
                if overlap:
                    raise ValueError(
                        f"{key} leakage between {left} and {right}: "
                        f"{sorted(overlap)[:5]}"
                    )


def _balanced_folds(
    samples_by_component: dict[str, list[ClusterSample]], n_folds: int
) -> dict[str, int]:
    sample_load = [0] * n_folds
    component_load = [0] * n_folds
    assignment = {}
    ordered = sorted(
        samples_by_component,
        key=lambda component: (-len(samples_by_component[component]), component),
    )
    for component in ordered:
        fold = min(
            range(n_folds),
            key=lambda index: (sample_load[index], component_load[index], index),
        )
        assignment[component] = fold
        sample_load[fold] += len(samples_by_component[component])
        component_load[fold] += 1
    return assignment


def _read_motif_ids(source_root: str | Path) -> dict[str, str]:
    motif_by_sample = {}
    for path in sorted(Path(source_root).glob("deeppbs_*_sources.tsv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sample_id = row["sample_id"]
                motif_id = row["motif_id"]
                existing = motif_by_sample.get(sample_id)
                if existing is not None and existing != motif_id:
                    raise ValueError(f"Conflicting motif ids for {sample_id}.")
                motif_by_sample[sample_id] = motif_id
    if not motif_by_sample:
        raise ValueError(f"No DeepPBS source manifests found under {source_root}.")
    return motif_by_sample


def _write_manifest(path: Path, samples: list[Path]) -> None:
    path.write_text("".join(f"{sample.resolve()}\n" for sample in samples))


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()[:16]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)
