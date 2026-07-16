from __future__ import annotations

from pathlib import Path

import numpy as np

from rbe.data.family_benchmark import (
    FamilySampleSpec,
    prepare_family_benchmark,
    read_family_specs,
    transform_family_sample,
)


def _source_arrays() -> dict[str, np.ndarray]:
    pwm = np.asarray(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ],
        dtype=np.float32,
    )
    matrix = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    return {
        "residue_aa": np.asarray(["A", "C"]),
        "pwm_target": pwm,
        "pwm_mask": np.asarray([1, 1, 0, 1], dtype=np.float32),
        "slot_to_dna_index": np.asarray([10, 11, -1, 13], dtype=np.int64),
        "A_label": matrix.copy(),
        "A_base_label": matrix.copy(),
        "A_base_mask": np.ones_like(matrix),
        "A_backbone_label": matrix.copy(),
        "A_contact_label": matrix.copy(),
        "site_label": np.ones(2, dtype=np.float32),
        "alignment_sequence": np.asarray("ACGT"),
        "canonical_reverse_complement": np.asarray(False, dtype=bool),
    }


def test_transform_family_sample_reverses_and_crops_every_slot_axis():
    spec = FamilySampleSpec(
        sample_id="sample",
        protein_group="P1",
        motif_id="M1",
        include=True,
        reverse_complement=True,
        crop_start=1,
        core_length=2,
        reason="test",
    )

    result = transform_family_sample(
        _source_arrays(), spec, "family_reference:ETS:v1"
    )

    np.testing.assert_allclose(
        result["pwm_target"],
        np.asarray([[0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]]),
    )
    np.testing.assert_array_equal(result["pwm_mask"], [0, 1])
    np.testing.assert_array_equal(result["slot_to_dna_index"], [-1, 11])
    np.testing.assert_array_equal(result["A_base_label"], [[3, 2], [7, 6]])
    np.testing.assert_array_equal(result["site_label"], [3, 7])
    assert str(result["alignment_sequence"]) == "CG"
    assert str(result["pwm_orientation"]) == "family_reference:ETS:v1"
    assert str(result["family_name"]) == "ETS"


def test_ets_v1_spec_is_grouped_and_auditable():
    root = Path(__file__).resolve().parents[1]
    specs = read_family_specs(
        root / "resources/family_benchmarks/ets_v1/samples.tsv"
    )
    included = [spec for spec in specs if spec.include]

    assert len(specs) == 37
    assert len(included) == 26
    assert len({spec.protein_group for spec in included}) == 12
    assert {spec.core_length for spec in included} == {9}


def test_prepare_family_benchmark_keeps_protein_groups_disjoint(tmp_path: Path):
    cache = tmp_path / "cache/processed"
    cache.mkdir(parents=True)
    rows = [
        ("g1a", "G1"),
        ("g1b", "G1"),
        ("g2", "G2"),
        ("g3", "G3"),
    ]
    for sample_id, _ in rows:
        np.savez_compressed(cache / f"{sample_id}.npz", **_source_arrays())
    spec = tmp_path / "samples.tsv"
    spec.write_text(
        "sample_id\tprotein_group\tmotif_id\tinclude\treverse_complement\t"
        "crop_start\tcore_length\treason\n"
        + "".join(
            f"{sample_id}\t{group}\tM1\t1\t0\t1\t2\ttest\n"
            for sample_id, group in rows
        )
    )

    result = prepare_family_benchmark(
        tmp_path / "cache",
        spec,
        tmp_path / "benchmark",
        family_name="ETS",
        version="v1",
    )

    assert result.included_samples == 4
    assert result.protein_groups == 3
    fold0_test = (result.fold_dir / "fold0_test.txt").read_text().splitlines()
    fold0_valid = (result.fold_dir / "fold0_valid.txt").read_text().splitlines()
    fold0_train = (result.fold_dir / "fold0_train.txt").read_text().splitlines()
    assert {Path(path).stem for path in fold0_test} == {"g1a", "g1b"}
    assert {Path(path).stem for path in fold0_valid} == {"g2"}
    assert {Path(path).stem for path in fold0_train} == {"g3"}
    assert set(fold0_test).isdisjoint(fold0_valid)
    assert set(fold0_test).isdisjoint(fold0_train)
    assert set(fold0_valid).isdisjoint(fold0_train)
