from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rbe.data.alignment import reverse_complement_sequence
from rbe.data.pwm import reverse_complement_pwm


SPEC_COLUMNS = (
    "sample_id",
    "protein_group",
    "motif_id",
    "include",
    "reverse_complement",
    "crop_start",
    "core_length",
    "reason",
)

SLOT_VECTOR_KEYS = ("pwm_mask", "slot_to_dna_index")
SLOT_MATRIX_KEYS = (
    "A_label",
    "A_base_label",
    "A_base_mask",
    "A_backbone_label",
    "A_contact_label",
)


@dataclass(frozen=True)
class FamilySampleSpec:
    sample_id: str
    protein_group: str
    motif_id: str
    include: bool
    reverse_complement: bool | None
    crop_start: int | None
    core_length: int | None
    reason: str


@dataclass(frozen=True)
class FamilyBenchmarkResult:
    processed_dir: Path
    fold_dir: Path
    sample_table: Path
    included_samples: int
    protein_groups: int


def read_family_specs(path: str | Path) -> list[FamilySampleSpec]:
    source = Path(path)
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Family spec has no header: {source}")
        missing = sorted(set(SPEC_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{source} is missing columns: {', '.join(missing)}")
        specs = [_spec_from_row(row, source) for row in reader]

    sample_ids = [spec.sample_id for spec in specs]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{source} contains duplicate sample_id values.")
    if not any(spec.include for spec in specs):
        raise ValueError(f"{source} has no included samples.")
    return specs


def transform_family_sample(
    arrays: dict[str, np.ndarray],
    spec: FamilySampleSpec,
    orientation: str,
) -> dict[str, np.ndarray]:
    if not spec.include:
        raise ValueError(f"Cannot transform excluded sample {spec.sample_id}.")
    if spec.reverse_complement is None or spec.crop_start is None or spec.core_length is None:
        raise ValueError(f"{spec.sample_id}: included sample has incomplete alignment fields.")
    if "pwm_target" not in arrays:
        raise ValueError(f"{spec.sample_id}: source cache is missing pwm_target.")

    result = {key: np.asarray(value).copy() for key, value in arrays.items()}
    motif_len = int(result["pwm_target"].shape[0])
    _validate_slot_shapes(result, motif_len, spec.sample_id)

    if spec.reverse_complement:
        result["pwm_target"] = reverse_complement_pwm(result["pwm_target"])
        for key in SLOT_VECTOR_KEYS:
            result[key] = result[key][::-1].copy()
        for key in SLOT_MATRIX_KEYS:
            result[key] = result[key][:, ::-1].copy()
        result["alignment_sequence"] = np.asarray(
            reverse_complement_sequence(str(result["alignment_sequence"]))
        )

    start = spec.crop_start
    stop = start + spec.core_length
    if start < 0 or stop > motif_len:
        raise ValueError(
            f"{spec.sample_id}: crop [{start}:{stop}] exceeds motif length {motif_len}."
        )
    result["pwm_target"] = result["pwm_target"][start:stop].copy()
    for key in SLOT_VECTOR_KEYS:
        result[key] = result[key][start:stop].copy()
    for key in SLOT_MATRIX_KEYS:
        result[key] = result[key][:, start:stop].copy()
    result["alignment_sequence"] = np.asarray(
        str(result["alignment_sequence"])[start:stop]
    )
    result["site_label"] = result["A_contact_label"].max(axis=1).astype(np.float32)

    result["pwm_orientation"] = np.asarray(orientation)
    result["family_name"] = np.asarray(orientation.split(":")[1])
    result["protein_group"] = np.asarray(spec.protein_group)
    result["source_sample_id"] = np.asarray(spec.sample_id)
    result["family_reverse_complement"] = np.asarray(
        spec.reverse_complement, dtype=bool
    )
    result["family_crop_start"] = np.asarray(start, dtype=np.int64)
    result["family_core_length"] = np.asarray(spec.core_length, dtype=np.int64)
    _validate_slot_shapes(result, spec.core_length, spec.sample_id)
    return result


def prepare_family_benchmark(
    cache_root: str | Path,
    spec_path: str | Path,
    out_root: str | Path,
    *,
    family_name: str,
    version: str,
) -> FamilyBenchmarkResult:
    cache_root = Path(cache_root)
    out_root = Path(out_root)
    processed_dir = out_root / "processed"
    fold_dir = out_root / "folds"
    processed_dir.mkdir(parents=True, exist_ok=True)
    fold_dir.mkdir(parents=True, exist_ok=True)

    specs = read_family_specs(spec_path)
    included = [spec for spec in specs if spec.include]
    orientation = f"family_reference:{family_name}:{version}"
    paths_by_group: dict[str, list[Path]] = {}
    sample_rows = []

    for spec in included:
        source = cache_root / "processed" / f"{spec.sample_id}.npz"
        if not source.exists():
            raise FileNotFoundError(f"Missing canonical source sample: {source}")
        with np.load(source, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        if "canonical_reverse_complement" not in arrays:
            raise ValueError(f"{source}: source sample is not canonical cache data.")
        transformed = transform_family_sample(arrays, spec, orientation)
        destination = (processed_dir / f"{spec.sample_id}.npz").resolve()
        np.savez_compressed(destination, **transformed)
        paths_by_group.setdefault(spec.protein_group, []).append(destination)
        sample_rows.append(
            {
                "sample_id": spec.sample_id,
                "protein_group": spec.protein_group,
                "motif_id": spec.motif_id,
                "path": str(destination),
                "consensus": _consensus(transformed["pwm_target"]),
                "visible_columns": str(int(transformed["pwm_mask"].sum())),
                "core_length": str(spec.core_length),
            }
        )

    groups = sorted(paths_by_group)
    if len(groups) < 3:
        raise ValueError("Family benchmark requires at least three protein groups.")
    for fold, test_group in enumerate(groups):
        valid_group = groups[(fold + 1) % len(groups)]
        train_groups = [group for group in groups if group not in {test_group, valid_group}]
        _write_manifest(
            fold_dir / f"fold{fold}_train.txt",
            [path for group in train_groups for path in paths_by_group[group]],
        )
        _write_manifest(fold_dir / f"fold{fold}_valid.txt", paths_by_group[valid_group])
        _write_manifest(fold_dir / f"fold{fold}_test.txt", paths_by_group[test_group])

    all_paths = [path for paths in paths_by_group.values() for path in paths]
    _write_manifest(out_root / "all.txt", all_paths)
    sample_table = out_root / "sample_table.tsv"
    _write_rows(sample_table, sample_rows)
    _write_fold_table(out_root / "fold_table.tsv", groups)
    return FamilyBenchmarkResult(
        processed_dir=processed_dir,
        fold_dir=fold_dir,
        sample_table=sample_table,
        included_samples=len(included),
        protein_groups=len(groups),
    )


def _spec_from_row(row: dict[str, str], source: Path) -> FamilySampleSpec:
    sample_id = row["sample_id"].strip()
    protein_group = row["protein_group"].strip()
    include = _parse_bool(row["include"], f"{source}:{sample_id}:include")
    if not sample_id or not protein_group or not row["motif_id"].strip():
        raise ValueError(f"{source}: family rows require sample_id, protein_group, motif_id.")
    return FamilySampleSpec(
        sample_id=sample_id,
        protein_group=protein_group,
        motif_id=row["motif_id"].strip(),
        include=include,
        reverse_complement=(
            _parse_bool(
                row["reverse_complement"],
                f"{source}:{sample_id}:reverse_complement",
            )
            if include
            else None
        ),
        crop_start=(int(row["crop_start"]) if include else None),
        core_length=(int(row["core_length"]) if include else None),
        reason=row["reason"].strip(),
    )


def _parse_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"{label} must be 0/1 or false/true, got {value!r}.")


def _validate_slot_shapes(arrays: dict[str, np.ndarray], motif_len: int, sample_id: str) -> None:
    if arrays["pwm_target"].shape != (motif_len, 4):
        raise ValueError(f"{sample_id}: invalid pwm_target shape {arrays['pwm_target'].shape}.")
    for key in SLOT_VECTOR_KEYS:
        if key not in arrays or arrays[key].shape != (motif_len,):
            raise ValueError(f"{sample_id}: {key} must have shape {(motif_len,)}.")
    n_residue = arrays["residue_aa"].shape[0]
    for key in SLOT_MATRIX_KEYS:
        if key not in arrays or arrays[key].shape != (n_residue, motif_len):
            raise ValueError(
                f"{sample_id}: {key} must have shape {(n_residue, motif_len)}."
            )


def _consensus(pwm: np.ndarray) -> str:
    return "".join("ACGT"[index] for index in np.asarray(pwm).argmax(axis=1))


def _write_manifest(path: Path, samples: list[Path]) -> None:
    path.write_text("".join(f"{sample}\n" for sample in samples))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_fold_table(path: Path, groups: list[str]) -> None:
    rows = []
    for fold, test_group in enumerate(groups):
        valid_group = groups[(fold + 1) % len(groups)]
        rows.append(
            {
                "fold": str(fold),
                "test_group": test_group,
                "valid_group": valid_group,
                "train_groups": ",".join(
                    group for group in groups if group not in {test_group, valid_group}
                ),
            }
        )
    _write_rows(path, rows)
