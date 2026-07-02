from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rbe.data.alignment_selection import (
    AlignmentContactConstraints,
    AlignmentSelectionConfig,
)
from rbe.data.contact_labels import ContactCutoffs
from rbe.data.deeppbs_curated import (
    curated_pwm_path,
    download_pdb,
    parse_deeppbs_entry,
    read_entries,
    resolve_fold_file,
    write_pwm,
)
from rbe.data.processed_sample import (
    ComplexProcessingConfig,
    build_processed_complex_sample,
    write_processed_complex_sample,
)
from rbe.data.pwm import read_pwm
from rbe.data.source_prepare import LabelFilterConfig, passes_label_filters


@dataclass(frozen=True)
class DeepPBSCuratedPrepareConfig:
    curated_root: str | Path
    fold_file: str
    out_root: str | Path
    limit: int = 20
    max_attempts: int = 0
    label_filter: LabelFilterConfig = LabelFilterConfig()
    device: str = "cuda"
    alignment_score: str = "deeppbs_ic_pcc"
    allow_multi_char_chain: bool = False


@dataclass(frozen=True)
class DeepPBSCuratedPrepareResult:
    manifest_path: Path
    sample_table: Path
    failed_table: Path
    success_count: int
    attempt_count: int
    failure_count: int


def prepare_deeppbs_curated(
    config: DeepPBSCuratedPrepareConfig,
) -> DeepPBSCuratedPrepareResult:
    paths = _prepare_paths(config)
    fold_file = resolve_fold_file(config.fold_file, paths.curated_root)
    paths.out_root.mkdir(parents=True, exist_ok=True)
    paths.train_dir.mkdir(parents=True, exist_ok=True)

    successes = []
    failures = []
    attempts = 0
    table_rows = [_sample_table_header()]

    for entry in read_entries(fold_file):
        if config.limit and len(successes) >= config.limit:
            break
        if config.max_attempts and attempts >= config.max_attempts:
            break

        pdb_id, protein_chain, pwm_id = parse_deeppbs_entry(entry)
        if len(protein_chain) != 1 and not config.allow_multi_char_chain:
            failures.append((entry, "skip_multi_char_chain"))
            continue

        attempts += 1
        sample_id = f"{pdb_id}_{protein_chain}_{pwm_id}"
        pdb_path = paths.pdb_dir / f"{pdb_id}.pdb"
        pwm_path = paths.pwm_dir / f"{pwm_id}.txt"
        out_npz = paths.train_dir / f"{sample_id}.npz"

        try:
            download_pdb(pdb_id, pdb_path)
            write_pwm(pwm_path, read_pwm(curated_pwm_path(paths.curated_root, pwm_id)))
            built = build_processed_complex_sample(
                _complex_config(config, pdb_path, pwm_path, protein_chain)
            )
            write_processed_complex_sample(out_npz, built)
            counts = built.label_counts

            if not passes_label_filters(counts, config.label_filter):
                reason = _low_contact_reason(counts)
                failures.append((entry, reason))
                out_npz.unlink(missing_ok=True)
                print(f"SKIP {sample_id}: {reason}")
                continue

            successes.append(out_npz.resolve())
            table_rows.append(
                _sample_table_row(
                    sample_id=sample_id,
                    pdb_path=pdb_path,
                    pwm_path=pwm_path,
                    protein_chain=protein_chain,
                    pwm_id=pwm_id,
                    counts=counts,
                )
            )
            print(f"OK {sample_id}: {_counts_message(counts)}")
        except Exception as exc:
            failures.append((entry, repr(exc)))
            print(f"FAIL {entry}: {exc}")

    _write_manifest(paths.manifest_path, successes)
    _write_table(paths.sample_table, table_rows)
    _write_failures(paths.failed_table, failures)

    return DeepPBSCuratedPrepareResult(
        manifest_path=paths.manifest_path,
        sample_table=paths.sample_table,
        failed_table=paths.failed_table,
        success_count=len(successes),
        attempt_count=attempts,
        failure_count=len(failures),
    )


@dataclass(frozen=True)
class _PreparePaths:
    curated_root: Path
    out_root: Path
    pdb_dir: Path
    pwm_dir: Path
    train_dir: Path
    manifest_path: Path
    sample_table: Path
    failed_table: Path


def _prepare_paths(config: DeepPBSCuratedPrepareConfig) -> _PreparePaths:
    out_root = Path(config.out_root)
    return _PreparePaths(
        curated_root=Path(config.curated_root).resolve(),
        out_root=out_root,
        pdb_dir=out_root / "raw" / "pdb",
        pwm_dir=out_root / "raw" / "pwm",
        train_dir=out_root / "train",
        manifest_path=out_root / "train_manifest.txt",
        sample_table=out_root / "sample_table.tsv",
        failed_table=out_root / "failed.tsv",
    )


def _complex_config(
    config: DeepPBSCuratedPrepareConfig,
    pdb_path: Path,
    pwm_path: Path,
    protein_chain: str,
) -> ComplexProcessingConfig:
    return ComplexProcessingConfig(
        structure_path=pdb_path,
        pwm_path=pwm_path,
        protein_chains=protein_chain,
        alignment=AlignmentSelectionConfig(
            score_mode=config.alignment_score,
            contact_policy="require_contact",
            contact_cutoffs=ContactCutoffs(),
            contact_constraints=AlignmentContactConstraints(
                min_base_pairs=config.label_filter.min_base_pairs,
                min_contact_pairs=config.label_filter.min_contact_pairs,
                min_site_residues=config.label_filter.min_site_residues,
            ),
        ),
        device=config.device,
    )


def _sample_table_header() -> str:
    return (
        "sample_id\tpdb_path\tpwm_path\tprotein_chains\tdna_chains\tpwm_id\t"
        "A_base_pos\tA_backbone_pos\tA_contact_pos\tsite_pos"
    )


def _sample_table_row(
    sample_id: str,
    pdb_path: Path,
    pwm_path: Path,
    protein_chain: str,
    pwm_id: str,
    counts: dict[str, int],
) -> str:
    return "\t".join(
        [
            sample_id,
            str(pdb_path),
            str(pwm_path),
            protein_chain,
            "",
            pwm_id,
            str(counts["A_base_pos"]),
            str(counts["A_backbone_pos"]),
            str(counts["A_contact_pos"]),
            str(counts["site_pos"]),
        ]
    )


def _counts_message(counts: dict[str, int]) -> str:
    return (
        f"A_base_pos={counts['A_base_pos']} "
        f"A_backbone_pos={counts['A_backbone_pos']} "
        f"A_contact_pos={counts['A_contact_pos']} "
        f"site_pos={counts['site_pos']}"
    )


def _low_contact_reason(counts: dict[str, int]) -> str:
    return "filtered_low_contact " + _counts_message(counts)


def _write_manifest(path: Path, successes: list[Path]) -> None:
    path.write_text("\n".join(str(sample_path) for sample_path in successes) + "\n")


def _write_table(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n")


def _write_failures(path: Path, failures: list[tuple[str, str]]) -> None:
    path.write_text(
        "entry\treason\n" + "\n".join(f"{entry}\t{reason}" for entry, reason in failures)
    )
