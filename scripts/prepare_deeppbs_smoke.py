from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

from rbe.data.pwm import normalize_pwm, read_pwm


DEFAULT_CURATED_ROOT = (
    Path(__file__).resolve().parents[1] / "resources" / "deeppbs_curated"
)


def parse_deeppbs_entry(entry: str) -> tuple[str, str, str]:
    name = Path(entry.strip()).name
    if name.endswith(".npz"):
        name = name[:-4]
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse DeepPBS entry: {entry}")
    pdb_id = parts[0].lower()
    protein_chain = parts[1]
    pwm_id = "_".join(parts[2:])
    return pdb_id, protein_chain, pwm_id


def write_pwm(path: Path, pwm) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("A C G T\n")
        for row in normalize_pwm(pwm):
            handle.write("\t".join(f"{float(value):.8f}" for value in row) + "\n")


def resolve_fold_file(fold_file: str, curated_root: Path) -> Path:
    path = Path(fold_file)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                curated_root / "folds" / fold_file,
                curated_root / "folds" / Path(fold_file).name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find fold file {fold_file}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def curated_pwm_path(curated_root: Path, pwm_id: str) -> Path:
    path = curated_root / "pwms" / f"{pwm_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Curated PWM not found for {pwm_id}: {path}")
    return path


def download_pdb(pdb_id: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    urllib.request.urlretrieve(url, path)


def read_entries(path: Path) -> list[str]:
    entries = []
    with path.open() as handle:
        for line in handle:
            item = line.strip()
            if item and not item.startswith("#"):
                entries.append(item)
    return entries


def run_process_complex(
    pdb_path: Path,
    pwm_path: Path,
    protein_chain: str,
    out_npz: Path,
    device: str,
    alignment_score: str,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "-m",
        "rbe.data.process_complex",
        "--pdb",
        str(pdb_path),
        "--pwm",
        str(pwm_path),
        "--protein-chains",
        protein_chain,
        "--output",
        str(out_npz),
        "--device",
        device,
        "--alignment-score",
        alignment_score,
    ]
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def label_counts(npz_path: str | Path) -> dict[str, int]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {
            "A_base_pos": int(data["A_base_label"].sum()),
            "A_backbone_pos": int(data["A_backbone_label"].sum()),
            "A_contact_pos": int(data["A_contact_label"].sum()),
            "site_pos": int(data["site_label"].sum()),
        }


def passes_label_filters(counts: dict[str, int], args: argparse.Namespace) -> bool:
    return (
        counts["A_contact_pos"] >= args.min_contact_pairs
        and counts["site_pos"] >= args.min_site_residues
        and counts["A_base_pos"] >= args.min_base_pairs
    )


def prepare(args: argparse.Namespace) -> None:
    curated_root = Path(args.curated_root).resolve()
    fold_file = resolve_fold_file(args.fold_file, curated_root)

    out_root = Path(args.out_root)
    pdb_dir = out_root / "raw" / "pdb"
    pwm_dir = out_root / "raw" / "pwm"
    train_dir = out_root / "train"
    manifest_path = out_root / "train_manifest.txt"
    table_path = out_root / "sample_table.tsv"
    failed_path = out_root / "failed.tsv"
    out_root.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    entries = read_entries(fold_file)
    successes = []
    failures = []
    attempts = 0
    table_rows = [
        "sample_id\tpdb_path\tpwm_path\tprotein_chains\tdna_chains\tpwm_id\t"
        "A_base_pos\tA_backbone_pos\tA_contact_pos\tsite_pos"
    ]

    for entry in entries:
        if args.limit and len(successes) >= args.limit:
            break
        if args.max_attempts and attempts >= args.max_attempts:
            break

        pdb_id, protein_chain, pwm_id = parse_deeppbs_entry(entry)
        if len(protein_chain) != 1 and not args.allow_multi_char_chain:
            failures.append((entry, "skip_multi_char_chain"))
            continue

        attempts += 1
        sample_id = f"{pdb_id}_{protein_chain}_{pwm_id}"
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        pwm_path = pwm_dir / f"{pwm_id}.txt"
        out_npz = train_dir / f"{sample_id}.npz"

        try:
            download_pdb(pdb_id, pdb_path)
            pwm = read_pwm(curated_pwm_path(curated_root, pwm_id))
            write_pwm(pwm_path, pwm)
            result = run_process_complex(
                pdb_path=pdb_path,
                pwm_path=pwm_path,
                protein_chain=protein_chain,
                out_npz=out_npz,
                device=args.device,
                alignment_score=args.alignment_score,
            )
            if result.returncode != 0:
                failures.append((entry, result.stdout.strip().replace("\n", " | ")))
                print(f"FAIL {entry}")
                print(result.stdout)
                continue
            counts = label_counts(out_npz)
            if not passes_label_filters(counts, args):
                reason = (
                    "filtered_low_contact "
                    f"A_base_pos={counts['A_base_pos']} "
                    f"A_backbone_pos={counts['A_backbone_pos']} "
                    f"A_contact_pos={counts['A_contact_pos']} "
                    f"site_pos={counts['site_pos']}"
                )
                failures.append((entry, reason))
                out_npz.unlink(missing_ok=True)
                print(f"SKIP {sample_id}: {reason}")
                continue
            successes.append(out_npz.resolve())
            table_rows.append(
                "\t".join(
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
            )
            last_line = result.stdout.strip().splitlines()[-1]
            print(f"OK {sample_id}: {last_line}")
        except Exception as exc:
            failures.append((entry, repr(exc)))
            print(f"FAIL {entry}: {exc}")

    manifest_path.write_text("\n".join(str(path) for path in successes) + "\n")
    table_path.write_text("\n".join(table_rows) + "\n")
    failed_path.write_text(
        "entry\treason\n" + "\n".join(f"{entry}\t{reason}" for entry, reason in failures)
    )
    print(
        f"wrote {manifest_path} successes={len(successes)} "
        f"attempts={attempts} failures={len(failures)}"
    )
    print(f"wrote {table_path}")
    print(f"wrote {failed_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an RBE smoke-test dataset from vendored DeepPBS mappings."
    )
    parser.add_argument("--curated-root", default=str(DEFAULT_CURATED_ROOT))
    parser.add_argument("--fold-file", default="valid0.txt")
    parser.add_argument("--out-root", default="data/deeppbs_smoke")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--min-base-pairs", type=int, default=0)
    parser.add_argument("--min-contact-pairs", type=int, default=1)
    parser.add_argument("--min-site-residues", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--alignment-score",
        choices=["deeppbs_ic_pcc", "ic_log_likelihood", "log_likelihood"],
        default="deeppbs_ic_pcc",
    )
    parser.add_argument(
        "--allow-multi-char-chain",
        action="store_true",
        help="Allow non-PDB-style multi-character protein chain ids.",
    )
    return parser


def main() -> None:
    prepare(build_argparser().parse_args())


if __name__ == "__main__":
    main()
