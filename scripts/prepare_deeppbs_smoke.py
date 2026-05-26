from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

from rbe.data.pwm import normalize_pwm


BASES = "ACGT"


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


def pwm_information_content(pwm: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    pwm = normalize_pwm(pwm, eps=eps)
    return 2.0 + np.sum(pwm * np.log2(pwm), axis=1)


def load_deeppbs_pwm(
    deeppbs_root: Path, pwm_id: str, trim_ic_threshold: float
) -> np.ndarray:
    pwm_pickle = deeppbs_root / "deeppbs" / "_data" / "pwms.pickle"
    with pwm_pickle.open("rb") as handle:
        pwm_dict = pickle.load(handle)
    if pwm_id not in pwm_dict:
        raise KeyError(f"PWM id not found in DeepPBS pwms.pickle: {pwm_id}")

    motif = pwm_dict[pwm_id]
    raw = np.asarray([motif.pwm[base] for base in BASES], dtype=np.float32).T
    pwm = normalize_pwm(raw)
    ic = pwm_information_content(pwm)
    keep = np.where(ic > trim_ic_threshold)[0]
    if keep.size == 0:
        return pwm
    return pwm[int(keep[0]) : int(keep[-1]) + 1]


def write_pwm(path: Path, pwm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("A C G T\n")
        for row in normalize_pwm(pwm):
            handle.write("\t".join(f"{float(value):.8f}" for value in row) + "\n")


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


def prepare(args: argparse.Namespace) -> None:
    deeppbs_root = Path(args.deeppbs_root).resolve()
    fold_file = Path(args.fold_file)
    if not fold_file.is_absolute():
        fold_file = deeppbs_root / fold_file

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
    table_rows = ["sample_id\tpdb_path\tpwm_path\tprotein_chains\tdna_chains\tpwm_id"]

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
            pwm = load_deeppbs_pwm(
                deeppbs_root, pwm_id, trim_ic_threshold=args.trim_ic_threshold
            )
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
        description="Prepare an RBE smoke-test dataset from DeepPBS fold mappings."
    )
    parser.add_argument("--deeppbs-root", default="../DeepPBS")
    parser.add_argument("--fold-file", default="run/folds/valid0.txt")
    parser.add_argument("--out-root", default="data/deeppbs_smoke")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--alignment-score",
        choices=["deeppbs_ic_pcc", "ic_log_likelihood", "log_likelihood"],
        default="deeppbs_ic_pcc",
    )
    parser.add_argument("--trim-ic-threshold", type=float, default=0.5)
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
