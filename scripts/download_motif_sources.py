from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rbe.data.deeppbs_curated import DEFAULT_CURATED_ROOT, write_pwm


HOCOMOCO_URLS = {
    "HUMAN": (
        "https://hocomoco11.autosome.org/final_bundle/hocomoco11/full/HUMAN/mono/"
        "HOCOMOCOv11_full_HUMAN_mono_jaspar_format.txt"
    ),
    "MOUSE": (
        "https://hocomoco11.autosome.org/final_bundle/hocomoco11/full/MOUSE/mono/"
        "HOCOMOCOv11_full_MOUSE_mono_jaspar_format.txt"
    ),
}
JASPAR_URL_TEMPLATE = "https://jaspar.elixir.no/api/v1/matrix/{matrix_id}.jaspar"
JASPAR_UNVALIDATED_URL = (
    "https://jaspar2022.genereg.net/download/data/2022/collections/"
    "JASPAR2022_UNVALIDATED_redundant_pfms_jaspar.txt"
)
INDEX_COLUMNS = [
    "motif_id",
    "motif_source",
    "motif_version",
    "motif_path",
]


def download_motif_sources(args: argparse.Namespace) -> None:
    curated_root = Path(args.curated_root)
    out_root = Path(args.out_root)
    motif_index = Path(args.motif_index)
    motif_ids = _selected_motif_ids(curated_root, args.motif_id)

    raw_dir = out_root / "raw"
    pwm_dir = out_root / "pwms"
    rows = []

    hocomoco_cache: dict[str, dict[str, np.ndarray]] = {}
    jaspar_unvalidated_cache: dict[str, np.ndarray] | None = None
    for motif_id in motif_ids:
        if is_jaspar_id(motif_id):
            matrix_id = jaspar_matrix_id(motif_id)
            if matrix_id.startswith("UN"):
                url = JASPAR_UNVALIDATED_URL
                raw_path = raw_dir / "jaspar" / Path(url).name
                if jaspar_unvalidated_cache is None:
                    text = download_text(url, raw_path, overwrite=args.overwrite)
                    jaspar_unvalidated_cache = parse_jaspar_collection(text)
                if matrix_id not in jaspar_unvalidated_cache:
                    raise KeyError(f"{matrix_id} not found in JASPAR 2022 UNVALIDATED collection")
                pwm = jaspar_unvalidated_cache[matrix_id]
                version = "JASPAR2022-unvalidated-untrimmed"
            else:
                url = JASPAR_URL_TEMPLATE.format(matrix_id=matrix_id)
                raw_path = raw_dir / "jaspar" / f"{matrix_id}.jaspar"
                text = download_text(url, raw_path, overwrite=args.overwrite)
                pwm = parse_jaspar_matrix(text)
                version = f"{matrix_id}-untrimmed"
            source = "JASPAR"
        else:
            species = hocomoco_species(motif_id)
            if species not in hocomoco_cache:
                url = HOCOMOCO_URLS[species]
                raw_path = raw_dir / "hocomoco" / Path(url).name
                text = download_text(url, raw_path, overwrite=args.overwrite)
                hocomoco_cache[species] = parse_hocomoco_jaspar_collection(text)
            pwm_by_id = hocomoco_cache[species]
            if motif_id not in pwm_by_id:
                raise KeyError(f"{motif_id} not found in HOCOMOCO v11 {species} full collection")
            pwm = pwm_by_id[motif_id]
            url = HOCOMOCO_URLS[species]
            source = "HOCOMOCO"
            version = "v11-full-untrimmed"

        pwm_path = pwm_dir / f"{motif_id}.txt"
        write_pwm(pwm_path, pwm)
        rows.append(
            {
                "motif_id": motif_id,
                "motif_source": source,
                "motif_version": version,
                "motif_path": _display_path(pwm_path, motif_index.parent),
            }
        )

    write_motif_index(motif_index, rows)
    print(f"wrote {motif_index} motifs={len(rows)}")


def _selected_motif_ids(curated_root: Path, requested: list[str]) -> list[str]:
    if requested:
        return sorted(set(requested))
    pwm_dir = curated_root / "pwms"
    motif_ids = sorted(path.stem for path in pwm_dir.glob("*.txt"))
    if not motif_ids:
        raise ValueError(f"No curated PWM ids found in {pwm_dir}")
    return motif_ids


def is_jaspar_id(motif_id: str) -> bool:
    return motif_id.endswith(".jaspar")


def jaspar_matrix_id(motif_id: str) -> str:
    if not is_jaspar_id(motif_id):
        raise ValueError(f"Not a JASPAR motif id: {motif_id}")
    return motif_id[: -len(".jaspar")]


def hocomoco_species(motif_id: str) -> str:
    if "_HUMAN." in motif_id:
        return "HUMAN"
    if "_MOUSE." in motif_id:
        return "MOUSE"
    raise ValueError(f"Cannot infer HOCOMOCO species for motif id: {motif_id}")


def download_text(url: str, path: Path, overwrite: bool = False) -> str:
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return path.read_text()
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    path.write_text(text)
    return text


def parse_jaspar_matrix(text: str) -> np.ndarray:
    rows_by_base: dict[str, list[float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(">"):
            continue
        base = line[0].upper()
        if base in {"A", "C", "G", "T"}:
            rows_by_base[base] = _numeric_tokens(line[1:])
    return _base_rows_to_pwm(rows_by_base)


def parse_jaspar_collection(text: str) -> dict[str, np.ndarray]:
    motifs: dict[str, np.ndarray] = {}
    motif_id: str | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal motif_id, lines
        if motif_id is None:
            return
        motifs[motif_id] = parse_jaspar_matrix("\n".join(lines))
        lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            motif_id = line[1:].split()[0]
            lines = [line]
            continue
        lines.append(line)
    flush()
    return motifs


def parse_hocomoco_jaspar_collection(text: str) -> dict[str, np.ndarray]:
    motifs: dict[str, np.ndarray] = {}
    motif_id: str | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        nonlocal motif_id, rows
        if motif_id is None:
            return
        if len(rows) != 4:
            raise ValueError(f"{motif_id}: expected 4 base rows, got {len(rows)}")
        motifs[motif_id] = _base_rows_to_pwm(
            {"A": rows[0], "C": rows[1], "G": rows[2], "T": rows[3]}
        )
        rows = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            motif_id = line[1:].split()[0]
            rows = []
            continue
        rows.append(_numeric_tokens(line))
    flush()
    return motifs


def _numeric_tokens(text: str) -> list[float]:
    return [float(token) for token in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]


def _base_rows_to_pwm(rows_by_base: dict[str, list[float]]) -> np.ndarray:
    missing = [base for base in ("A", "C", "G", "T") if base not in rows_by_base]
    if missing:
        raise ValueError(f"Missing base rows: {', '.join(missing)}")
    lengths = {len(rows_by_base[base]) for base in ("A", "C", "G", "T")}
    if len(lengths) != 1:
        raise ValueError(f"Base rows have inconsistent lengths: {sorted(lengths)}")
    counts = np.asarray([rows_by_base[base] for base in ("A", "C", "G", "T")], dtype=np.float32)
    if counts.shape[1] == 0:
        raise ValueError("PWM has zero columns")
    return counts.T


def write_motif_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=INDEX_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["motif_id"]):
            writer.writerow(row)


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download untrimmed motif sources for DeepPBS fold motif ids and write "
            "an index usable by import_deeppbs_source_manifest.py."
        )
    )
    parser.add_argument("--curated-root", default=str(DEFAULT_CURATED_ROOT))
    parser.add_argument("--out-root", default="resources/motif_sources")
    parser.add_argument("--motif-index", default="resources/motif_sources/motif_index.tsv")
    parser.add_argument(
        "--motif-id",
        action="append",
        default=[],
        help="Download only this motif id. Can be repeated. Default: all curated ids.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    download_motif_sources(build_argparser().parse_args())


if __name__ == "__main__":
    main()
