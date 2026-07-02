from __future__ import annotations

from pathlib import Path
import urllib.request

from rbe.data.pwm import normalize_pwm


DEFAULT_CURATED_ROOT = (
    Path(__file__).resolve().parents[3] / "resources" / "deeppbs_curated"
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


def resolve_fold_file(fold_file: str, curated_root: Path) -> Path:
    path = Path(fold_file)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                curated_root / "folds" / fold_file,
                curated_root / "folds" / path.name,
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


def read_entries(path: Path) -> list[str]:
    entries = []
    with path.open() as handle:
        for line in handle:
            item = line.strip()
            if item and not item.startswith("#"):
                entries.append(item)
    return entries


def write_pwm(path: Path, pwm) -> None:
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
