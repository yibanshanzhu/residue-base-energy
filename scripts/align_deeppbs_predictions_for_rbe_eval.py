from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


BASES = "ACGT"
COMPLEMENT_INDEX = np.asarray([3, 2, 1, 0], dtype=np.int64)
COMPLEMENT_TRANS = str.maketrans("ACGT", "TGCA")


def one_hot_to_seq(one_hot: np.ndarray) -> str:
    return "".join(BASES[int(idx)] for idx in one_hot.argmax(axis=1))


def complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT_TRANS)


def reverse_complement(sequence: str) -> str:
    return complement(sequence)[::-1]


def read_manifest(path: Path) -> list[Path]:
    root = path.resolve().parent
    samples = []
    with path.open() as handle:
        for line in handle:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            sample = Path(item)
            samples.append(sample if sample.is_absolute() else root / sample)
    if not samples:
        raise ValueError(f"No samples found in manifest: {path}")
    return samples


def _candidate_from_index(
    pwm: np.ndarray,
    sequence: str,
    indices: np.ndarray,
    aligned_sequence: str,
) -> tuple[np.ndarray, str, str] | None:
    if indices.min(initial=0) < 0 or indices.max(initial=-1) >= pwm.shape[0]:
        return None

    raw = "".join(sequence[int(idx)] for idx in indices)
    picked = pwm[indices]
    if raw == aligned_sequence:
        return picked, "slot_index_direct", raw
    if complement(raw) == aligned_sequence:
        return picked[:, COMPLEMENT_INDEX], "slot_index_complement", complement(raw)
    if raw[::-1] == aligned_sequence:
        return picked[::-1], "slot_index_reverse", raw[::-1]
    if reverse_complement(raw) == aligned_sequence:
        return (
            picked[::-1][:, COMPLEMENT_INDEX],
            "slot_index_reverse_complement",
            reverse_complement(raw),
        )
    return None


def _candidate_from_window(
    pwm: np.ndarray,
    sequence: str,
    motif_len: int,
    aligned_sequence: str,
) -> tuple[np.ndarray, str, str] | None:
    if pwm.shape[0] < motif_len:
        return None

    for start in range(pwm.shape[0] - motif_len + 1):
        raw = sequence[start : start + motif_len]
        window = pwm[start : start + motif_len]
        if raw == aligned_sequence:
            return window, f"window_direct:{start}", raw
        if complement(raw) == aligned_sequence:
            return (
                window[:, COMPLEMENT_INDEX],
                f"window_complement:{start}",
                complement(raw),
            )
        if raw[::-1] == aligned_sequence:
            return window[::-1], f"window_reverse:{start}", raw[::-1]
        if reverse_complement(raw) == aligned_sequence:
            return (
                window[::-1][:, COMPLEMENT_INDEX],
                f"window_reverse_complement:{start}",
                reverse_complement(raw),
            )
    return None


def select_deeppbs_pwm(
    sample_npz: Path,
    deeppbs_pred_npz: Path,
) -> tuple[np.ndarray, str, str]:
    with np.load(sample_npz, allow_pickle=False) as target:
        target_pwm = np.asarray(target["pwm_target"], dtype=np.float32)
        indices = np.asarray(target["slot_to_dna_index"], dtype=np.int64)
        aligned_sequence = str(target["alignment_sequence"])

    with np.load(deeppbs_pred_npz, allow_pickle=False) as pred:
        deeppbs_pwm = np.asarray(pred["P"], dtype=np.float32)
        deeppbs_sequence = one_hot_to_seq(np.asarray(pred["Seq"], dtype=np.float32))

    motif_len = int(target_pwm.shape[0])
    if len(aligned_sequence) != motif_len:
        raise ValueError(
            f"alignment_sequence length {len(aligned_sequence)} != PWM length {motif_len}"
        )

    candidate = _candidate_from_index(
        deeppbs_pwm,
        deeppbs_sequence,
        indices,
        aligned_sequence,
    )
    if candidate is None:
        candidate = _candidate_from_window(
            deeppbs_pwm,
            deeppbs_sequence,
            motif_len,
            aligned_sequence,
        )
    if candidate is None:
        raise ValueError(
            "could not align DeepPBS Seq to RBE alignment_sequence "
            f"target={aligned_sequence} deep_seq={deeppbs_sequence} "
            f"P_len={deeppbs_pwm.shape[0]} M={motif_len} idx={indices.tolist()}"
        )
    return candidate


def align_predictions(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    pred_dir = Path(args.deeppbs_pred_dir)
    out_dir = Path(args.out_dir)
    aligned_pred_dir = out_dir / "preds_aligned"
    aligned_pred_dir.mkdir(parents=True, exist_ok=True)

    mode_rows = ["sample\tmode\tsequence"]
    failure_rows = ["sample\treason"]
    aligned_samples = []

    for sample in read_manifest(manifest):
        source = pred_dir / f"{sample.stem}.npz_predict.npz"
        destination = aligned_pred_dir / f"{sample.stem}.pred.npz"
        try:
            pwm, mode, sequence = select_deeppbs_pwm(sample, source)
        except Exception as exc:
            failure_rows.append(f"{sample.stem}\t{exc}")
            continue

        np.savez_compressed(destination, P=pwm)
        aligned_samples.append(sample.resolve())
        mode_rows.append(f"{sample.stem}\t{mode}\t{sequence}")

    (out_dir / "aligned_manifest.txt").write_text(
        "".join(f"{sample}\n" for sample in aligned_samples)
    )
    (out_dir / "alignment_modes.tsv").write_text("\n".join(mode_rows) + "\n")
    (out_dir / "alignment_failures.tsv").write_text("\n".join(failure_rows) + "\n")

    n_failures = len(failure_rows) - 1
    print(f"aligned={len(aligned_samples)} failures={n_failures}")
    print(f"wrote {aligned_pred_dir}")
    print(f"wrote {out_dir / 'aligned_manifest.txt'}")
    print(f"wrote {out_dir / 'alignment_modes.tsv'}")
    print(f"wrote {out_dir / 'alignment_failures.tsv'}")
    if args.strict and n_failures:
        raise SystemExit(1)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align DeepPBS DNA-position predictions to RBE PWM slots so they can "
            "be evaluated with rbe.eval.evaluate_manifest."
        )
    )
    parser.add_argument("--manifest", required=True, help="RBE manifest with target npz files.")
    parser.add_argument(
        "--deeppbs-pred-dir",
        required=True,
        help="Directory containing DeepPBS '*.npz_predict.npz' prediction files.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any manifest sample cannot be aligned.",
    )
    return parser


def main() -> None:
    align_predictions(build_argparser().parse_args())


if __name__ == "__main__":
    main()
