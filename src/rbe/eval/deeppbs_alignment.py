from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rbe.data.pwm import canonicalize_pwm
from rbe.eval.io import read_manifest


BASES = "ACGT"
COMPLEMENT_INDEX = np.asarray([3, 2, 1, 0], dtype=np.int64)
COMPLEMENT_TRANS = str.maketrans("ACGT", "TGCA")


@dataclass(frozen=True)
class DeepPBSAlignmentResult:
    aligned_dir: Path
    aligned_manifest: Path
    mode_table: Path
    failure_table: Path
    aligned_count: int
    failure_count: int


def align_deeppbs_predictions(
    manifest: str | Path,
    deeppbs_pred_dir: str | Path,
    out_dir: str | Path,
) -> DeepPBSAlignmentResult:
    pred_dir = Path(deeppbs_pred_dir)
    output_root = Path(out_dir)
    aligned_pred_dir = output_root / "preds_aligned"
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

        with np.load(sample, allow_pickle=False) as target:
            orientation = str(target["pwm_orientation"])
        canonical_rc = False
        if orientation == "canonical":
            pwm, canonical_rc = canonicalize_pwm(pwm)
        np.savez_compressed(
            destination,
            P=pwm,
            canonical_reverse_complement=np.asarray(canonical_rc, dtype=bool),
            pwm_orientation=np.asarray(orientation),
        )
        aligned_samples.append(sample.resolve())
        mode_rows.append(f"{sample.stem}\t{mode}\t{sequence}")

    aligned_manifest = output_root / "aligned_manifest.txt"
    mode_table = output_root / "alignment_modes.tsv"
    failure_table = output_root / "alignment_failures.tsv"
    aligned_manifest.write_text("".join(f"{sample}\n" for sample in aligned_samples))
    mode_table.write_text("\n".join(mode_rows) + "\n")
    failure_table.write_text("\n".join(failure_rows) + "\n")

    return DeepPBSAlignmentResult(
        aligned_dir=aligned_pred_dir,
        aligned_manifest=aligned_manifest,
        mode_table=mode_table,
        failure_table=failure_table,
        aligned_count=len(aligned_samples),
        failure_count=len(failure_rows) - 1,
    )


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


def one_hot_to_seq(one_hot: np.ndarray) -> str:
    return "".join(BASES[int(idx)] for idx in one_hot.argmax(axis=1))


def complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT_TRANS)


def reverse_complement(sequence: str) -> str:
    return complement(sequence)[::-1]


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
