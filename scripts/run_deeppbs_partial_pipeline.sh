#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-configs/dna_v1_contact.yaml}"
DATA_ROOT="${DATA_ROOT:-data/deeppbs_partial}"
RUN_ROOT="${RUN_ROOT:-runs/deeppbs_partial}"

SPLITS=(train0 train1 train2 train3 train4 valid0 valid1 valid2 valid3 valid4 id)

for split in "${SPLITS[@]}"; do
  python scripts/prepare_source_manifest.py \
    --source-manifest "metadata/deeppbs/${split}_sources.tsv" \
    --out-root "${DATA_ROOT}/${split}" \
    --download-structures \
    --alignment-score deeppbs_ic_pcc \
    --alignment-contact-policy require_contact \
    --min-contact-pairs 1 \
    --min-site-residues 1 \
    --device "$DEVICE"
done

TRAIN_EXTRA=()
if [[ -n "${EPOCHS:-}" ]]; then
  TRAIN_EXTRA+=(--epochs "$EPOCHS")
fi

for fold in 0 1 2 3 4; do
  python -m rbe.train \
    --manifest "${DATA_ROOT}/train${fold}/processed_manifest.txt" \
    --config "$CONFIG" \
    --out-dir "${RUN_ROOT}/fold${fold}" \
    --device "$DEVICE" \
    "${TRAIN_EXTRA[@]}"
done

python -m rbe.eval.predict_ensemble_manifest \
  --manifest "${DATA_ROOT}/id/processed_manifest.txt" \
  --pred-dir "${RUN_ROOT}/id_ensemble/preds" \
  --checkpoints \
    "${RUN_ROOT}/fold0/best.pt" \
    "${RUN_ROOT}/fold1/best.pt" \
    "${RUN_ROOT}/fold2/best.pt" \
    "${RUN_ROOT}/fold3/best.pt" \
    "${RUN_ROOT}/fold4/best.pt" \
  --device "$DEVICE" \
  --overwrite-pred

python -m rbe.eval.evaluate_manifest \
  --manifest "${DATA_ROOT}/id/processed_manifest.txt" \
  --pred-dir "${RUN_ROOT}/id_ensemble/preds" \
  --summary-tsv "${RUN_ROOT}/id_ensemble/eval_summary.tsv" \
  --per-sample-tsv "${RUN_ROOT}/id_ensemble/eval_per_sample.tsv"

cat "${RUN_ROOT}/id_ensemble/eval_summary.tsv"
