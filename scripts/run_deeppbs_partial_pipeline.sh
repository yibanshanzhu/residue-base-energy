#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-configs/dna_v1_contact.yaml}"
SOURCE_ROOT="${SOURCE_ROOT:-metadata/generated}"
DATA_ROOT="${DATA_ROOT:-data/deeppbs_canonical}"
RUN_ROOT="${RUN_ROOT:-runs/canonical_pwm}"

python scripts/prepare_deeppbs_shared_cache.py \
  --source-root "$SOURCE_ROOT" \
  --out-root "$DATA_ROOT" \
  --download-structures \
  --device "$DEVICE"

TRAIN_EXTRA=()
if [[ -n "${EPOCHS:-}" ]]; then
  TRAIN_EXTRA+=(--epochs "$EPOCHS")
fi

for fold in 0 1 2 3 4; do
  python -m rbe.train \
    --manifest "${DATA_ROOT}/manifests/train${fold}.txt" \
    --valid-manifest "${DATA_ROOT}/manifests/valid${fold}.txt" \
    --config "$CONFIG" \
    --out-dir "${RUN_ROOT}/fold${fold}" \
    --device "$DEVICE" \
    "${TRAIN_EXTRA[@]}"
done

python -m rbe.eval.predict_ensemble_manifest \
  --manifest "${DATA_ROOT}/manifests/id.txt" \
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
  --manifest "${DATA_ROOT}/manifests/id.txt" \
  --pred-dir "${RUN_ROOT}/id_ensemble/preds" \
  --summary-tsv "${RUN_ROOT}/id_ensemble/eval_summary.tsv" \
  --per-sample-tsv "${RUN_ROOT}/id_ensemble/eval_per_sample.tsv"

cat "${RUN_ROOT}/id_ensemble/eval_summary.tsv"
