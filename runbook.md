# Canonical PWM Orientation Runbook

本 runbook 用于在服务器上完成未裁剪 PWM 数据准备、canonical orientation 五折训练、五模型 ensemble 和评估。

评估主指标为：每个 sample 在完整 PWM 上计算 position-wise L1 mean，再对 samples 求均值。

## 1. 更新代码与环境

```bash
cd /path/to/residue-base-energy
git fetch origin
git switch canonical-pwm-orientation
git pull --ff-only

conda activate rbe_gpu
pip install -e .
```

## 2. 下载未裁剪 PWM

已经下载完成时跳过本节。

```bash
python scripts/download_motif_sources.py \
  --out-root resources/motif_sources \
  --motif-index resources/motif_sources/motif_index.tsv

test -s resources/motif_sources/motif_index.tsv
```

## 3. 生成五折 Source Manifests

```bash
mkdir -p metadata/generated data/raw/structures

for SPLIT in train{0..4} valid{0..4} id; do
  python scripts/import_deeppbs_source_manifest.py \
    --fold-file "${SPLIT}.txt" \
    --output "metadata/generated/deeppbs_${SPLIT}_sources.tsv" \
    --structure-format mmcif \
    --structure-cache-dir "$PWD/data/raw/structures" \
    --motif-index resources/motif_sources/motif_index.tsv
done
```

## 4. 构建 Canonical Cache

使用独立目录，不能复用旧的 untrimmed/masked cache。

```bash
python scripts/prepare_deeppbs_shared_cache.py \
  --source-root metadata/generated \
  --out-root data/deeppbs_canonical \
  --download-structures \
  --device cuda
```

验证缓存契约：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from rbe.data.pwm import canonicalize_pwm

root = Path("data/deeppbs_canonical")
samples = [
    Path(line.strip())
    for line in (root / "processed_manifest.txt").read_text().splitlines()
    if line.strip()
]
flipped = 0
for sample in samples:
    with np.load(sample, allow_pickle=False) as data:
        pwm = data["pwm_target"]
        assert "canonical_reverse_complement" in data.files, sample
        assert not canonicalize_pwm(pwm)[1], sample
        m = pwm.shape[0]
        assert data["pwm_mask"].shape == (m,), sample
        assert data["slot_to_dna_index"].shape == (m,), sample
        for key in ("A_base_label", "A_base_mask", "A_backbone_label", "A_contact_label"):
            assert data[key].shape[1] == m, (sample, key)
        flipped += int(data["canonical_reverse_complement"])
print(f"samples={len(samples)} canonical_rc_applied={flipped}")
PY

wc -l data/deeppbs_canonical/manifests/*.txt
tail -n +2 data/deeppbs_canonical/failed.tsv | wc -l
```

## 5. 训练五个模型

`best.pt` 按对应 validation fold 的 canonical full per-sample PWM MAE 选择。

```bash
mkdir -p runs/canonical_pwm

for FOLD in {0..4}; do
  mkdir -p "runs/canonical_pwm/fold${FOLD}"
  python -m rbe.train \
    --manifest "data/deeppbs_canonical/manifests/train${FOLD}.txt" \
    --valid-manifest "data/deeppbs_canonical/manifests/valid${FOLD}.txt" \
    --config configs/dna_v1.yaml \
    --out-dir "runs/canonical_pwm/fold${FOLD}" \
    --device cuda \
    2>&1 | tee "runs/canonical_pwm/fold${FOLD}/train.log"
done

ls -lh runs/canonical_pwm/fold{0..4}/best.pt
```

## 6. 评估各折 Validation

```bash
for FOLD in {0..4}; do
  python -m rbe.eval.evaluate_manifest \
    --manifest "data/deeppbs_canonical/manifests/valid${FOLD}.txt" \
    --pred-dir "runs/canonical_pwm/fold${FOLD}/valid_preds" \
    --checkpoint "runs/canonical_pwm/fold${FOLD}/best.pt" \
    --device cuda \
    --overwrite-pred \
    --summary-json "runs/canonical_pwm/fold${FOLD}/valid_summary.json"
done
```

## 7. 五模型 Ensemble ID 评估

```bash
python -m rbe.eval.predict_ensemble_manifest \
  --manifest data/deeppbs_canonical/manifests/id.txt \
  --pred-dir runs/canonical_pwm/id_ensemble/preds \
  --checkpoints \
    runs/canonical_pwm/fold0/best.pt \
    runs/canonical_pwm/fold1/best.pt \
    runs/canonical_pwm/fold2/best.pt \
    runs/canonical_pwm/fold3/best.pt \
    runs/canonical_pwm/fold4/best.pt \
  --device cuda \
  --overwrite-pred

python -m rbe.eval.evaluate_manifest \
  --manifest data/deeppbs_canonical/manifests/id.txt \
  --pred-dir runs/canonical_pwm/id_ensemble/preds \
  --per-sample-tsv runs/canonical_pwm/id_ensemble/eval_per_sample.tsv \
  --summary-tsv runs/canonical_pwm/id_ensemble/eval_summary.tsv \
  --summary-json runs/canonical_pwm/id_ensemble/eval_summary.json
```

查看主指标：

```bash
awk -F '\t' 'NR == 1 || $1 == "pwm_mae"' \
  runs/canonical_pwm/id_ensemble/eval_summary.tsv
```

`pwm_mae.n` 必须等于 ID sample 数。`pwm_mask` 不参与 PWM MAE，只用于结构 contact/map 的监督与评估。
