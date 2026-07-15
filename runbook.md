# Masked PWM Per-Sample Runbook

本 runbook 用于在服务器上完成未裁剪 PWM 数据准备、五折训练、ensemble prediction 和评估。当前 `pwm_mae` 的口径是：每个 sample 只在 `pwm_mask=1` 的结构可见 columns 上计算 MAE，再对 samples 求均值。

## 1. 更新代码与环境

```bash
cd /path/to/residue-base-energy
git fetch origin
git switch masked-pwm-per-sample
git pull --ff-only

conda env create -f environment.gpu.yml
conda activate rbe_gpu
pip install -e .
```

环境已经创建时，只需执行 `conda activate rbe_gpu` 和 `pip install -e .`。

## 2. 下载未裁剪 PWM

```bash
python scripts/download_motif_sources.py \
  --out-root resources/motif_sources \
  --motif-index resources/motif_sources/motif_index.tsv
```

检查结果：

```bash
test -s resources/motif_sources/motif_index.tsv
head resources/motif_sources/motif_index.tsv
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

## 4. 构建 Training Cache

```bash
python scripts/prepare_deeppbs_shared_cache.py \
  --source-root metadata/generated \
  --out-root data/deeppbs_untrimmed \
  --download-structures \
  --device cuda
```

该命令按 `sample_id` 合并 11 个 source manifests。结构解析、ESM2 embedding 和 labels 对每个 unique sample 只计算一次；fold membership 单独写入 `data/deeppbs_untrimmed/manifests/*.txt`。

检查成功和失败数量：

```bash
wc -l data/deeppbs_untrimmed/processed_manifest.txt
tail -n +2 data/deeppbs_untrimmed/failed.tsv | wc -l
wc -l data/deeppbs_untrimmed/manifests/*.txt
```

验证所有评估样本都有至少一个有效 PWM column：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

for split in ("valid0", "valid1", "valid2", "valid3", "valid4", "id"):
    manifest = Path(f"data/deeppbs_untrimmed/manifests/{split}.txt")
    samples = [Path(line.strip()) for line in manifest.read_text().splitlines() if line.strip()]
    valid = 0
    total = 0
    for sample in samples:
        with np.load(sample, allow_pickle=False) as data:
            mask = data["pwm_mask"]
            assert mask.shape == (data["pwm_target"].shape[0],), sample
            assert np.any(mask == 1), sample
            valid += int((mask == 1).sum())
            total += mask.size
    print(split, f"samples={len(samples)}", f"valid_columns={valid}/{total}")
PY
```

## 5. 训练五个模型

```bash
mkdir -p runs/masked_pwm_per_sample

for FOLD in {0..4}; do
  mkdir -p "runs/masked_pwm_per_sample/fold${FOLD}"
  python -m rbe.train \
    --manifest "data/deeppbs_untrimmed/manifests/train${FOLD}.txt" \
    --config configs/dna_v1.yaml \
    --out-dir "runs/masked_pwm_per_sample/fold${FOLD}" \
    --device cuda \
    2>&1 | tee "runs/masked_pwm_per_sample/fold${FOLD}/train.log"
done
```

训练完成后应存在：

```bash
ls -lh runs/masked_pwm_per_sample/fold{0..4}/best.pt
```

## 6. 评估各折 Validation

```bash
for FOLD in {0..4}; do
  python -m rbe.eval.evaluate_manifest \
    --manifest "data/deeppbs_untrimmed/manifests/valid${FOLD}.txt" \
    --pred-dir "runs/masked_pwm_per_sample/fold${FOLD}/valid_preds" \
    --checkpoint "runs/masked_pwm_per_sample/fold${FOLD}/best.pt" \
    --device cuda \
    --overwrite-pred \
    --summary-json "runs/masked_pwm_per_sample/fold${FOLD}/valid_summary.json"
done
```

## 7. 五模型 Ensemble ID 评估

先对五个模型的 prediction arrays 求均值：

```bash
python -m rbe.eval.predict_ensemble_manifest \
  --manifest data/deeppbs_untrimmed/manifests/id.txt \
  --pred-dir runs/masked_pwm_per_sample/id_ensemble/preds \
  --checkpoints \
    runs/masked_pwm_per_sample/fold0/best.pt \
    runs/masked_pwm_per_sample/fold1/best.pt \
    runs/masked_pwm_per_sample/fold2/best.pt \
    runs/masked_pwm_per_sample/fold3/best.pt \
    runs/masked_pwm_per_sample/fold4/best.pt \
  --device cuda \
  --overwrite-pred
```

再用 ensemble predictions 计算 masked per-sample MAE：

```bash
python -m rbe.eval.evaluate_manifest \
  --manifest data/deeppbs_untrimmed/manifests/id.txt \
  --pred-dir runs/masked_pwm_per_sample/id_ensemble/preds \
  --per-sample-tsv runs/masked_pwm_per_sample/id_ensemble/eval_per_sample.tsv \
  --summary-tsv runs/masked_pwm_per_sample/id_ensemble/eval_summary.tsv \
  --summary-json runs/masked_pwm_per_sample/id_ensemble/eval_summary.json
```

关键输出：

| 文件 | 内容 |
|---|---|
| `fold*/valid_preds/eval_summary.tsv` | 五个模型各自在对应 validation fold 上的结果 |
| `id_ensemble/eval_per_sample.tsv` | ID 集合每个 sample 的 ensemble masked `pwm_mae` |
| `id_ensemble/eval_summary.tsv` | ID benchmark 的 sample mean/std，`pwm_mae.n` 等于 sample 数 |
| `id_ensemble/eval_summary.json` | ID benchmark 完整评估记录 |

查看 MAE：

```bash
for FOLD in {0..4}; do
  echo "=== valid${FOLD} ==="
  awk -F '\t' 'NR == 1 || $1 == "pwm_mae"' \
    "runs/masked_pwm_per_sample/fold${FOLD}/valid_preds/eval_summary.tsv"
done

echo "=== id ensemble ==="
awk -F '\t' 'NR == 1 || $1 == "pwm_mae"' \
  runs/masked_pwm_per_sample/id_ensemble/eval_summary.tsv
```
