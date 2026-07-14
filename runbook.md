# Masked PWM Per-Sample Runbook

本 runbook 用于在服务器上完成未裁剪 PWM 数据准备、训练和评估。当前 `pwm_mae` 的口径是：每个 sample 只在 `pwm_mask=1` 的结构可见 columns 上计算 MAE，再对 samples 求均值。

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

## 3. 生成 Fold 0 Source Manifests

```bash
mkdir -p metadata/generated data/raw/structures

for SPLIT in train0 valid0 id; do
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
for SPLIT in train0 valid0 id; do
  python scripts/prepare_source_manifest.py \
    --source-manifest "metadata/generated/deeppbs_${SPLIT}_sources.tsv" \
    --out-root "data/deeppbs_untrimmed/${SPLIT}" \
    --download-structures \
    --device cuda
done
```

检查成功和失败数量：

```bash
for SPLIT in train0 valid0 id; do
  echo "=== ${SPLIT} ==="
  wc -l "data/deeppbs_untrimmed/${SPLIT}/processed_manifest.txt"
  tail -n +2 "data/deeppbs_untrimmed/${SPLIT}/failed.tsv" | wc -l
done
```

验证所有评估样本都有至少一个有效 PWM column：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

for split in ("valid0", "id"):
    manifest = Path(f"data/deeppbs_untrimmed/{split}/processed_manifest.txt")
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

## 5. 训练

```bash
mkdir -p runs/masked_pwm_per_sample_fold0

python -m rbe.train \
  --manifest data/deeppbs_untrimmed/train0/processed_manifest.txt \
  --config configs/dna_v1.yaml \
  --out-dir runs/masked_pwm_per_sample_fold0 \
  --device cuda \
  2>&1 | tee runs/masked_pwm_per_sample_fold0/train.log
```

训练完成后应存在：

```bash
ls -lh runs/masked_pwm_per_sample_fold0/best.pt \
       runs/masked_pwm_per_sample_fold0/last.pt
```

## 6. 评估

```bash
for SPLIT in valid0 id; do
  python -m rbe.eval.evaluate_manifest \
    --manifest "data/deeppbs_untrimmed/${SPLIT}/processed_manifest.txt" \
    --pred-dir "runs/masked_pwm_per_sample_fold0/${SPLIT}_preds" \
    --checkpoint runs/masked_pwm_per_sample_fold0/best.pt \
    --device cuda \
    --overwrite-pred \
    --summary-json "runs/masked_pwm_per_sample_fold0/${SPLIT}_summary.json"
done
```

关键输出：

| 文件 | 内容 |
|---|---|
| `${SPLIT}_preds/eval_per_sample.tsv` | 每个 sample 的 masked `pwm_mae` 和其他指标 |
| `${SPLIT}_preds/eval_summary.tsv` | 对 sample 指标求 mean/std，`pwm_mae.n` 应等于 sample 数 |
| `${SPLIT}_summary.json` | 完整评估记录 |

查看 MAE：

```bash
for SPLIT in valid0 id; do
  echo "=== ${SPLIT} ==="
  awk -F '\t' 'NR == 1 || $1 == "pwm_mae"' \
    "runs/masked_pwm_per_sample_fold0/${SPLIT}_preds/eval_summary.tsv"
done
```
