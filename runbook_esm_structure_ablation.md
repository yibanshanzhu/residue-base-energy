# ESM 与结构信息消融 Runbook

本实验回答一个问题：蛋白质结构或 ESM 表示是否为 ETS binding specificity
提供独立、可泛化的增量信息。评价单位是 held-out UniProt，不以单个复合物数量扩大
样本量。

## 1. 实验定义

| Variant | ESM2 | 氨基酸身份 | 蛋白结构 | 训练目标 |
|---|---:|---:|---:|---|
| `full` | 是 | 是 | 是 | 完整 RBE losses |
| `esm_only` | 是 | 是 | 否 | 与 full 相同 |
| `structure_only` | 否 | 是 | 是 | 与 full 相同 |
| `family_mean` | 否 | 否 | 否 | 无训练；training-fold PWM prior |

`esm_only` 不读取坐标、结构边或边特征；`structure_only` 不读取 ESM2。两者与
`full` 使用同一 fold、family prior、loss、优化器、epoch 数、validation UniProt
选模和 residual scale 校准。氨基酸身份是两种模态共有的最低限度残基标识，因此
`structure_only` 表示“无 PLM 的带类型结构模型”，不是无化学身份的点云。

三个神经网络的参数形状和数量完全相同。关闭 ESM 时以同形状零特征替代；关闭结构
时将半径归零、结构边置空，同时保留 EGNN 的逐节点变换。因此比较只改变可见信息，
不同时改变网络深度或容量。

所有命令在仓库根目录执行：

```bash
cd ~/residue-base-energy
git switch esm-structure-ablation
git pull --ff-only
pip install -e .
```

复用 `data/family_ets_v1` 和已经完成的 `runs/ets_family_v1/full`。本分支的 full
默认数据流与原实验相同，不需要重新训练。

## 2. 训练两个模态消融

```bash
for SPEC in \
  "esm_only configs/ets_family_esm_only_v1.yaml" \
  "structure_only configs/ets_family_structure_only_v1.yaml"; do
  read -r VARIANT CONFIG <<< "$SPEC"
  for FOLD in {0..11}; do
    OUT="runs/ets_family_v1/${VARIANT}/fold${FOLD}"
    mkdir -p "$OUT"
    python -m rbe.train \
      --manifest "data/family_ets_v1/folds/fold${FOLD}_train.txt" \
      --valid-manifest "data/family_ets_v1/folds/fold${FOLD}_valid.txt" \
      --config "$CONFIG" \
      --out-dir "$OUT" \
      --device cuda \
      2>&1 | tee "$OUT/train.log"
  done
done
```

检查24个训练是否完成：

```bash
find runs/ets_family_v1/{esm_only,structure_only} -name best.pt | wc -l
find runs/ets_family_v1/{esm_only,structure_only} -name train.log \
  -exec tail -n 1 {} \;
```

第一条应输出 `24`，第二条的每行应以 `100` 开头。

## 3. 生成 Validation 与 Test 预测

```bash
for VARIANT in esm_only structure_only; do
  for FOLD in {0..11}; do
    python -m rbe.eval.evaluate_manifest \
      --manifest "data/family_ets_v1/folds/fold${FOLD}_valid.txt" \
      --pred-dir "runs/ets_family_v1/${VARIANT}/fold${FOLD}/valid_preds" \
      --checkpoint "runs/ets_family_v1/${VARIANT}/fold${FOLD}/best.pt" \
      --device cuda \
      --overwrite-pred

    python -m rbe.eval.evaluate_manifest \
      --manifest "data/family_ets_v1/folds/fold${FOLD}_test.txt" \
      --pred-dir "runs/ets_family_v1/${VARIANT}/fold${FOLD}/preds" \
      --checkpoint "runs/ets_family_v1/${VARIANT}/fold${FOLD}/best.pt" \
      --device cuda \
      --overwrite-pred
  done
done
```

## 4. 只用 Validation UniProt 校准

```bash
for VARIANT in esm_only structure_only; do
  python scripts/calibrate_family_residual.py \
    --benchmark-root data/family_ets_v1 \
    --prediction-root "runs/ets_family_v1/${VARIANT}" \
    --out-root "runs/ets_family_v1/${VARIANT}_calibrated"
done
```

每个 test UniProt 只能使用其 fold 的 validation-selected scale；禁止根据 test
结果选择 scale 或 checkpoint。

## 5. 配对评估

先确保原实验 baseline 已存在；若不存在则生成：

```bash
python scripts/predict_family_baselines.py \
  --benchmark-root data/family_ets_v1 \
  --out-root runs/ets_family_v1/baselines
```

相对 full 评估条件增量：

```bash
python scripts/evaluate_family_methods.py \
  --benchmark-root data/family_ets_v1 \
  --method 'full=runs/ets_family_v1/full_calibrated/fold{fold}/preds' \
  --method 'esm_only=runs/ets_family_v1/esm_only_calibrated/fold{fold}/preds' \
  --method 'structure_only=runs/ets_family_v1/structure_only_calibrated/fold{fold}/preds' \
  --method 'family_mean=runs/ets_family_v1/baselines/fold{fold}/family_mean/preds' \
  --method 'nearest_esm=runs/ets_family_v1/baselines/fold{fold}/nearest_esm/preds' \
  --reference-method full \
  --out-root runs/ets_family_v1/modality_evaluation/full_reference
```

再以 family mean 为参照，检验每种模态能否独立超过 prior：

```bash
python scripts/evaluate_family_methods.py \
  --benchmark-root data/family_ets_v1 \
  --method 'family_mean=runs/ets_family_v1/baselines/fold{fold}/family_mean/preds' \
  --method 'esm_only=runs/ets_family_v1/esm_only_calibrated/fold{fold}/preds' \
  --method 'structure_only=runs/ets_family_v1/structure_only_calibrated/fold{fold}/preds' \
  --method 'full=runs/ets_family_v1/full_calibrated/fold{fold}/preds' \
  --reference-method family_mean \
  --out-root runs/ets_family_v1/modality_evaluation/prior_reference
```

## 6. 结论规则

`paired_pwm_mae.tsv` 定义 `mean_delta = method MAE - reference MAE`。

| 比较 | 增量问题 | 支持增量的方向 |
|---|---|---|
| `esm_only - full` | 给定 ESM 后，结构是否有增量 | full-reference 中 `mean_delta > 0` |
| `structure_only - full` | 给定结构后，ESM 是否有增量 | full-reference 中 `mean_delta > 0` |
| `structure_only - family_mean` | 无 PLM 时结构是否独立有效 | prior-reference 中 `mean_delta < 0` |
| `esm_only - family_mean` | 无结构时 ESM 是否独立有效 | prior-reference 中 `mean_delta < 0` |

`paired_metrics.tsv` 对 MAE、KL、IC-PCC 和可用模型的 A-base AP 分别给出配对
统计。只有效应方向正确、95% paired bootstrap CI 不跨0，并且 exact paired
sign-flip `p <= 0.05`，才表述为“在该 ETS benchmark 上提供可泛化增量”。CI
跨0时应表述为“当前12个 UniProt 的证据不足”，不能表述为该模态无效。

`A-base AP` 用于解释结构定位机制，不作为 specificity 增量成立的替代证据。
