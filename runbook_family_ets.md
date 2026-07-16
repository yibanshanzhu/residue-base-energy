# ETS 家族内机制 Benchmark Runbook

本流程检验两个问题：RBE 能否预测 **ETS 家族内部**的 specificity 差异，以及
residue-base supervision/配对是否真正贡献了预测能力。

## 1. 实验口径

| 项目 | 约束 |
|---|---|
| family | ETS |
| sample | 26 个单结合位点复合物 |
| group | 12 个 UniProt accession |
| target | 固定方向、固定 9 bp，`GGAA/GGAT` 位于 0-based slots `3:7` |
| split | leave-one-protein-out；同一 UniProt 的全部结构只在一个 split |
| checkpoint | 只按独立 validation UniProt 的 PWM MAE 选择 `best.pt` |
| 汇总 | 先对同一 UniProt 的结构求均值，再对 12 个 UniProt 等权求均值 |

每个 test UniProt 只能使用对应的 LOPO 模型。**不能 ensemble 12 个模型**，因为其他
fold 的模型见过当前 test UniProt，会造成训练测试泄漏。

所有命令均在仓库根目录执行：

```bash
cd ~/residue-base-energy
git switch family-mechanism-benchmark
git pull --ff-only
pip install -e .
```

## 2. 生成 ETS 数据与 Folds

复用已经生成好的 canonical cache；这里只做经过审计的方向变换和 9 bp 裁剪，不重新
计算结构或 ESM2 embedding。

```bash
python scripts/prepare_family_benchmark.py \
  --cache-root data/deeppbs_canonical \
  --spec resources/family_benchmarks/ets_v1/samples.tsv \
  --out-root data/family_ets_v1 \
  --family-name ETS \
  --version v1
```

预期输出：`samples=26 protein_groups=12`。检查：

```bash
wc -l data/family_ets_v1/folds/fold*_test.txt
column -t -s $'\t' data/family_ets_v1/sample_table.tsv
column -t -s $'\t' data/family_ets_v1/fold_table.tsv
```

## 3. 训练 Full RBE

```bash
for FOLD in {0..11}; do
  OUT="runs/ets_family_v1/full/fold${FOLD}"
  mkdir -p "$OUT"
  python -m rbe.train \
    --manifest "data/family_ets_v1/folds/fold${FOLD}_train.txt" \
    --valid-manifest "data/family_ets_v1/folds/fold${FOLD}_valid.txt" \
    --config configs/ets_family_v1.yaml \
    --out-dir "$OUT" \
    --device cuda \
    2>&1 | tee "$OUT/train.log"
done
```

## 4. 训练 PWM-only 对照

该对照使用完全相同的模型结构，只保留最终 PWM loss；contact、site 和 teacher-gate
监督权重均为 0，用于隔离结构监督的贡献。

```bash
for FOLD in {0..11}; do
  OUT="runs/ets_family_v1/pwm_only/fold${FOLD}"
  mkdir -p "$OUT"
  python -m rbe.train \
    --manifest "data/family_ets_v1/folds/fold${FOLD}_train.txt" \
    --valid-manifest "data/family_ets_v1/folds/fold${FOLD}_valid.txt" \
    --config configs/ets_family_pwm_only_v1.yaml \
    --out-dir "$OUT" \
    --device cuda \
    2>&1 | tee "$OUT/train.log"
done
```

## 5. 生成各 Fold 测试预测

```bash
for VARIANT in full pwm_only; do
  for FOLD in {0..11}; do
    python -m rbe.eval.evaluate_manifest \
      --manifest "data/family_ets_v1/folds/fold${FOLD}_test.txt" \
      --pred-dir "runs/ets_family_v1/${VARIANT}/fold${FOLD}/preds" \
      --checkpoint "runs/ets_family_v1/${VARIANT}/fold${FOLD}/best.pt" \
      --device cuda \
      --overwrite-pred
  done
done
```

## 6. 生成非神经网络 Baselines

`family_mean` 先在每个训练 UniProt 内求平均，再对训练 UniProt 等权平均；
`nearest_esm` 只从当前 fold 的训练组中选择 pooled-ESM2 cosine 最近邻。

```bash
python scripts/predict_family_baselines.py \
  --benchmark-root data/family_ets_v1 \
  --out-root runs/ets_family_v1/baselines
```

## 7. 生成机制消融

`uniform_gate` 保持每个 motif slot 的 gate 总量，但抹掉 residue 定位；
`shuffled_energy` 保持 gate 和 energy 的边际分布，但打乱 residue 对应关系。

```bash
python scripts/ablate_family_mechanism.py \
  --benchmark-root data/family_ets_v1 \
  --prediction-root runs/ets_family_v1/full \
  --out-root runs/ets_family_v1/mechanism_ablations
```

## 8. UniProt 等权评估

模板中的 `{fold}` 必须用单引号保护，避免 shell 展开。

```bash
python scripts/evaluate_family_methods.py \
  --benchmark-root data/family_ets_v1 \
  --method 'full=runs/ets_family_v1/full/fold{fold}/preds' \
  --method 'pwm_only=runs/ets_family_v1/pwm_only/fold{fold}/preds' \
  --method 'nearest_esm=runs/ets_family_v1/baselines/fold{fold}/nearest_esm/preds' \
  --method 'family_mean=runs/ets_family_v1/baselines/fold{fold}/family_mean/preds' \
  --method 'uniform_gate=runs/ets_family_v1/mechanism_ablations/fold{fold}/uniform_gate/preds' \
  --method 'shuffled_energy=runs/ets_family_v1/mechanism_ablations/fold{fold}/shuffled_energy/preds' \
  --reference-method full \
  --out-root runs/ets_family_v1/evaluation
```

| 输出 | 含义 |
|---|---|
| `per_sample.tsv` | 每个结构的原始指标 |
| `per_group.tsv` | 同一 UniProt 内先平均后的指标 |
| `summary.tsv` | 12 个 UniProt 等权汇总 |
| `paired_pwm_mae.tsv` | 每个方法相对 full RBE 的组配对差值 |

`paired_pwm_mae.tsv` 中 `mean_delta = method MAE - full MAE`；正值表示 full 更好。
结论必须分别回答：full 是否优于 `family_mean/nearest_esm`、是否优于 `pwm_only`，以及
两种 residue-base 配对消融是否使 MAE 上升。
