# Broad TF Modality Ablation Runbook

本实验在完整 DeepPBS-derived canonical 数据上检验 ESM 与蛋白结构对 TF binding
specificity 的独立、可泛化增量。它与 ETS benchmark 分开，不使用 family prior。

## 1. 数据与泄漏约束

| Item | Definition |
|---|---|
| Source | 653 canonical protein-DNA complexes |
| Protein grouping | MMseqs2 30% identity, 80% bidirectional coverage |
| Label grouping | canonical PWM 数值 hash；不同 motif ID 的相同 PWM 也合并 |
| Split unit | sequence cluster 与 PWM hash 的 connected component |
| CV | 5-fold；1 test、1 validation、3 training folds |
| Selection | validation component-equal PWM MAE |
| Evaluation | test component-equal paired metrics |

component 不能跨 train/validation/test，因此近同源蛋白和完全相同 PWM target 都不会
泄漏。所有命令在仓库根目录执行。

```bash
cd ~/residue-base-energy
git switch broad-modality-ablation
git pull --ff-only
pip install -e .
```

## 2. 重建当前契约的 Canonical Cache

旧 `data/deeppbs_canonical` 缺少 `pwm_orientation`，禁止在读取端 fallback 或原地
修改。新建独立 cache：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_deeppbs_shared_cache.py \
  --source-root metadata/generated \
  --out-root data/deeppbs_broad_v1 \
  --device cuda
```

预期 `653` 个成功样本，且失败数为0。

## 3. 构建 Component-Disjoint Folds

MMseqs2 使用独立工具环境：

```bash
conda create -y -p ~/tools/mmseqs-env -c conda-forge -c bioconda mmseqs2

python scripts/prepare_cluster_benchmark.py \
  --cache-root data/deeppbs_broad_v1 \
  --source-root metadata/generated \
  --out-root data/broad_component_v1 \
  --mmseqs ~/tools/mmseqs-env/bin/mmseqs \
  --min-seq-id 0.3 \
  --coverage 0.8 \
  --folds 5 \
  --threads 16
```

检查 `protocol.json`、`component_table.tsv` 和 `fold_table.tsv`。只有每折 train、
validation、test 的 sequence cluster 与 PWM hash 均无交集后才能训练；准备脚本会
硬性执行这一验证。

## 4. 四组同容量模型

| Variant | ESM2 | Amino-acid identity | Protein geometry |
|---|---:|---:|---:|
| `full` | yes | yes | yes |
| `esm_only` | yes | yes | no |
| `structure_only` | no | yes | yes |
| `aa_only` | no | yes | no |

配置除 `use_esm/use_geometry` 外完全相同：

```bash
for SPEC in \
  "full configs/broad_full_v1.yaml" \
  "esm_only configs/broad_esm_only_v1.yaml" \
  "structure_only configs/broad_structure_only_v1.yaml" \
  "aa_only configs/broad_aa_only_v1.yaml"; do
  read -r VARIANT CONFIG <<< "$SPEC"
  for FOLD in {0..4}; do
    OUT="runs/broad_modality_v1/${VARIANT}/fold${FOLD}"
    mkdir -p "$OUT"
    python -m rbe.train \
      --manifest "data/broad_component_v1/folds/fold${FOLD}_train.txt" \
      --valid-manifest "data/broad_component_v1/folds/fold${FOLD}_valid.txt" \
      --valid-group-table data/broad_component_v1/sample_table.tsv \
      --valid-group-column component \
      --config "$CONFIG" \
      --out-dir "$OUT" \
      --device cuda \
      2>&1 | tee "$OUT/train.log"
  done
done
```

## 5. Test Predictions

```bash
for VARIANT in full esm_only structure_only aa_only; do
  for FOLD in {0..4}; do
    python -m rbe.eval.evaluate_manifest \
      --manifest "data/broad_component_v1/folds/fold${FOLD}_test.txt" \
      --pred-dir "runs/broad_modality_v1/${VARIANT}/fold${FOLD}/preds" \
      --checkpoint "runs/broad_modality_v1/${VARIANT}/fold${FOLD}/best.pt" \
      --device cuda \
      --overwrite-pred
  done
done
```

## 6. Component-Equal Evaluation

Full 参照检验条件增量：

```bash
python scripts/evaluate_cluster_methods.py \
  --benchmark-root data/broad_component_v1 \
  --method 'full=runs/broad_modality_v1/full/fold{fold}/preds' \
  --method 'esm_only=runs/broad_modality_v1/esm_only/fold{fold}/preds' \
  --method 'structure_only=runs/broad_modality_v1/structure_only/fold{fold}/preds' \
  --method 'aa_only=runs/broad_modality_v1/aa_only/fold{fold}/preds' \
  --reference-method full \
  --out-root runs/broad_modality_v1/evaluation/full_reference
```

AA-only 参照检验独立增量：

```bash
python scripts/evaluate_cluster_methods.py \
  --benchmark-root data/broad_component_v1 \
  --method 'aa_only=runs/broad_modality_v1/aa_only/fold{fold}/preds' \
  --method 'esm_only=runs/broad_modality_v1/esm_only/fold{fold}/preds' \
  --method 'structure_only=runs/broad_modality_v1/structure_only/fold{fold}/preds' \
  --method 'full=runs/broad_modality_v1/full/fold{fold}/preds' \
  --reference-method aa_only \
  --out-root runs/broad_modality_v1/evaluation/aa_reference
```

## 7. 结论规则

`mean_delta = method - reference`。MAE/KL 越低越好，IC-PCC/A-base AP 越高越好。

| Comparison | Question |
|---|---|
| `esm_only - full` | structure given ESM |
| `structure_only - full` | ESM given structure |
| `esm_only - aa_only` | ESM without geometry |
| `structure_only - aa_only` | structure without PLM |

只有效应方向正确、95% component bootstrap CI 不跨0且 paired sign-flip
`p <= 0.05`，才称为可泛化增量。component 数超过20时使用固定 seed 的100,000次
paired sign-flip Monte Carlo；CI 跨0时只能说证据不足。
