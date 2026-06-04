# Benchmark 欠缺与解决计划

## 现在先别绕晕

截至 2026-06-02，本地已经核对到：

| 阶段 | 状态 | 本地产物 |
|---|---|---|
| DeepPBS `id.txt` prepare | 已完成 | `data/deeppbs_id_contactalign/train_manifest.txt`，112 条 |
| prepare 失败记录 | 已完成 | `data/deeppbs_id_contactalign/failed.tsv`，18 条 |
| RBE 5-fold training | 已完成 | `runs/deeppbs_fold{0..4}_contactalign/best.pt` |
| RBE 5-fold ensemble prediction | 已完成 | `runs/deeppbs_id_ensemble/preds/*.pred.npz`，112 个 |
| RBE benchmark evaluation | 已完成 | `runs/deeppbs_id_ensemble/preds/eval_summary.tsv` |

所以现在不要再卡在 Step 1-4。它们已经是 **复现实验记录**，不是当前待办。

真正剩下的硬缺口只有一个：

```text
DeepPBS 在同一批 112 个 contact-valid samples 上 rerun，
然后用同一套 PWM metrics 和 RBE 结果比较。
```

如果只是写阶段报告或组会汇报，可以先停在这个结论：

```text
RBE 在 DeepPBS independent benchmark 的 contact-valid subset 上表现出有希望的 PWM 和 protein-site 信号；
但由于还没有 DeepPBS same-subset rerun，不能声称超过 DeepPBS。
```

如果要冲严格比较，下一步不是再跑 RBE，而是：

```bash
mkdir -p reports
sed 's#.*/##' data/deeppbs_id_contactalign/train_manifest.txt \
  > reports/deeppbs_id_contactvalid_entries.txt
```

然后用这个 112 条列表去约束 DeepPBS rerun，并把 DeepPBS 输出转换到 `rbe.eval.evaluate_pwm` 或 `rbe.eval.evaluate_manifest` 能读的同一指标口径。

## 当前结论先收紧

现在的结果说明模型有信号，但还不能宣称超过 DeepPBS、EquiPNAS 或 EquiPPIS。

原因不是数值一定差，而是比较条件还没有对齐。

| 对象 | 我们现在有什么 | 还缺什么 |
|---|---|---|
| DeepPBS | `id.txt` contact-valid subset 上的 RBE 5-fold ensemble 指标 | DeepPBS 在同一 112 条 subset 上 rerun |
| EquiPNAS | `id.txt` contact-valid subset 上的 protein-site AP/MCC/F1 | EquiPNAS 在同一批样本、同一套 site label、同一套评估脚本上 rerun |
| EquiPPIS | 只能作为 EGNN 思路参考 | 任务是 PPI，不是 protein-DNA，不适合直接数值比较 |

## 已完成进展

### DeepPBS independent benchmark 初步结果

2026-05-29 已完成：

```text
DeepPBS id.txt -> contact-valid prepare -> 5-fold RBE ensemble -> evaluate_manifest
```

原始 `id.txt` 有 130 条，当前 RBE contact-valid subset 成功评估 112 条。

`id.txt` 虽然也位于 DeepPBS 的 `run/folds/` 目录下，但它不是 `train0-4/valid0-4` cross-validation 的一部分，而是 independent benchmark。`train0 + valid0` 基本覆盖 cross-validation 全量样本；`id.txt` 与这部分样本不重叠。

| 项 | 数值 |
|---|---:|
| benchmark 原始条目 | 130 |
| RBE contact-valid samples | 112 |
| dropped / failed samples | 18 |

PWM 结果：

| 指标 | mean | std | n |
|---|---:|---:|---:|
| `pwm_mae` | 0.580072 | 0.435180 | 112 |
| `pwm_kl` | 0.525056 | 0.686145 | 112 |
| `pwm_ic_pcc` | 0.713164 | 0.388330 | 112 |
| `pwm_rc_aware_kl` | 0.377593 | 0.579355 | 112 |

注意：`pwm_mae` 已更新为 DeepPBS-style position L1 口径：

```text
per-position MAE = |A误差| + |C误差| + |G误差| + |T误差|
pwm_mae = mean(per-position MAE)
```

Residue-base/contact 解释性结果：

| 指标 | mean | std | n |
|---|---:|---:|---:|
| `A_base_ap` | 0.498321 | 0.332448 | 112 |
| `A_backbone_ap` | 0.388026 | 0.334698 | 112 |
| `A_contact_ap` | 0.425036 | 0.324541 | 112 |
| `A_contact_top_l_precision` | 0.394395 | 0.332070 | 112 |

Protein-site 结果：

| 指标 | mean | std | n |
|---|---:|---:|---:|
| `site_ap` | 0.776989 | 0.216148 | 112 |
| `site_mcc` | 0.650235 | 0.280367 | 112 |
| `site_f1` | 0.664596 | 0.271775 | 112 |
| `site_global_ap_diagnostic` | 0.746427 | 0.000000 | 15889 |

当前解读：

| 结论 | 说明 |
|---|---|
| PWM 已经接近 DeepPBS 量级 | DeepPBS-style `pwm_mae=0.580`，接近 DeepPBS official / reproduced benchmark 的 `0.48-0.54` 量级 |
| site 结果也有竞争力 | `site_ap=0.777`，`site_mcc=0.650` |
| A map 有解释性信号 | `A_contact_ap=0.425`，但还不是最终强项 |
| 仍不能直接宣称超过 DeepPBS | RBE 是 112 条 contact-valid subset；DeepPBS official 是 full 130，且还缺 DeepPBS same-subset rerun |

## 当前最大欠缺

### 1. 已用 DeepPBS independent benchmark，但还没 rerun DeepPBS same-subset

DeepPBS 正式 benchmark 不是 `valid0.txt`，而是：

```text
resources/deeppbs_curated/folds/id.txt
```

这个文件对应 DeepPBS 论文里的 independent benchmark，原始条目数是 130。

我们现在已经完成：

```text
id.txt -> RBE contact-valid subset -> RBE 5-fold ensemble
```

但还没有完成最严格的同子集比较：

```text
同一批 112 个 contact-valid samples
DeepPBS rerun
RBE ensemble
同一套 PWM metrics
```

所以现在可以说 RBE 在 DeepPBS independent benchmark 的 contact-valid subset 上表现很好，但还不能说严格超过 DeepPBS。

### 2. 已做 5-fold ensemble，但还要固定最终报告口径

DeepPBS benchmark 报告用的是 5-fold ensemble。

当前 RBE 已按同样思路完成 ensemble：

```text
fold0 model
fold1 model
fold2 model
fold3 model
fold4 model
        ↓
同一个 benchmark sample 分别预测
        ↓
PWM / A / site_prob 逐元素平均
        ↓
再评估
```

### 3. 我们的可用 benchmark 子集可能少于 DeepPBS 原始 130 条

RBE 需要生成 residue-base contact labels：

```text
A_base(i,j)
A_backbone(i,j)
A_contact(i,j)
site_label(i)
```

所以我们比 DeepPBS 多了结构监督要求。

有些样本会因为以下原因不能 prepare：

| 类型 | 例子 | 原因 |
|---|---|---|
| DNA 原子缺失 | `5wc9_E_MA0784.2.jaspar` | 某个 DNA residue 没有 base heavy atoms |
| DNA 片段太短 | `2l1g_A_THAP1_HUMAN.H11MO.0.C` | PDB DNA 长度小于 PWM 长度 |
| contact 不成立 | motif window 没有 protein contact | 无法定义有效 `A(i,j)` 监督 |

这些不能强行当 negative sample，因为：

```text
没有原子 != 没有接触
无法映射 != 负样本
```

最终论文或报告里必须写成：

```text
DeepPBS independent benchmark contact-valid subset
```

不能假装用了完整 130 条。

### 4. DeepPBS、RBE、EquiPNAS 的任务口径不同

| 方法 | 输入 | 输出 | 可比部分 |
|---|---|---|---|
| DeepPBS | protein-DNA complex | PWM | PWM |
| RBE | protein monomer + motif length | PWM + A map + site | PWM、site、A |
| EquiPNAS | protein structure | protein-DNA/RNA binding site | site |
| EquiPPIS | protein structure | protein-protein interface site | 不直接比较 |

因此：

```text
RBE vs DeepPBS: 只能主比 PWM
RBE vs EquiPNAS: 只能主比 protein binding site
RBE vs EquiPPIS: 不做主表数值比较
```

## 已执行流程与剩余计划

### Step 1. 准备 DeepPBS independent benchmark

目标：把 `id.txt` 处理成 RBE 可用的 contact-valid benchmark。

状态：已完成。当前本地结果是 112 条成功、18 条失败。

命令：

```bash
python scripts/prepare_deeppbs_curated.py \
  --fold-file id.txt \
  --out-root data/deeppbs_id_contactalign \
  --limit 0 \
  --device cuda
```

检查成功样本数：

```bash
wc -l data/deeppbs_id_contactalign/train_manifest.txt
```

检查失败原因：

```bash
column -t -s $'\t' data/deeppbs_id_contactalign/failed.tsv | head -n 50
cut -f2 data/deeppbs_id_contactalign/failed.tsv | sort | uniq -c | sort -nr
```

验收：

| 文件 | 要求 |
|---|---|
| `train_manifest.txt` | 至少几十个有效 benchmark samples |
| `failed.tsv` | 每个失败样本有明确原因 |
| `sample_table.tsv` | 每个成功样本有 `A_contact_pos/site_pos` |

### Step 2. 训练 5 个 fold 模型

目标：复刻 DeepPBS 5-fold ensemble 的评估设定。

状态：已完成。当前本地已有：

```text
runs/deeppbs_fold0_contactalign/best.pt
runs/deeppbs_fold1_contactalign/best.pt
runs/deeppbs_fold2_contactalign/best.pt
runs/deeppbs_fold3_contactalign/best.pt
runs/deeppbs_fold4_contactalign/best.pt
```

每个 fold 用对应 train 文件训练：

```bash
for i in 0 1 2 3 4; do
  python scripts/prepare_deeppbs_curated.py \
    --fold-file train${i}.txt \
    --out-root data/deeppbs_train${i}_contactalign \
    --limit 0 \
    --device cuda

  python -m rbe.train \
    --manifest data/deeppbs_train${i}_contactalign/train_manifest.txt \
    --config configs/dna_v1_contact.yaml \
    --out-dir runs/deeppbs_fold${i}_contactalign \
    --epochs 100 \
    --device cuda
done
```

验收：

| 检查项 | 目的 |
|---|---|
| 每个 fold 都有 `best.pt` | ensemble 有 5 个模型 |
| 每个 fold 的 train loss 正常下降 | 排除训练崩溃 |
| 每个 fold 样本数记录下来 | 说明 contact-valid filtering 后的数据规模 |

### Step 3. 在 DeepPBS benchmark 上做 5-fold ensemble

目标：每个 `id.txt` benchmark sample 用五个模型预测，然后平均。

状态：已完成。当前 `runs/deeppbs_id_ensemble/preds/` 下已有 112 个 `.pred.npz`。

命令：

```bash
python -m rbe.eval.predict_ensemble_manifest \
  --manifest data/deeppbs_id_contactalign/train_manifest.txt \
  --pred-dir runs/deeppbs_id_ensemble/preds \
  --checkpoints \
    runs/deeppbs_fold0_contactalign/best.pt \
    runs/deeppbs_fold1_contactalign/best.pt \
    runs/deeppbs_fold2_contactalign/best.pt \
    runs/deeppbs_fold3_contactalign/best.pt \
    runs/deeppbs_fold4_contactalign/best.pt \
  --device cuda
```

ensemble 定义：

```text
PWM = mean(PWM_0, PWM_1, PWM_2, PWM_3, PWM_4)
A = mean(A_0, A_1, A_2, A_3, A_4)
site_prob = mean(site_prob_0, site_prob_1, site_prob_2, site_prob_3, site_prob_4)
```

还要固定最终报告中使用的 checkpoint、训练 epoch、contact-valid 样本列表和失败样本列表。

### Step 4. 评估 RBE benchmark 结果

状态：已完成。结果在：

```text
runs/deeppbs_id_ensemble/preds/eval_summary.tsv
```

命令：

```bash
python -m rbe.eval.evaluate_manifest \
  --manifest data/deeppbs_id_contactalign/train_manifest.txt \
  --pred-dir runs/deeppbs_id_ensemble/preds \
  --device cuda
```

主要看：

| 指标 | 作用 |
|---|---|
| `pwm_mae` | 和 DeepPBS 主指标最接近 |
| `pwm_kl` | 看概率分布是否真的接近 |
| `pwm_ic_pcc` | 看高信息量 motif columns 是否对齐 |
| `A_contact_ap` | 看 residue-base contact map 是否可解释 |
| `site_ap` | 可和 EquiPNAS 方向比较 |

### Step 5. 形成 DeepPBS 对比表

状态：保守表已经能写；严格表还差 DeepPBS same-subset rerun。

#### 5.1 用本地 DeepPBS rerun 结果做同子集评估

当前可用的 DeepPBS rerun 目录：

```text
/home/dangqi/deeppbs_contact_aware_exp
```

这个目录下已有 `benchmark_id/baseline_filtered/npzs/*.npz_predict.npz` 和 `benchmark_id/contact_aware/npzs/*.npz_predict.npz`。这些文件里的 `P` 是 DeepPBS 对 DNA positions 的预测，不一定和 RBE 的 `pwm_target` 长度相同，所以不能直接逐列比较。

先把 DeepPBS DNA-position predictions 对齐到 RBE 的 motif slots：

```bash
python scripts/align_deeppbs_predictions_for_rbe_eval.py \
  --manifest data/deeppbs_id_contactalign/train_manifest.txt \
  --deeppbs-pred-dir /home/dangqi/deeppbs_contact_aware_exp/benchmark_id/baseline_filtered/npzs \
  --out-dir runs/deeppbs_same_subset/baseline_filtered

python scripts/align_deeppbs_predictions_for_rbe_eval.py \
  --manifest data/deeppbs_id_contactalign/train_manifest.txt \
  --deeppbs-pred-dir /home/dangqi/deeppbs_contact_aware_exp/benchmark_id/contact_aware/npzs \
  --out-dir runs/deeppbs_same_subset/contact_aware
```

输出：

| 文件 | 含义 |
|---|---|
| `preds_aligned/*.pred.npz` | 已裁剪/翻转到 RBE motif slots 的 DeepPBS PWM |
| `aligned_manifest.txt` | 能确认 alignment 一致的样本 |
| `alignment_modes.tsv` | 每个样本用 slot index 还是 sequence window 对齐 |
| `alignment_failures.tsv` | 不能确认 alignment 一致的样本 |

2026-06-02 当前结果：

| 目录 | aligned | failures | 解释 |
|---|---:|---:|---|
| `baseline_filtered` | 97 | 15 | 可做 common-alignable subset 比较 |
| `contact_aware` | 97 | 15 | 可作为额外变体，不要命名成 official DeepPBS |

这说明当前本地 DeepPBS rerun 还不是完整 112/112 strict comparison。15 条 failure 的主要原因是 DeepPBS preprocessing 里的 DNA sequence/PWM trimming 和 RBE contact-constrained alignment 不完全一致。要完成真正 112/112，需要让 DeepPBS rerun 使用和 RBE 完全一致的：

```text
contact-valid sample list
vendored PWM trimming
DNA slot/window
strand/orientation
```

在 97 条 common-alignable subset 上评估：

```bash
python -m rbe.eval.evaluate_manifest \
  --manifest runs/deeppbs_same_subset/baseline_filtered/aligned_manifest.txt \
  --pred-dir runs/deeppbs_same_subset/baseline_filtered/preds_aligned \
  --summary-tsv runs/deeppbs_same_subset/baseline_filtered/eval_summary.tsv \
  --per-sample-tsv runs/deeppbs_same_subset/baseline_filtered/eval_per_sample.tsv \
  --device cpu

python -m rbe.eval.evaluate_manifest \
  --manifest runs/deeppbs_same_subset/contact_aware/aligned_manifest.txt \
  --pred-dir runs/deeppbs_same_subset/contact_aware/preds_aligned \
  --summary-tsv runs/deeppbs_same_subset/contact_aware/eval_summary.tsv \
  --per-sample-tsv runs/deeppbs_same_subset/contact_aware/eval_per_sample.tsv \
  --device cpu

python -m rbe.eval.evaluate_manifest \
  --manifest runs/deeppbs_same_subset/baseline_filtered/aligned_manifest.txt \
  --pred-dir runs/deeppbs_id_ensemble/preds \
  --summary-tsv runs/deeppbs_same_subset/rbe_ensemble_97/eval_summary.tsv \
  --per-sample-tsv runs/deeppbs_same_subset/rbe_ensemble_97/eval_per_sample.tsv \
  --device cpu
```

当前 97 条 common-alignable subset 结果：

| 方法 | PWM MAE | PWM KL | IC-PCC | RC-aware KL | n |
|---|---:|---:|---:|---:|---:|
| RBE ensemble | 0.544151 | 0.464686 | 0.744590 | 0.391272 | 97 |
| DeepPBS `baseline_filtered` | 0.633143 | 0.537630 | 0.732972 | 0.529608 | 97 |
| DeepPBS `contact_aware` | 0.639378 | 0.512745 | 0.740452 | 0.500680 | 97 |

这张表可以写成“common-alignable subset diagnostic”，还不能替代 112 条 strict comparison。

先写两个层级的结论。

保守主表：

| 方法 | Benchmark | 输入 | PWM MAE |
|---|---|---|---:|
| DeepPBS official | official `id.txt`, full 130 | protein-DNA complex | median 0.4796 |
| DeepPBS reproduced | official `id.txt`, full 130 | protein-DNA complex | median 0.4985, mean 0.5412 |
| RBE ensemble | `id.txt` contact-valid 112 | protein monomer + motif length | mean 0.5801 |

更公平的补充表：

| 方法 | Benchmark subset | 输入 | MAE | KL | IC-PCC |
|---|---|---|---:|---:|---:|
| DeepPBS rerun | same contact-valid subset | protein-DNA complex | 待跑 |
| RBE ensemble | same contact-valid subset | protein monomer + motif length | 0.5801 | 0.5251 | 0.7132 |

第一张表能说明量级，但不是严格同条件比较。

第二张表才接近公平比较；这里的 `DeepPBS rerun` 是当前真正待办。

### Step 6. EquiPNAS 只做 site 对比

状态：后续可选。它不是当前 DeepPBS benchmark 卡点。

不要拿 EquiPNAS 比 PWM。

只比较：

```text
protein binding site prediction
```

报告时写：

| 方法 | 数据集 | 指标 | 说明 |
|---|---|---|---|
| EquiPNAS | Test_129 / Test_181 | PR-AUC | 文献结果 |
| RBE | DeepPBS id contact-valid subset | site AP | 初步结果 |

如果要严格比较，需要后续把 EquiPNAS 跑在同一批 benchmark protein 上。

## 当前不能说的话

不能写：

```text
RBE outperforms DeepPBS.
RBE outperforms EquiPNAS.
RBE is SOTA.
```

现在最多写：

```text
On the DeepPBS independent benchmark contact-valid subset, RBE shows promising PWM and protein-site signals.
A strict outperform claim requires rerunning DeepPBS on the same contact-valid subset with the same PWM metrics.
```

## 最短验收标准

| 阶段 | 通过标准 | 当前状态 |
|---|---|---|
| benchmark prepare | `id.txt` 成功样本数、失败原因可解释 | done |
| 5-fold training | 5 个 `best.pt` 都存在 | done |
| ensemble prediction | 每个 benchmark sample 有 `.pred.npz` | done |
| RBE benchmark eval | 输出 `eval_summary.tsv` | done |
| DeepPBS comparison | 至少有 official number vs RBE subset number | done，可写保守表 |
| 严格比较 | 同一 contact-valid subset 上 rerun DeepPBS | todo，真正剩余卡点 |
