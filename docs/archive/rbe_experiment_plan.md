# RBE 实验计划

## 核心原则

当前最重要的问题不是先追更高分，而是先把实验口径做干净。

| 原则 | 要求 |
|---|---|
| 样本独立性 | 不把同一个 complex 里的多条 chain 当成完全独立样本 |
| 输入口径清楚 | DeepPBS 是 complex-input，RBE 是 monomer-input，不能混成同任务 |
| 指标口径透明 | 同时报告 entry-level 和 group-level 指标 |
| 数据子集透明 | 每个 benchmark 都要写清原始条目数、成功数、失败数、去重后数量 |
| 先验证链路 | 单样本、少量样本、全量实验逐级推进 |

## 分组定义

| 名称 | 定义 | 用途 |
|---|---|---|
| entry-level | `PDB + chain + PWM` | 对齐 DeepPBS 原始 fold 文件 |
| PDB+PWM group | `PDB + PWM` | 解决同一 complex 同一 target 多 chain 重复计权 |
| PDB group | `PDB` | 最严格结构级去重 |
| PWM group | `PWM id` | 检查某些 motif 是否被过度代表 |

例子：

```text
4m9v_C_ZFP57_MOUSE.H11MO.0.B.npz
4m9v_F_ZFP57_MOUSE.H11MO.0.B.npz
```

entry-level 是两条，但 PDB+PWM group 只是一组：

```text
4m9v + ZFP57_MOUSE.H11MO.0.B
```

## P0. 数据审计实验

### E00. DeepPBS fold 重复计权审计

目标：量化 DeepPBS fold 中 same-PDB / same-PDB+PWM 多 chain 重复现象。

| 检查项 | 输出 |
|---|---|
| 每个 fold 的 entry 数 | `n_entry` |
| unique PDB 数 | `n_pdb` |
| unique PDB+PWM 数 | `n_pdb_pwm` |
| same PDB 多 entry 数 | `multi_pdb_groups` |
| same PDB+PWM 多 entry 数 | `multi_pdb_pwm_groups` |
| 重复组明细 | `duplicate_groups.tsv` |

重点文件：

```text
resources/deeppbs_curated/folds/train0.txt
resources/deeppbs_curated/folds/valid0.txt
resources/deeppbs_curated/folds/id.txt
```

验收标准：

```text
明确证明 DeepPBS 是 chain-level entry 口径，不是 independent complex 口径。
```

### E01. Train/valid/id group 泄漏审计

目标：确认同一个 PDB 或 PDB+PWM group 是否跨 train/valid/id 出现。

| 检查 | 解释 |
|---|---|
| train vs valid 是否共享 PDB | 检查结构级泄漏 |
| train vs valid 是否共享 PDB+PWM | 检查同 target 结构泄漏 |
| train/valid vs id 是否共享 PDB | 检查 independent benchmark 是否真独立 |
| train/valid vs id 是否共享 PWM | 检查 motif target overlap |

注意：这个实验和 E00 不同。E00 是重复计权，E01 是泄漏。

### E02. DeepPBS id benchmark 去重覆盖率

目标：统计 `id.txt` 在不同口径下的 benchmark 大小。

| 口径 | 输出 |
|---|---|
| raw entry-level | 原始 130 条 |
| PDB+PWM 去重 | 去重后数量 |
| PDB 去重 | 最严格数量 |
| contact-valid subset | RBE 可处理数量 |
| masked/partial-valid subset | 后续支持 partial mask 后的可处理数量 |

验收标准：

```text
后续所有 benchmark 表格必须写明使用哪个口径。
```

## P1. 公平 benchmark 实验

### E10. RBE raw entry-level benchmark

目标：复现和 DeepPBS 原始 `id.txt` 最接近的 entry-level 评估。

| 项 | 设置 |
|---|---|
| train | 5 fold RBE 模型 |
| test | `id.txt` contact-valid entries |
| prediction | 5-fold ensemble |
| metric | PWM MAE/KL/IC-PCC、A map、site |

用途：

```text
只作为 DeepPBS raw 口径参考，不作为最终最干净主结论。
```

### E11. RBE PDB+PWM group-level benchmark

目标：解决同一个 complex + same PWM 多 chain 重复测试的问题。

做法：

```text
id.txt raw entries
  ↓
按 PDB+PWM 分组
  ↓
每组只贡献一个 metric
```

可选聚合方式：

| 聚合方式 | 含义 |
|---|---|
| best-contact chain | 每组保留 `A_contact_pos` 最多的 chain |
| mean-over-chains | 每组内先平均 metric，再对 group 平均 |

主报告建议：

```text
PDB+PWM group-level mean-over-chains
```

理由：不丢 entry，但每个 group 权重相同。

### E12. RBE PDB-level strict benchmark

目标：最严格地避免同一个 PDB complex 被重复计分。

做法：

```text
每个 PDB 只贡献一个最终 metric
```

用途：

```text
作为保守敏感性分析，不一定作为主指标。
```

### E13. DeepPBS same-subset rerun

目标：在 RBE 可处理的同一批样本上 rerun DeepPBS，避免 full set vs subset 不公平。

| 对比项 | 要求 |
|---|---|
| same entry subset | DeepPBS 和 RBE 用同一批 entry |
| same PDB+PWM group subset | DeepPBS 和 RBE 用同一批 group |
| same metric script | 用同一套 PWM metric |

结论口径：

```text
DeepPBS = complex-input reference
RBE = monomer-input model
```

不能写成直接同输入击败。

### E14. DeepPBS raw vs dedup 敏感性实验

目标：专门验证 DeepPBS 的 chain-level 重复计权对指标影响多大。

| DeepPBS 口径 | 目的 |
|---|---|
| raw entry-level | 复现官方口径 |
| PDB+PWM group-level | 去掉 same-complex same-target 重复计权 |
| PDB-level | 最严格结构去重 |

如果 raw 和 dedup 差异大，说明 benchmark 受重复计权影响明显。

## P2. 训练数据去重实验

### E20. RBE raw train folds

目标：作为当前 baseline。

```text
train fold 使用 DeepPBS 原始 chain-level entries
```

问题：

```text
同一 PDB+PWM 多 chain 会在 loss 中重复计权。
```

### E21. RBE PDB+PWM dedup train folds

目标：训练阶段也不让同一 complex+PWM 重复贡献 loss。

做法：

```text
每个 train fold
  ↓
按 PDB+PWM 分组
  ↓
每组只保留一条
```

保留规则：

| 规则 | 说明 |
|---|---|
| max `A_contact_pos` | 主方案，保留 contact 最充分的 chain |
| max `A_base_pos` | 辅助分析，偏 PWM-specific contact |
| first entry | 只作对照，不建议主用 |

主报告：

```text
PDB+PWM dedup train + PDB+PWM group-level test
```

### E22. RBE group-weighted train folds

目标：不丢 chain，但每个 PDB+PWM group 的总 loss 权重相同。

做法：

```text
同一 group 内每条 entry 权重 = 1 / group_size
```

用途：

```text
和 E21 比较：去重丢信息 vs group weighting 保留信息。
```

### E23. RBE PDB-level dedup train folds

目标：最严格训练去重。

做法：

```text
每个 PDB 只保留一个 entry
```

用途：

```text
判断当前模型性能是否依赖 same-PDB 多 chain 重复增强。
```

## P3. 数据处理与标签实验

### E30. Alignment policy 对照

目标：比较 PWM-DNA 对齐策略对训练标签的影响。

| 策略 | 含义 |
|---|---|
| `sequence_only` | 只按 PWM-DNA sequence score 对齐 |
| `require_contact` | 当前 contact-constrained 对齐 |
| manual known mapping | 有可靠 mapping 时人工指定 |

指标：

| 指标 | 用途 |
|---|---|
| `A_base_pos` | base contact 标签是否足够 |
| `A_contact_pos` | site 标签是否足够 |
| PWM metric | 对最终 PWM 的影响 |
| site metric | 对 binding site 的影响 |

### E31. Contact cutoff 消融

目标：确认接触阈值是否过松或过紧。

| 参数 | 候选 |
|---|---|
| base cutoff | `4.0, 4.5, 5.0` |
| backbone cutoff | `4.5, 5.0, 5.5` |

观察：

```text
接触标签数量、A map AP、site AP、PWM 质量。
```

### E32. Partial PWM / masked supervision

目标：处理 DNA 片段短于 PWM 或只覆盖部分 motif 的样本。

原因：

```text
当前完整 slot_to_dna_index 要求会丢掉一批样本。
DeepPBS 支持 partial overlap + mask。
```

输出：

| 字段 | 含义 |
|---|---|
| `pwm_mask` | 哪些 PWM slot 有监督 |
| `A_mask` | 哪些 residue-slot contact 可监督 |
| masked PWM loss | 只在有效 slot 上算 |

用途：

```text
提高 id benchmark 覆盖率，减少 contact-valid subset 偏差。
```

### E33. Data prepare 加速实验

目标：把数据处理速度从瓶颈变成可接受。

| 实验 | 目的 |
|---|---|
| ESM2 cache | 避免同一 PDB chain 重复跑 ESM2 |
| existing npz skip | 避免重复 prepare 已成功样本 |
| direct function call | 避免每个样本 subprocess 重新加载 |
| distance precompute | 加速 contact-constrained alignment |

指标：

```text
每 100 个样本 prepare 用时、GPU 显存、失败率。
```

## P4. 模型消融实验

### E40. 单样本 overfit

目标：确认模型和 loss 没硬伤。

| 样本 | 目标 |
|---|---|
| SMAD3 1OZJ | 真实小样本 |
| 随机 3 个 contact-rich 样本 | 排除单例偶然 |

验收：

```text
训练集 PWM loss、A loss、site loss 都能明显下降。
```

### E41. Encoder 消融

目标：确认 ESM2 和 EGNN 分别贡献什么。

| 模型 | 目的 |
|---|---|
| full ESM2 + EGNN | baseline |
| no EGNN | 只看 ESM2 + 坐标半径 |
| no ESM2 | 只看 AA one-hot + 结构 |
| no radius | 检查 radius 是否有用 |
| fewer EGNN layers | 检查 2/4/6 层差异 |

主观察：

```text
site_ap、A_contact_ap、pwm_ic_pcc。
```

### E42. Slot 表达消融

目标：验证当前最大风险点：motif slot 是否表达不足。

| 模型 | 目的 |
|---|---|
| slot embedding only | 去掉 `slot_pos` |
| slot embedding + pos | 当前 baseline |
| sinusoidal pos | 更强位置编码 |
| reverse-complement augmentation | 处理 motif 方向不确定 |

观察：

```text
A_base_ap 和 pwm_ic_pcc 是否提升。
```

### E43. Pair 表达消融

目标：验证 `z(i,j)` 是否足够强。

| 版本 | 改动 |
|---|---|
| baseline | `[h_i, s_j, h_i*s_j, pos_j]` |
| no product | 去掉 `h_i*s_j` |
| add abs diff | 加 `|h_i - s_j|` |
| deeper pair_mlp | 增加 pair MLP 深度 |

判断：

```text
如果 no product 明显变差，说明交互项有用。
如果 deeper pair_mlp 提升小，说明瓶颈不在 head 复杂度。
```

### E44. Head 消融

目标：验证 linear head 是否足够。

| 版本 | 改动 |
|---|---|
| linear heads | 当前 baseline |
| 2-layer heads | A/E head 前加 MLP |
| shared vs separate pair_mlp | A 和 E 是否共用 z |

预期：

```text
优先验证，不预设 linear head 有问题。
```

## P5. Loss 与训练策略实验

### E50. Loss 权重消融

目标：找出 PWM、A、site 之间的权衡。

| 配置 | 重点 |
|---|---|
| `dna_v1.yaml` | PWM 更均衡 |
| `dna_v1_contact.yaml` | contact/site 更强 |
| no `loss_pwm_teacher` | 看 teacher gate 是否必要 |
| no `loss_noncontact` | 看非接触惩罚是否必要 |
| no `loss_sparse` | 看 A 是否发散 |

观察：

```text
PWM 提升是否以 A/site 下降为代价。
```

### E51. Epoch 和 early stopping 实验

目标：确认 50/100/200 epoch 的收益。

| epoch | 用途 |
|---|---|
| 20 | 快速趋势 |
| 50 | 当前默认 |
| 100 | 中等训练 |
| 200 | 长训练 |

输出：

```text
train loss 曲线、valid metric 曲线、best epoch。
```

### E52. 5-fold ensemble 实验

目标：复刻 DeepPBS ensemble 口径，同时检查 RBE 稳定性。

| 版本 | 说明 |
|---|---|
| single fold | 单模型 |
| 5-fold mean ensemble | 预测平均 |
| 5-fold metric variance | 检查不同 fold 稳定性 |

## P6. 输出解释性实验

### E60. A map 可视化检查

目标：确认 `A_base/A_backbone` 不是只在数值上好看。

检查：

| 内容 | 目的 |
|---|---|
| top A_base residue-slot pair | 是否真的靠近 DNA base |
| top A_backbone pair | 是否真的靠近 backbone |
| site_prob top residues | 是否在 DNA-binding interface |

样本：

```text
SMAD3 1OZJ
id benchmark 中若干高分/低分样本
```

### E61. E map 解释性检查

目标：看 `E(i,j,b)` 是否学到合理 base preference。

检查：

| 内容 | 目的 |
|---|---|
| 高 A_base pair 的 E 分布 | 是否有明确碱基偏好 |
| 低 A_base pair 的 E 贡献 | 是否被 noncontact loss 压住 |
| PWM column contribution | 哪些 residue 主导某个 PWM column |

## P7. 最终报告实验表

最终报告至少应该有这些表。

### 表 1. 数据口径表

| 数据集 | raw entries | unique PDB | unique PDB+PWM | contact-valid entries | contact-valid groups |
|---|---:|---:|---:|---:|---:|
| train0 | TBD | TBD | TBD | TBD | TBD |
| valid0 | TBD | TBD | TBD | TBD | TBD |
| id | TBD | TBD | TBD | TBD | TBD |

### 表 2. DeepPBS raw vs dedup

| 方法 | 输入 | 评估口径 | n | PWM MAE | PWM IC-PCC |
|---|---|---|---:|---:|---:|
| DeepPBS | complex | raw entry | TBD | TBD | TBD |
| DeepPBS | complex | PDB+PWM group | TBD | TBD | TBD |
| DeepPBS | complex | PDB group | TBD | TBD | TBD |

### 表 3. RBE raw train vs dedup train

| 训练口径 | 测试口径 | n | PWM MAE | A_contact AP | site AP |
|---|---|---:|---:|---:|---:|
| raw entry train | raw entry test | TBD | TBD | TBD | TBD |
| raw entry train | PDB+PWM group test | TBD | TBD | TBD | TBD |
| PDB+PWM dedup train | PDB+PWM group test | TBD | TBD | TBD | TBD |
| group-weighted train | PDB+PWM group test | TBD | TBD | TBD | TBD |

### 表 4. 模型消融

| 模型 | PWM MAE | PWM IC-PCC | A_base AP | A_contact AP | site AP |
|---|---:|---:|---:|---:|---:|
| full | TBD | TBD | TBD | TBD | TBD |
| no EGNN | TBD | TBD | TBD | TBD | TBD |
| no ESM2 | TBD | TBD | TBD | TBD | TBD |
| no product pair | TBD | TBD | TBD | TBD | TBD |
| deeper pair head | TBD | TBD | TBD | TBD | TBD |

## 优先级

| 优先级 | 实验 | 原因 |
|---|---|---|
| P0 | E00, E01, E02 | 先把 DeepPBS 数据口径讲清楚 |
| P1 | E10, E11, E14 | 证明 raw entry 和 group-level 差异 |
| P1 | E20, E21 | 训练阶段解决重复计权 |
| P2 | E13 | 和 DeepPBS 做 same-subset reference |
| P2 | E40, E50, E52 | 确认模型训练可靠 |
| P3 | E41-E44 | 做模型结构消融 |
| P3 | E30-E33 | 改善数据质量和速度 |

## 当前主线

最短主线如下：

```text
1. 审计 DeepPBS fold 重复计权
2. 生成 PDB+PWM group-level benchmark
3. raw RBE 先做 group-level 重新评估
4. 训练 PDB+PWM dedup RBE
5. 比较 raw train vs dedup train
6. rerun DeepPBS same-subset raw/group-level
7. 再做模型和 loss 消融
```

如果这条主线跑完，RBE 的实验口径会比 DeepPBS 原始 chain-level 口径更干净。
