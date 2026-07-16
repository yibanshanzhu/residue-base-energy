# Method

RBE 的核心问题是：

```text
给定 protein monomer，预测它偏好的 DNA motif/PWM，并解释哪些 protein residues 读取哪些 motif slots。
```

## Core Variables

| 符号 | shape | 含义 |
|---|---:|---|
| `A_base(i,j)` | `N,M` | residue `i` 是否读取 motif slot `j` 的 base |
| `A_backbone(i,j)` | `N,M` | residue `i` 是否接触 slot `j` 的 sugar/phosphate backbone |
| `A_contact(i,j)` | `N,M` | `max(A_base, A_backbone)` |
| `E(i,j,b)` | `N,M,4` | residue `i` 对 slot `j` 上 base `b` 的偏好 |
| `PWM(j,b)` | `M,4` | motif slot `j` 的 A/C/G/T 分布 |

模型聚合方式：

```text
PWM[j,b] = softmax_b Σ_i A_base(i,j) * E(i,j,b)
```

## Train vs Inference

| 阶段 | 输入 | 输出 |
|---|---|---|
| 训练 | protein-DNA complex structure + motif PWM | contact labels、site label、PWM target |
| 推理 | protein monomer structure + motif length `M` | PWM、protein site、residue-slot contacts |

## Data Definition

样本源头统一由 source manifest 定义：

```text
PDB/mmCIF structure + motif database PWM + chain selection + split
```

具体 schema 见 [`../metadata/README.md`](../metadata/README.md)。

motif target 必须来自公共 motif database 的未裁剪完整 PWM。DeepPBS vendored PWM 经过端点 trimming，只用于追溯旧资源，不进入当前 source manifest。

## Structural Visibility

完整 PWM 的长度是 `M`，但 PDB 结构不一定包含所有对应 DNA bases：

| 值 | 含义 |
|---|---|
| `slot_to_dna_index[j] >= 0` | motif column `j` 在结构中找到对应 DNA residue |
| `slot_to_dna_index[j] == -1` | motif column `j` 没有结构对应物 |
| `pwm_mask[j] == 1` | 该 column 可生成结构 contact label |
| `pwm_mask[j] == 0` | 该 column 不参与 contact/map 监督与评估 |

`pwm_mask` 描述结构可见性，不是 PWM target 是否有效。PWM target、PWM loss 和 PWM metrics 均使用完整的 `M x 4` 矩阵；它只限定 contact/map 监督与评估。

## Canonical Orientation

protein-only 输入无法确定 motif 的任意链方向。因此对每个 PWM 及其 reverse complement 展平后作字典序比较，固定选择较大者作为 canonical orientation。若方向发生翻转，`slot_to_dna_index`、`pwm_mask`、所有 `N x M` contact labels 和 `E` 的 slot/base 轴同步变换。

## PWM Evaluation

对每个 sample，当前 MAE 定义为：

```text
MAE_sample = mean_j sum_b |PWM_target[j,b] - PWM_pred[j,b]|
```

数据集结果是 `mean_sample(MAE_sample)`。每个 sample 权重相同，不会把不同长度 PWM 的 columns 跨样本汇集。KL 和 IC-PCC 同样在 canonical 完整 PWM 上计算；不使用 `min(direct, RC)` oracle。

## Boundary

| 项 | 当前边界 |
|---|---|
| 与 DeepPBS | DeepPBS 是 complex-input reference；RBE 推理时不使用 DNA 坐标 |
| 可比主指标 | PWM metrics |
| site 指标 | 可作为附加结果，不和 DeepPBS 直接等价 |
| partial structure | 用 `pwm_mask` 和 `slot_to_dna_index=-1` 表达不可见 columns |
| PWM 范围 | 所有 PWM 指标评估 canonical 完整 PWM |
