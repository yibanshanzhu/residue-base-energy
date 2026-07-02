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

## Boundary

| 项 | 当前边界 |
|---|---|
| 与 DeepPBS | DeepPBS 是 complex-input reference；RBE 推理时不使用 DNA 坐标 |
| 可比主指标 | PWM metrics |
| site 指标 | 可作为附加结果，不和 DeepPBS 直接等价 |
| partial PWM | 已用 `pwm_mask` 和 `slot_to_dna_index=-1` 表达不可见 columns |
