# 顶会级 RBE Idea

## 一句话

把 RBE 从一个 PWM prediction 工程模型，升级成一个更一般的计算问题：

```text
只给 protein monomer structure，
模型要推断缺失的 DNA binding partner，
并同时预测 PWM、binding site 和 residue-base 解释图。
```

英文版：

```text
Monomer-only protein-DNA specificity prediction as latent interaction-partner inference.
```

## 现在版本为什么还不够顶会

当前 RBE V1 是：

```text
protein monomer structure + ESM2
  ↓
EGNN
  ↓
abstract motif slot embedding
  ↓
A_base / A_backbone / E
  ↓
PWM + site
```

它有价值，但顶会风险是：

| 问题 | reviewer 可能怎么说 |
|---|---|
| slot 只是 embedding | motif position 没有明确几何含义 |
| 模型像任务拼接 | ESM2 + EGNN + MLP heads，方法新意不够强 |
| 主要是应用结果 | 更像 bioinformatics application，不像一般 ML contribution |
| DeepPBS 对比口径复杂 | DeepPBS 输入 complex，RBE 输入 monomer，不能直接说击败 |
| benchmark 原始口径不干净 | DeepPBS fold 是 chain-level，有 same-complex 重复计权 |

所以不能只讲：

```text
我们预测 PWM 和 binding site。
```

要讲：

```text
我们从 monomer structure 中推断缺失的 interaction partner。
```

## 核心问题重定义

DeepPBS 类方法的问题定义是：

```text
输入：protein-DNA complex
输出：PWM
```

也就是测试时已经给了 DNA 坐标：

```text
DNA 在哪里
motif slot 对应哪个 DNA base
protein 哪一面接触 DNA
```

这些信息在输入里已经部分存在。

RBE 要解决的问题应该定义成：

```text
输入：protein monomer structure + motif length M
输出：
  1. latent DNA binding geometry
  2. residue-slot-base map
  3. PWM
  4. protein binding site
```

也就是：

```text
训练时可以看 complex
测试时只能看 monomer
```

这是一个更一般的问题：

```text
learning with privileged complex supervision
missing modality inference
latent structured prediction
geometric reasoning under missing interaction partner
```

## 核心假设

蛋白单体结构里包含部分 DNA 结合信息：

| 信息 | 来源 |
|---|---|
| 可能的 binding surface | residue geometry + electrostatics/protein language model |
| 哪些 residue 参与 DNA contact | surface pattern + ESM2 |
| motif slot 顺序 | DNA-binding domain geometry |
| base specificity | residue identity + local structural context |

所以模型可以学习：

```text
从蛋白表面几何和 residue 语义中，反推出 DNA partner 的 latent pose 和 sequence preference。
```

## 方法升级方向

### V1 当前做法

当前 slot 是抽象编号：

```text
slot_j = learned embedding(j) + scalar position(j)
```

问题：

```text
slot_j 没有 DNA 几何含义。
模型必须自己在黑箱 embedding 里学 motif 方向、间距、strand、RC 对称。
```

### 顶会版做法：latent DNA partner

把 motif slot 从抽象 embedding 升级成 latent DNA base-pair frame：

```text
protein monomer
  ↓
predict latent DNA axis / binding frame
  ↓
construct M latent base-pair frames
  ↓
residue interacts with latent slot geometry
  ↓
A_base / A_backbone / E
  ↓
PWM + site
```

每个 motif slot 不再只是 `slot_j`，而是：

```text
slot_j = {
  3D center,
  local frame,
  helix position,
  strand/orientation representation
}
```

也就是：

```text
latent DNA geometry replaces pure slot embedding.
```

## 推荐模型结构

### 输入

```text
protein monomer PDB or AF2 structure
motif length M
```

特征：

| 特征 | 说明 |
|---|---|
| residue coordinates | Cα or backbone atoms |
| residue graph | spatial neighbors |
| ESM2 representation | residue semantic/context |
| AA identity | residue type |
| optional surface/electrostatics | 后续可加，不作为第一版必需 |

### Protein encoder

```text
ESM2 + EGNN / SE(3)-equivariant encoder
```

输出：

```text
h_i = residue i 的结构上下文表示
```

### Latent DNA geometry module

目标：从蛋白表面预测 DNA partner 的 latent 几何。

可能实现：

| 方案 | 说明 |
|---|---|
| anchor residue attention | 从蛋白表面选出 DNA-binding anchor |
| latent helix axis prediction | 预测 DNA helix 轴方向和中心 |
| M slot frames | 沿 helix 生成 M 个 base-pair frame |
| orientation head | 预测 motif 正向/反向或用 RC-invariant loss |

输出：

```text
g_j = latent motif slot j 的几何表示
```

### Residue-slot interaction module

当前是：

```text
pair_input = [h_i, s_j, h_i * s_j, pos_j]
```

升级后：

```text
pair_input = [
  h_i,
  g_j,
  residue-to-slot distance,
  residue-to-slot direction,
  relative frame features,
  helix position,
  h_i * slot_feature_j
]
```

输出：

```text
z(i,j)
```

然后：

```text
A_base(i,j)
A_backbone(i,j)
E(i,j,b)
```

### Readout

保持 RBE 的核心解释公式：

```text
PWM[j,b] = softmax_b Σ_i A_base(i,j) * E(i,j,b)
```

site：

```text
A_contact(i,j) = max(A_base(i,j), A_backbone(i,j))
site_prob(i) = max_j A_contact(i,j)
```

这个公式应该保留，因为它是 RBE 的解释性核心。

## 训练监督

训练时输入 complex，可以获得 privileged supervision。

| 监督 | 来源 | 用途 |
|---|---|---|
| true DNA coordinates | complex PDB | 监督 latent DNA pose/frame |
| A_base_label | residue-DNA base distance | 监督 residue-slot base contact |
| A_backbone_label | residue-DNA backbone distance | 监督 residue-slot backbone contact |
| PWM target | curated PWM | 监督 specificity |
| site_label | contact labels | 监督 binding site |

训练目标：

```text
L =
  L_pwm
  + L_A_base
  + L_A_backbone
  + L_site
  + L_latent_DNA_pose
  + L_sparse
  + L_noncontact
  + optional RC-invariant loss
```

关键点：

```text
训练时可以用 DNA 坐标监督；
推理时不给 DNA 坐标。
```

这就是：

```text
privileged information during training, missing modality during inference.
```

## 为什么这比当前版本强

| 维度 | 当前 RBE | 顶会版 RBE |
|---|---|---|
| slot 表示 | abstract embedding | latent DNA frame |
| 几何归纳偏置 | 弱 | 强 |
| 可解释性 | A/E heatmap | A/E + predicted DNA pose |
| ML 问题 | task-specific prediction | missing partner inference |
| 对 DeepPBS 差异 | 输入少 | 输入少 + 显式推断缺失 partner |
| reviewer 感知 | 应用模型 | 新问题 + 新结构化方法 |

## 必须做的关键实验

### E1. Monomer-only vs complex-input reference

| 方法 | 输入 | 输出 |
|---|---|---|
| DeepPBS | protein-DNA complex | PWM |
| RBE-current | protein monomer + M | PWM/site/A/E |
| RBE-latentDNA | protein monomer + M | latent DNA + PWM/site/A/E |

目标：

```text
证明在不给 DNA 坐标的情况下，RBE-latentDNA 接近 complex-input reference。
```

### E2. Latent DNA geometry accuracy

训练时有 true DNA，可以评估预测的 latent DNA pose。

指标：

| 指标 | 说明 |
|---|---|
| slot center RMSD | predicted slot centers vs true DNA base centers |
| helix axis angle error | predicted DNA axis 是否对 |
| contact recovery | predicted slot 是否落在真实 contact 区域 |
| RC-aware alignment | 允许反向互补对称 |

这个实验是计算 reviewer 会喜欢的：

```text
模型不只是输出 PWM，还学到了缺失 partner 的几何。
```

### E3. Slot representation ablation

| 模型 | 目的 |
|---|---|
| abstract slot embedding | 当前 baseline |
| learned latent slots without geometry loss | 看几何监督是否必要 |
| latent DNA frame + geometry loss | 主模型 |
| true DNA slot oracle | 上限 |

这个实验能证明：

```text
latent DNA geometry 不是装饰，而是关键贡献。
```

### E4. Group-level benchmark

DeepPBS 原始 fold 是 chain-level entry：

```text
PDB + chain + PWM
```

RBE 应该报告：

| 口径 | 目的 |
|---|---|
| raw entry-level | 对齐 DeepPBS 原始结果 |
| PDB+PWM group-level | 去掉 same-complex same-target 重复计权 |
| PDB-level | 最严格结构去重 |

重点：

```text
每个 complex-target group 只贡献一次 metric。
```

这能打计算 reviewer：

```text
benchmark hygiene 更严格。
```

### E5. Leave-family / leave-PWM split

目标：证明模型不是记住 TF family 或 PWM。

| split | 说明 |
|---|---|
| leave PDB | 不见过同结构 |
| leave PDB+PWM | 不见过同 complex-target |
| leave TF family | 不见过同家族 |
| leave PWM id | 不见过同 PWM target |

顶会更看重泛化，不只看随机 split。

### E6. AF2/ESMFold monomer inference

目标：证明实际使用时不需要 co-crystal complex。

输入：

```text
AlphaFold-predicted monomer
```

输出：

```text
PWM + predicted binding site + latent DNA pose
```

对生物 reviewer 很重要：

```text
很多 TF 没有 protein-DNA co-crystal structure。
```

### E7. Mutation / residue masking

目标：证明模型机制解释可信。

做法：

| 实验 | 预期 |
|---|---|
| mask top A_base residue | 对应 PWM column 改变 |
| mask low A_base residue | PWM 基本不变 |
| mutate known DNA-reading residue | motif preference 改变 |
| compare known recognition code | Arg/Guanine 等经典规律 |

这个实验打生物 reviewer：

```text
模型解释能对应已知 recognition mechanism。
```

### E8. A/E map case studies

选择若干经典 TF：

| 样本 | 展示 |
|---|---|
| SMAD3 | residue-slot map |
| MEF2 / ZFP57 / p53 | known DNA-reading residues |
| high-performance sample | 模型成功案例 |
| failure sample | 失败原因分析 |

展示内容：

```text
protein structure 上标 top site residues
A_base heatmap
E contribution heatmap
predicted latent DNA pose
target vs predicted PWM
```

## 最重要的对照表

### 表 1. 输入条件

| 方法 | 输入 DNA 坐标 | 输入 protein | 输出 PWM | 输出 site | 输出 explanation | 输出 latent DNA |
|---|---|---|---|---|---|---|
| DeepPBS | yes | complex protein | yes | no/implicit | limited | no |
| RBE-current | no | monomer | yes | yes | A/E | no |
| RBE-latentDNA | no | monomer | yes | yes | A/E | yes |
| true-DNA oracle | yes | complex protein | yes | yes | A/E | true geometry |

### 表 2. Benchmark hygiene

| 口径 | 样本单位 | 是否重复计权 same complex | 用途 |
|---|---|---|---|
| entry-level | PDB+chain+PWM | 可能 | 对齐 DeepPBS raw |
| PDB+PWM group | PDB+PWM | 否，同 target 去重 | 主报告 |
| PDB-level | PDB | 否，最严格 | 敏感性分析 |

### 表 3. 方法消融

| 模型 | PWM | site | A map | latent DNA pose |
|---|---|---|---|---|
| current slot embedding | TBD | TBD | TBD | none |
| latent slots no geometry loss | TBD | TBD | TBD | weak |
| latent DNA frame | TBD | TBD | TBD | yes |
| true DNA oracle | upper bound | upper bound | upper bound | true |

## 论文标题方向

候选标题：

```text
Monomer-only prediction of protein-DNA binding specificity via latent partner inference
```

```text
Inferring latent DNA interaction geometry from protein monomers for binding specificity prediction
```

```text
Learning residue-base recognition from privileged protein-DNA complexes for monomer-only specificity prediction
```

## Abstract 逻辑

文章摘要应该这样讲：

```text
Protein-DNA binding specificity is usually predicted from either sequence motifs
or protein-DNA complex structures. However, complex structures are unavailable
for most transcription factors, while monomer structures are increasingly
available through structure prediction.

We formulate monomer-only specificity prediction as a latent partner inference
problem: during training, protein-DNA complexes provide privileged supervision
for DNA geometry and residue-base contacts; during inference, only the protein
monomer and motif length are provided.

Our model infers latent DNA slot frames, residue-slot contact maps, and
residue-base energy contributions, from which PWM and binding-site predictions
are composed.

We evaluate under group-level complex-target benchmarks to avoid chain-level
duplicate weighting, and show that monomer-only RBE approaches complex-input
references while providing interpretable residue-base mechanisms.
```

## 最终卖点

| 卖点 | 打谁 |
|---|---|
| monomer-only，不需要 DNA complex | 生物 reviewer |
| latent partner inference | 计算 reviewer |
| privileged complex supervision | 计算 reviewer |
| group-level benchmark hygiene | 计算 reviewer |
| A/E residue-base explanation | 生物 reviewer |
| AF2 monomer 可用 | 生物 reviewer |

## 最短实现路线

不要一口吃成完整 DNA diffusion model。先按下面路线推进：

```text
Step 1. 做 group-level benchmark 和数据审计
Step 2. 固定当前 RBE-current baseline
Step 3. 加 latent DNA slot centers，不先预测完整原子
Step 4. 用 true DNA slot center 做 geometry supervision
Step 5. 加 residue-to-slot distance/direction features
Step 6. 比较 slot embedding vs latent DNA geometry
Step 7. 做 AF2 monomer 和解释性 case study
```

第一版 latent DNA 不需要生成真实双螺旋全原子，只需要：

```text
M 个 latent slot centers + direction/frame
```

这样实现成本可控，但论文问题定义已经明显升级。

## 具体怎么做

核心原则：

```text
第一版只预测 latent DNA slot geometry，
不预测完整 DNA 双螺旋全原子结构。
```

也就是说，我们不做：

```text
protein -> full DNA atom coordinates
```

而是做：

```text
protein -> M 个 motif slot 的 3D center / axis / frame
```

这些 latent geometry 只服务于：

```text
让 residue i 和 motif slot j 的关系有几何依据。
```

### Phase 0. 先把真实 DNA slot label 存进数据

目标：训练时从 complex PDB 里拿到每个 motif slot 的真实几何标签。

当前 `.npz` 已有：

```text
slot_to_dna_index
A_base_label
A_backbone_label
A_contact_label
```

需要新增：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `slot_center_xyz` | `M,3` | 每个 motif slot 对应 DNA residue 的中心 |
| `slot_base_center_xyz` | `M,3` | DNA base heavy atoms 的中心 |
| `slot_backbone_center_xyz` | `M,3` | sugar/phosphate heavy atoms 的中心 |
| `slot_mask` | `M` | 这个 slot 是否有真实 DNA 几何监督 |
| `dna_axis_hint` | `3` 或 `M,3` | DNA 局部方向，可选 |

第一版最简单：

```text
slot_center_xyz = base heavy atom centroid
```

也可以同时存：

```text
base centroid
backbone centroid
```

这样后面能分别监督：

```text
A_base 对应 base geometry
A_backbone 对应 backbone geometry
```

验收标准：

```text
每个成功样本都能在 npz 里读到 M 个 slot center。
```

### Phase 1. 做 true-DNA oracle

目标：先不让模型预测 latent DNA，直接把真实 slot geometry 喂给 pair module。

这是开卷参考：

```text
protein residue h_i
true DNA slot center c_j
  ↓
residue-slot geometric features
  ↓
A/E/PWM/site
```

pair 输入从当前：

```text
[h_i, s_j, h_i * s_j, pos_j]
```

升级成 oracle 版本：

```text
[
  h_i,
  s_j,
  h_i * s_j,
  pos_j,
  distance(residue_i, true_slot_j),
  direction(residue_i -> true_slot_j)
]
```

形状示意：

| 特征 | 形状 |
|---|---:|
| residue embedding `h_i` | `N,M,H` |
| slot embedding `s_j` | `N,M,H` |
| distance | `N,M,1` |
| direction | `N,M,3` |
| pair input | `N,M,3H+1+4` |

为什么先做 oracle：

| 结果 | 解释 |
|---|---|
| oracle 明显好 | 说明几何 slot 有价值，下一步学 latent geometry |
| oracle 也不好 | 说明问题在 A/E readout、loss 或标签，不该急着做 latent |

验收标准：

```text
true-DNA oracle 的 A_contact_AP / PWM 指标应该高于 current slot embedding baseline。
```

### Phase 2. 预测 latent slot centers

目标：推理时不给 DNA，模型自己预测：

```text
pred_slot_center_xyz: M,3
```

最短实现：

```text
protein encoder 得到 h_i 和 coord_i
  ↓
attention pool 得到 binding-site global context
  ↓
slot query j attends to residues
  ↓
输出每个 slot center 的 residue-weighted coordinate
```

一种稳定设计：

```text
w_ij = softmax_i score(h_i, slot_j)
pred_center_j = Σ_i w_ij * coord_i + offset_j
```

其中：

| 变量 | 含义 |
|---|---|
| `w_ij` | slot j 关注哪些 residue |
| `coord_i` | residue i 坐标 |
| `offset_j` | 从蛋白表面向外的偏移 |
| `pred_center_j` | 预测的 motif slot center |

这样做的好处：

```text
预测点会锚定在蛋白附近，不容易飞到无意义位置。
```

训练 loss：

```text
L_slot_center = smooth_l1(pred_slot_center_xyz, true_slot_center_xyz)
```

注意要处理平移：

```text
所有坐标都用 centered protein coordinate。
```

验收标准：

```text
pred_slot_center 到 true slot_center 的 RMSD 明显低于随机 surface baseline。
```

### Phase 3. 用 latent slot geometry 改造 pair_input

目标：让 `A_base/A_backbone/E` 不只看抽象 slot embedding，还看几何。

新增几何特征：

| 特征 | 形状 | 含义 |
|---|---:|---|
| `d_ij` | `N,M,1` | residue i 到 slot j 的距离 |
| `u_ij` | `N,M,3` | residue i 指向 slot j 的方向 |
| `rbf(d_ij)` | `N,M,K` | 距离 RBF 编码 |
| `slot_pos` | `N,M,1` | motif 相对位置 |

pair 输入变成：

```text
pair_input = [
  h_i,
  s_j,
  h_i * s_j,
  slot_pos,
  rbf(distance_i_j),
  direction_i_j
]
```

输出仍然保持：

```text
A_base(i,j)
A_backbone(i,j)
E(i,j,b)
```

核心公式不变：

```text
PWM[j,b] = softmax_b Σ_i A_base(i,j) * E(i,j,b)
```

验收标准：

```text
latent geometry 版本优于 current slot embedding 版本。
```

### Phase 4. 从 slot centers 升级到 helix axis

如果 Phase 2/3 有收益，再加 DNA axis。

目标：

```text
预测一条 DNA helix axis，
slot centers 沿 axis 排列。
```

简单参数化：

```text
axis_origin: 3
axis_direction: 3, normalized
slot_spacing: scalar or fixed
slot_center_j = axis_origin + alpha_j * axis_direction + local_offset_j
```

其中：

```text
alpha_j = j 在 motif 中的位置
```

优点：

| 优点 | 说明 |
|---|---|
| slot 顺序更稳定 | motif slot 不再是散点 |
| 几何更像 DNA | base positions 沿一条轴排列 |
| 参数更少 | 比直接预测 M 个独立点更稳 |

loss：

| loss | 作用 |
|---|---|
| `L_center` | slot center 接近真实 DNA |
| `L_axis_angle` | axis 方向接近真实 DNA 主轴 |
| `L_spacing` | 相邻 slot 距离合理 |
| `L_smooth` | slot center 不乱跳 |

验收标准：

```text
slot center RMSD 降低，A_base_AP / PWM 指标不下降。
```

### Phase 5. 再考虑 base-pair frame，不做全原子

如果 axis 还不够，可以加 local frame：

```text
slot_j = {
  center c_j,
  tangent t_j,
  normal n_j,
  binormal b_j
}
```

用途：

```text
区分 major groove / minor groove / backbone side。
```

但第一版不要直接预测：

```text
P, C1', N1/N9, full base atoms
```

原因：

| 不做全原子 DNA 的原因 |
|---|
| 难度会变成 docking / structure prediction |
| clash、bond length、base orientation 都会引入新问题 |
| 当前任务只需要 residue-slot 对齐和 PWM |
| reviewer 会要求和 docking 方法比较 |

推荐表述：

```text
We infer latent DNA slot geometry rather than full atomic DNA structures.
```

### Phase 6. 推理阶段怎么跑

推理输入仍然是：

```text
protein monomer PDB
motif length M
```

流程：

```text
protein monomer
  ↓
ESM2 + EGNN
  ↓
predict latent slot centers / axis
  ↓
compute residue-slot geometric features
  ↓
A_base / A_backbone / E
  ↓
PWM + site_prob
```

输出可以新增：

| 输出 | 用途 |
|---|---|
| `pred_slot_center_xyz` | 可视化 latent DNA |
| `pred_dna_axis` | 可视化 DNA binding direction |
| `A_base` | residue-slot base contact |
| `E` | residue-base preference |
| `PWM` | specificity prediction |
| `site_prob` | binding site prediction |

## 分阶段实验表

| 阶段 | 模型 | 是否用真实 DNA 几何 | 是否预测 latent geometry | 目的 |
|---|---|---|---|---|
| S0 | RBE-current | 否 | 否 | 当前 baseline |
| S1 | true-DNA oracle | 是 | 否 | 测几何 slot 上限 |
| S2 | latent center | 否 | M 个 center | 最小 latent DNA |
| S3 | latent axis | 否 | axis + centers | 加强 DNA 结构先验 |
| S4 | latent frame | 否 | base-pair frame | 区分 groove/backbone |

关键对照：

```text
S1 - S0 = 几何信息本身的价值
S2 - S0 = 模型能否从 monomer 推断有用 geometry
S1 - S2 = latent geometry 还有多少 gap
S3 - S2 = DNA axis prior 是否有用
```

## 实现优先级

| 优先级 | 任务 | 原因 |
|---|---|---|
| P0 | 存真实 `slot_center_xyz` | 后面所有 geometry 实验的基础 |
| P0 | true-DNA oracle | 先确认几何信息值得做 |
| P1 | latent center prediction | 最小可行 latent DNA |
| P1 | geometric pair features | 让 A/E 真正使用 latent geometry |
| P2 | latent axis | 给 slot 顺序加 DNA 先验 |
| P3 | latent frame | 更细解释 major/minor groove |

## 不要做的事

第一阶段不要做：

| 不做 | 原因 |
|---|---|
| full atomic DNA generation | 任务会变成 docking，复杂度暴涨 |
| diffusion DNA structure model | 和当前核心问题距离太远 |
| 大规模重构所有数据 schema | 先加最小字段验证 idea |
| 一开始就引入太多物理约束 | 容易过度设计，调不动 |

最短路径：

```text
slot_center label
  ↓
true-DNA oracle
  ↓
latent center predictor
  ↓
geometry pair features
```

## 关键风险

| 风险 | 应对 |
|---|---|
| latent DNA pose 学不准 | 先做 true-DNA oracle 和 geometry-supervised slot centers |
| monomer 信息不足 | 报告 oracle gap，说明问题难度 |
| benchmark 数据少 | partial PWM mask 提高覆盖率 |
| reviewer 认为只是应用 | 强调 missing partner inference 和 group-level benchmark |
| 和 DeepPBS 不可比 | 明确 DeepPBS 是 complex-input reference |

## 当前最重要的一句话

```text
RBE should not be framed as just another PWM predictor.
It should be framed as a monomer-only latent interaction-partner inference model
trained with privileged protein-DNA complex supervision.
```
