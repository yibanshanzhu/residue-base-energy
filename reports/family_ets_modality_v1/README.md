# ETS ESM/Structure Modality Ablation v1

## Question

本实验检验：在 held-out ETS protein 上，蛋白结构或 ESM2 是否为 binding
specificity 提供独立、可泛化的增量信息。目标不是寻找最低 PWM MAE，而是隔离
信息来源并检查结果能否跨 UniProt 泛化。

## Protocol

| Item | Value |
|---|---:|
| Complexes | 26 |
| Independent UniProt groups | 12 |
| PWM window | fixed 8 bp |
| Split | leave-one-UniProt-out |
| Validation | one independent UniProt per fold |
| Aggregation | structures within UniProt, then 12 groups equally |
| Seed | 7 |

| Variant | ESM2 | Amino-acid identity | Protein geometry |
|---|---:|---:|---:|
| full | yes | yes | yes |
| esm_only | yes | yes | no |
| structure_only | no | yes | yes |
| family_mean | no | no | no |

三个神经网络具有完全相同的参数形状和数量。关闭 ESM 时使用同形状零特征；关闭
结构时将半径归零并移除结构边，但保留 EGNN 的逐节点变换。所有变体共享训练
fold family prior、loss、优化器、epoch、validation checkpoint selection 和
validation-only residual calibration。

训练在 node5 上使用 commit `e78a7f3`；多指标评估使用 commit `2e392c2`。24/24
日志均达到 epoch 100，未发现 traceback、OOM 或 RuntimeError；48 tests passed。

## Group-Equal Results

| Method | PWM MAE ↓ | PWM KL ↓ | PWM IC-PCC ↑ | A-base AP ↑ |
|---|---:|---:|---:|---:|
| family mean | **0.3062** | **0.1632** | **0.9340** | - |
| nearest ESM | 0.3086 | 0.3278 | 0.9271 | - |
| structure only | 0.3117 | 0.1895 | 0.9220 | 0.5290 |
| ESM only | 0.3150 | 0.3012 | 0.9260 | 0.4956 |
| full RBE | 0.3202 | 0.3483 | 0.9211 | **0.6093** |

## Specificity Evidence

下表使用 `delta = ablated/model MAE - reference MAE`。Full-reference 中正值表示
加入被消融模态后 Full 更好；prior-reference 中负值表示该模态模型优于 family
prior。

| Question | Comparison | Mean delta | 95% paired CI | Sign-flip p | Result |
|---|---|---:|---:|---:|---|
| structure given ESM | esm_only - full | -0.0052 | [-0.0498, 0.0257] | 0.9209 | unsupported |
| ESM given structure | structure_only - full | -0.0085 | [-0.0471, 0.0305] | 0.6787 | unsupported |
| structure without PLM | structure_only - family_mean | +0.0055 | [-0.0294, 0.0395] | 0.7695 | unsupported |
| ESM without geometry | esm_only - family_mean | +0.0088 | [-0.0287, 0.0485] | 0.6738 | unsupported |

MAE、KL、IC-PCC 中没有一个正向模态增量同时满足预先规定的两个判据：95% CI
不跨零且 exact sign-flip `p <= 0.05`。ESM-only 的 KL 反而显著差于 family mean：
`delta=+0.1379`, CI `[0.0101, 0.3555]`, `p=0.0459`。

因此，本实验**没有证明**结构或 ESM 对 held-out ETS specificity 提供独立且稳定的
预测增量。这个结论是“当前证据不支持”，不是“结构或 ESM 不含 specificity
信息”。

## Mechanistic Evidence

Full RBE 的 A-base AP 为 `0.6093`，高于 structure-only 的 `0.5290` 和 ESM-only
的 `0.4956`。相对 ESM-only，Full 在8/12组更高，配对 AP delta
`(esm_only - full)=-0.1138`，bootstrap CI `[-0.2686,-0.0086]`，但 sign-flip
`p=0.0747`；相对 structure-only 的 CI 跨0、`p=0.4673`。因此 contact
localization 结果是提示性证据，没有通过预设的双重统计门槛，也没有转化为更好的
held-out PWM。

Full 相对 ESM-only 在12组中赢8、输3、平1，中位 delta 为 `+0.0102`，但均值 delta
为 `-0.0052`。主要原因是少数大失败，例如 SPI1/P17433：Full MAE `0.6223`，而
ESM-only 为 `0.4080`、family mean 为 `0.4889`。因此当前最具体的诊断不是“模型
完全忽略结构”，而是：

> 结构通道与更高的 contact localization 均值相关，并使 Full 在多数 UniProt 上
> 相对 ESM-only 获得小幅 PWM 改善；但联合 residual 在少数蛋白上不稳定，足以
> 抵消这些收益。

## Claim Boundary

当前可以说：

> 在12个 held-out ETS UniProt 的单 seed 探索实验中，Full 呈现更高的机制定位
> 均值，但统计证据尚不充分，也没有获得超出 ESM-only 或 family prior 的稳定
> specificity 增量。

当前不能说：

- 结构对 binding specificity 没有信息；
- RBE 已退化为普通 PLM 模型；
- family mean 在更大数据或其他 TF 家族上必然更好。

主要限制是只有12个独立蛋白、单个训练 seed、每折仅一个 validation UniProt，以及
显著的蛋白间效应异质性。

## Artifacts

| File | Content |
|---|---|
| `summary.tsv` | group-equal method/metric summary |
| `per_group.tsv` | selected metrics per method and held-out UniProt |
| `full_reference_paired_metrics.tsv` | conditional specificity/contact comparisons |
| `prior_reference_paired_metrics.tsv` | independent comparisons against family prior |
| `esm_only_residual_scales.tsv` | validation-only ESM-only scale selection |
| `structure_only_residual_scales.tsv` | validation-only structure-only scale selection |
