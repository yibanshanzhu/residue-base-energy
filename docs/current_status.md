# Current Status

## One-Line State

RBE 已闭合未裁剪 PWM、canonical 全局评估和 ETS 家族内机制评估；当前证据支持
residue-base 机制定位，但不支持其 PWM 精度超过简单家族先验。

## Completed

| 事项 | 状态 | 位置 |
|---|---|---|
| 未裁剪 motif source | done | [`../metadata/README.md`](../metadata/README.md) |
| shared canonical cache | done | unique sample 只 prepare 一次 |
| 显式 PWM orientation contract | done | `canonical` / `family_reference:*` |
| full PWM per-sample metrics | done | `rbe.eval.metrics` |
| ETS fixed-core benchmark | done | `resources/family_benchmarks/ets_v1` |
| UniProt-disjoint LOPO folds | done | 26 structures / 12 groups |
| family mean + nearest-ESM baselines | done | `rbe.eval.family_baselines` |
| group-equal evaluation | done | `rbe.eval.family_evaluation` |
| family prior + residual energy | done | `rbe.data.family_prior` + model |
| validation-only residual calibration | done | `rbe.eval.family_residual_calibration` |
| residue mechanism ablations | done | `rbe.eval.mechanism_ablation` |
| node5 end-to-end verification | done | 44 tests; 24 logs reached epoch 100 |

## ETS Result

| Method | PWM MAE | A-base AP |
|---|---:|---:|
| full RBE | 0.3202 | 0.6093 |
| PWM-only RBE | 0.3143 | 0.0289 |
| nearest ESM | 0.3086 | - |
| family mean | **0.3062** | - |
| uniform gate | 0.4678 | - |
| shuffled energy | 0.4799 | - |

完整协议、逐组结果和 claim boundary 见
[`../reports/family_ets_v1/README.md`](../reports/family_ets_v1/README.md)。

## Current Claim Boundary

| 可以说 | 不能说 |
|---|---|
| RBE contact map 在 held-out ETS proteins 上有结构信号 | RBE 已超过 family mean PWM baseline |
| PWM 依赖 residue 定位与 residue-energy pairing | 消融退化等于 residue energy 已完全符合真实物理能量 |
| 同一 UniProt 的重复结构被组内平均 | 26 structures 等于 26 个独立蛋白样本 |
| family target 使用固定方向和固定 8 bp core | ETS 结果已代表 unseen-family 泛化 |

## Current Bottleneck

| 问题 | 影响 |
|---|---|
| 仅 12 个独立 ETS UniProt | 外层均值方差高 |
| residual 在少数组过度修正 | 平均 PWM MAE 落后 family prior `0.0140` |
| 单 validation group 的 scale 泛化有限 | validation calibration 未改善外层均值 |

下一步必须扩大独立 protein groups，或在 outer-train groups 内做真正 nested group-CV
来约束 residual；不能通过 test-set 调 scale 来制造正结果。

## Reading Path

| 目的 | 文档 |
|---|---|
| 快速上手 | [`../README.md`](../README.md) |
| canonical benchmark | [`../runbook.md`](../runbook.md) |
| ETS family benchmark | [`../runbook_family_ets.md`](../runbook_family_ets.md) |
| 方法定义 | [`method.md`](method.md) |
