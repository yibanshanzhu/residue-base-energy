# Current Status

## One-Line State

RBE 已具备完整的数据准备、训练、推理和评估入口；motif 源头已切换为公共数据库的未裁剪 PWM，当前需要基于新数据源重建数据并重跑 benchmark。

## Completed

| 事项 | 状态 | 位置 |
|---|---|---|
| source manifest 数据源头 | done | [`metadata/README.md`](../metadata/README.md) |
| 按 DeepPBS motif ID 下载未裁剪 PWM | done | `scripts/download_motif_sources.py` |
| fold + motif index 转 source manifest | done | `scripts/import_deeppbs_source_manifest.py` |
| PDB/mmCIF 结构解析 | done | `rbe.data.structure_io` |
| contact-constrained PWM-DNA alignment | done | `rbe.data.alignment_selection` |
| base/backbone/contact/site labels | done | `rbe.data.contact_labels` |
| single complex 到 training cache | done | `rbe.data.processed_sample` |
| source manifest 批量 prepare | done | `rbe.data.source_prepare` |
| DeepPBS folds shared cache | done | unique sample 只 prepare 一次，fold membership 单独写 manifest |
| partial structure supervision | done | `pwm_mask` + `slot_to_dna_index=-1` |
| manifest 评估 | done | `rbe.eval.evaluate_manifest` |
| masked PWM MAE 的 per-sample 指标 | done | `rbe.eval.metrics` + `rbe.eval.summary` |
| DeepPBS prediction 对齐到 RBE slots | partial | `rbe.eval.deeppbs_alignment` |

## Current Metric Definition

| 项 | 当前行为 |
|---|---|
| PWM target | 公共数据库的未裁剪完整 PWM |
| 单样本 MAE | 每个有效 column 的四碱基 L1 之和，再对该 sample 的 `pwm_mask=1` columns 求均值 |
| 数据集汇总 | 先得到每个 sample 的指标，再对 samples 求均值 |
| `pwm_mask` | 用于有结构依据的 contact/map 监督与评估，同时限定 PWM MAE 的 columns |
| 其他 PWM 指标 | KL、IC-PCC、RC-aware KL 仍在完整 PWM 上按 sample 计算 |

`masked-pwm-per-sample` 分支已经实现上述 MAE 口径，不包含跨样本 pooling columns。

## Current Bottleneck

| 问题 | 影响 |
|---|---|
| 旧 training cache 需要重建 | 旧缓存内仍可能保存 DeepPBS 裁剪后的 PWM target |
| 部分 DNA residue 缺 base atoms | `A_base_mask=0`，base contact 不参与监督 |
| DeepPBS same-subset rerun 没完全闭合 | 不能声称严格超过 DeepPBS |

当前 failure 统计入口：

| 文件 | 内容 |
|---|---|
| [`../reports/prepare_failures/failure_summary.tsv`](../reports/prepare_failures/failure_summary.tsv) | failure label 汇总 |
| [`../reports/prepare_failures/failure_details.tsv`](../reports/prepare_failures/failure_details.tsv) | 单样本 failure 明细 |

## Next Work

| 优先级 | 任务 | 验收标准 |
|---:|---|---|
| P0 | 用未裁剪 motif index 重建 source manifest 和 training cache | manifest 的 `motif_path` 全部指向未裁剪 PWM |
| P0 | 重新训练和评估 | 输出新的 `eval_summary.tsv` |
| P1 | DeepPBS same-subset rerun | 同一批 samples、同一套 PWM metrics 可比较 |
| P2 | group-level benchmark | 去掉 same-PDB / same-PDB+PWM 重复计权干扰 |

## Current Claim Boundary

| 可以说 | 不能说 |
|---|---|
| RBE 支持完整 PWM target 与部分结构监督 | RBE 已严格超过 DeepPBS |
| RBE 推理时不需要 DNA 坐标 | RBE 和 DeepPBS 是完全相同任务 |
| 新 source manifest 流程只接受未裁剪 motif index | 旧 benchmark 已代表新数据源结果 |

## Active Reading Path

| 目的 | 文档 |
|---|---|
| 快速上手 | [`../README.md`](../README.md) |
| 方法定义 | [`method.md`](method.md) |
| 代码边界 | [`code_structure.md`](code_structure.md) |
| 数据 manifest | [`../metadata/README.md`](../metadata/README.md) |
