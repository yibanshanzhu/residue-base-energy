# Current Status

## One-Line State

RBE 已经有可运行的数据入口、训练入口和评估入口；当前主要问题是数据监督覆盖率和严格 benchmark 对齐还没完全解决。

## Completed

| 事项 | 状态 | 位置 |
|---|---|---|
| source manifest 数据源头 | done | [`metadata/README.md`](../metadata/README.md) |
| PDB/mmCIF 结构解析 | done | `rbe.data.structure_io` |
| contact-constrained PWM-DNA alignment | done | `rbe.data.alignment_selection` |
| base/backbone/contact/site labels | done | `rbe.data.contact_labels` |
| single complex 到 training cache | done | `rbe.data.processed_sample` |
| source manifest 批量 prepare | done | `rbe.data.source_prepare` |
| DeepPBS fold 转 source manifest | done | `scripts/import_deeppbs_source_manifest.py` |
| manifest 评估 | done | `rbe.eval.evaluate_manifest` |
| DeepPBS prediction 对齐到 RBE slots | partial | `rbe.eval.deeppbs_alignment` |
| 代码职责拆分 | done | [`code_structure.md`](code_structure.md) |

## Current Bottleneck

| 问题 | 影响 |
|---|---|
| PDB/mmCIF 可见 DNA 片段短于 motif PWM | 一批样本无法生成完整 `slot_to_dna_index` 和 `A_*_label` |
| 部分 DNA residue 缺 base atoms | contact label 不能直接计算 |
| DeepPBS same-subset rerun 没完全闭合 | 不能声称严格超过 DeepPBS |

当前 failure 统计入口：

| 文件 | 内容 |
|---|---|
| [`../reports/prepare_failures/failure_summary.tsv`](../reports/prepare_failures/failure_summary.tsv) | failure label 汇总 |
| [`../reports/prepare_failures/failure_details.tsv`](../reports/prepare_failures/failure_details.tsv) | 单样本 failure 明细 |

## Next Work

| 优先级 | 任务 | 验收标准 |
|---:|---|---|
| P0 | partial PWM/masked supervision | `dna_shorter_than_pwm` 样本能保留可见 DNA overlap 部分 |
| P0 | loss/eval 支持 `pwm_mask`、`A_mask` | PWM 和 contact metrics 只在有监督位置计算 |
| P1 | missing base atom 样本处理 | 缺失 atom 不被当成 negative contact |
| P1 | DeepPBS same-subset rerun | 同一批 samples、同一套 PWM metrics 可比较 |
| P2 | group-level benchmark | 去掉 same-PDB / same-PDB+PWM 重复计权干扰 |

## Current Claim Boundary

| 可以说 | 不能说 |
|---|---|
| RBE 在 contact-valid subset 上有 PWM/site 信号 | RBE 已严格超过 DeepPBS |
| RBE 推理时不需要 DNA 坐标 | RBE 和 DeepPBS 是完全相同任务 |
| source manifest 已经把数据源头切到 PDB/mmCIF + motif database | 当前 benchmark 已完全闭合 |

## Active Reading Path

| 目的 | 文档 |
|---|---|
| 快速上手 | [`../README.md`](../README.md) |
| 方法定义 | [`method.md`](method.md) |
| 代码边界 | [`code_structure.md`](code_structure.md) |
| 数据 manifest | [`../metadata/README.md`](../metadata/README.md) |
| 历史记录 | [`archive/README.md`](archive/README.md) |
