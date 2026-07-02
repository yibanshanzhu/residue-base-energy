# Current Status

## One-Line State

RBE 已经有可运行的数据入口、训练入口和评估入口；partial PWM/masked supervision 已进入代码，当前需要重建数据并重新跑 benchmark。

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
| partial PWM/masked supervision | done | `pwm_mask` + `slot_to_dna_index=-1` |
| manifest 评估 | done | `rbe.eval.evaluate_manifest` |
| DeepPBS prediction 对齐到 RBE slots | partial | `rbe.eval.deeppbs_alignment` |
| 代码职责拆分 | done | [`code_structure.md`](code_structure.md) |

## Current Bottleneck

| 问题 | 影响 |
|---|---|
| 旧 training cache 需要重建 | 新 schema 才会包含 `pwm_mask` 和 partial `slot_to_dna_index` |
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
| P0 | 重建 source manifest training cache | `dna_shorter_than_pwm` 样本能保留可见 DNA overlap 部分 |
| P0 | 重新训练和评估 | 输出新的 `eval_summary.tsv` |
| P1 | 统计新 prepare failure | 确认 missing base atom 和 DNA shorter failure 下降 |
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
