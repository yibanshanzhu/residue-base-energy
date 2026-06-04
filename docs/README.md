# Residue-Base Energy Docs

这个目录记录 RBE 的方法来源、数据处理、benchmark 口径和当前缺口。首次阅读建议先看根目录 [`README.md`](../README.md)，再按下面顺序进入细节。

## 推荐阅读顺序

| 顺序 | 文档 | 适合什么时候读 |
|---:|---|---|
| 1 | [`idea_from_prior_work.md`](idea_from_prior_work.md) | 想理解 RBE 为什么学习 `A_base(i,j)`、`A_backbone(i,j)` 和 `E(i,j,b)` |
| 2 | [`deeppbs_data_alignment.md`](deeppbs_data_alignment.md) | 要准备 DeepPBS curated 数据、理解 PWM-DNA 自动对齐逻辑 |
| 3 | [`benchmark_gap_and_plan.md`](benchmark_gap_and_plan.md) | 要看当前 benchmark 结果、缺口和后续严格对比计划 |
| 4 | [`prepare_failure_reason_analysis.md`](prepare_failure_reason_analysis.md) | 要分析 prepare 失败原因、判断哪些样本值得救 |

## 快速任务入口

| 任务 | 入口 |
|---|---|
| 安装、单样本 smoke test、训练、推理、评估 | [`../README.md`](../README.md) |
| 跑 SMAD3 1OZJ 示例 | [`../examples/smad3_1ozj/README.md`](../examples/smad3_1ozj/README.md) |
| 查看 vendored DeepPBS curated fold/PWM 资源 | [`../resources/deeppbs_curated/README.md`](../resources/deeppbs_curated/README.md) |
| 准备 DeepPBS curated `.npz` 训练样本 | [`deeppbs_data_alignment.md`](deeppbs_data_alignment.md) |
| 检查 independent benchmark 口径 | [`benchmark_gap_and_plan.md`](benchmark_gap_and_plan.md) |

## 当前文档边界

这些文档偏向研究和实验记录，不是完整论文。当前最重要的约束是：

```text
RBE 已在 DeepPBS-derived contact-valid subset 上显示信号，
但严格结论需要 DeepPBS 在同一 subset 上 rerun 后再比较。
```

对外报告时应使用 `contact-valid subset` 口径，避免把无法生成 residue-base contact label 的样本当成负样本。
