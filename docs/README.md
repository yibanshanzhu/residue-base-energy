# Docs

这个目录只保留与当前代码一致的文档。过时的计划、实验草稿和已被代码取代的分析不再保留。

## Active Docs

| 文档 | 用途 |
|---|---|
| [`current_status.md`](current_status.md) | 已完成、当前瓶颈、下一步、对外表述边界 |
| [`method.md`](method.md) | RBE 的核心变量、训练/推理边界、和 DeepPBS 的关系 |
| [`code_structure.md`](code_structure.md) | 当前代码模块职责和 CLI 边界 |
| [`../metadata/README.md`](../metadata/README.md) | source manifest schema 和数据准备命令 |
| [`../resources/deeppbs_curated/README.md`](../resources/deeppbs_curated/README.md) | DeepPBS fold 映射资源及未裁剪 PWM 入口 |
| [`../runbook.md`](../runbook.md) | 服务器数据准备、训练和评估步骤 |

## What To Read First

| 你要做什么 | 读哪里 |
|---|---|
| 看仓库现在到哪一步 | [`current_status.md`](current_status.md) |
| 跑数据准备 | [`../metadata/README.md`](../metadata/README.md) |
| 改代码 | [`code_structure.md`](code_structure.md) |
| 理解模型 | [`method.md`](method.md) |
