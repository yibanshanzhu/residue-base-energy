# Code Structure

## Data Pipeline

| 模块 | 职责 |
|---|---|
| `rbe.data.source_manifest` | source manifest TSV schema、读写、路径解析 |
| `rbe.data.structure_types` | `AtomRecord`、`ResidueRecord` 基础类型 |
| `rbe.data.structure_io` | PDB/mmCIF atom records 解析 |
| `rbe.data.residue_select` | chain 解析、protein/DNA residue 选择、序列转换 |
| `rbe.data.atom_geometry` | residue heavy atom、base/backbone atom、CA/centroid 坐标 |
| `rbe.data.alignment` | PWM 与 DNA 序列候选枚举和序列打分 |
| `rbe.data.alignment_selection` | manual/contact-constrained PWM-DNA alignment 选择 |
| `rbe.data.contact_labels` | `A_base_label`、`A_backbone_label`、`A_contact_label`、`site_label` |
| `rbe.data.processed_sample` | 单个 complex + PWM 到训练样本数组 |
| `rbe.data.source_prepare` | source manifest 批量生成 training cache、过滤、报告 |
| `rbe.data.shared_cache` | 跨 DeepPBS folds 去重生成 shared cache，再写各 fold processed manifest |
| `rbe.data.deeppbs_curated` | DeepPBS fold entry 和资源路径解析 |

## Evaluation Pipeline

| 模块 | 职责 |
|---|---|
| `rbe.eval.io` | NPZ/PWM/manifest/prediction path IO |
| `rbe.eval.pair_metrics` | 单个 target/pred pair 指标 |
| `rbe.eval.summary` | 多样本均值、global site diagnostic |
| `rbe.eval.reports` | TSV/JSON 报告写出 |
| `rbe.eval.prediction` | checkpoint 加载、单样本预测数组写出 |
| `rbe.eval.deeppbs_alignment` | DeepPBS prediction 到 RBE PWM slot 的方向和窗口对齐 |
| `rbe.eval.evaluate_manifest` | manifest 级评估编排 |

## CLI Boundary

| 脚本 | 职责 |
|---|---|
| `scripts/prepare_source_manifest.py` | 参数解析并调用 `rbe.data.source_prepare` |
| `scripts/prepare_deeppbs_shared_cache.py` | 一次生成 DeepPBS unique sample cache 和 11 个 fold manifests |
| `scripts/download_motif_sources.py` | 按 DeepPBS motif ID 下载未裁剪公共 PWM 并生成四列 index |
| `scripts/import_deeppbs_source_manifest.py` | 用 DeepPBS fold 和 motif index 生成 source manifest |
| `scripts/align_deeppbs_predictions_for_rbe_eval.py` | 参数解析并调用 `rbe.eval.deeppbs_alignment` |

规则：脚本只做 CLI；业务逻辑放在 `src/rbe` 下有名字的模块里。
