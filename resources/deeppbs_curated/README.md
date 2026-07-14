# DeepPBS Curated Mapping Resources

这个目录保留 DeepPBS fold 映射和旧 PWM，用于追溯 motif ID。

| Path | Content |
|---|---|
| `folds/*.txt` | DeepPBS curated `pdb_chain_pwmid.npz` fold entries |
| `pwms/*.txt` | 从 DeepPBS `pwms.pickle` 导出的 trimmed PWM，仅用于 ID 枚举和历史核对 |
| `summary.tsv` | Resource counts |

`pwms/*.txt`使用 DeepPBS 的 trimming 规则：从两端删除低信息量 columns，直到端点 information content `> 0.5`。

当前数据流程禁止把这些 trimmed PWM 写入 source manifest。先下载未裁剪公共 PWM 并生成 index：

```bash
python scripts/download_motif_sources.py \
  --out-root resources/motif_sources \
  --motif-index resources/motif_sources/motif_index.tsv
```

再用 fold 和 index 生成 source manifest。`--motif-index`只读取已有 index，不会重复下载：

```bash
python scripts/import_deeppbs_source_manifest.py \
  --fold-file valid0.txt \
  --output metadata/deeppbs_valid0_sources.tsv \
  --motif-index resources/motif_sources/motif_index.tsv
```

Then generate training cache from the source manifest:

```bash
python scripts/prepare_source_manifest.py \
  --source-manifest metadata/deeppbs_valid0_sources.tsv \
  --out-root data/deeppbs_valid0_sources \
  --download-structures
```
