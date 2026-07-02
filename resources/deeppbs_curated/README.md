# DeepPBS Curated Mapping Resources

This directory vendors the small DeepPBS curated resources needed to import
DeepPBS folds into RBE source manifests.

| Path | Content |
|---|---|
| `folds/*.txt` | DeepPBS curated `pdb_chain_pwmid.npz` fold entries |
| `pwms/*.txt` | A/C/G/T PWM matrices exported from DeepPBS `pwms.pickle` |
| `summary.tsv` | Resource counts |

PWM files are trimmed with the DeepPBS rule: remove low-information columns from both ends until information content is `> 0.5`.

Runtime data prep uses these resources through source manifest conversion.

Convert fold entries to source manifest rows:

```bash
python scripts/import_deeppbs_source_manifest.py \
  --fold-file valid0.txt \
  --output metadata/deeppbs_valid0_sources.tsv
```

Then generate training cache from the source manifest:

```bash
python scripts/prepare_source_manifest.py \
  --source-manifest metadata/deeppbs_valid0_sources.tsv \
  --out-root data/deeppbs_valid0_sources \
  --download-structures
```
