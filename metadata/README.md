# Source Data Manifests

RBE 的样本源头由 source manifest 定义。

每一行定义一个可审计样本：

```text
PDB/mmCIF structure + motif database entry + chain selection + split
```

## Required Columns

| Column | Meaning |
|---|---|
| `sample_id` | Stable sample key, used as training cache sample name |
| `split` | Train/valid/id/custom split label |
| `structure_id` | Public structure id, usually PDB id |
| `structure_path` | Local PDB/mmCIF path; blank means prepare uses the structure cache |
| `structure_format` | `pdb` or `mmcif` |
| `protein_chains` | Comma-separated protein chains; blank means all protein chains |
| `dna_chains` | Comma-separated DNA chains; blank means all DNA chains |
| `motif_id` | Motif id from the motif database |
| `motif_source` | Motif database name, for example `JASPAR` or `HOCOMOCO` |
| `motif_version` | Motif database version |
| `motif_path` | Local PWM file path |
| `notes` | Optional provenance notes |

## Generate Training Cache

```bash
python scripts/prepare_source_manifest.py \
  --source-manifest metadata/samples.example.tsv \
  --out-root data/example_from_sources \
  --download-structures \
  --device cuda
```

This writes:

```text
data/example_from_sources/processed/*.npz
data/example_from_sources/processed_manifest.txt
data/example_from_sources/sample_table.tsv
data/example_from_sources/failed.tsv
```

## Import DeepPBS Folds As Source Manifests

For full PWM targets, first download untrimmed motif sources:

```bash
python scripts/download_motif_sources.py \
  --out-root resources/motif_sources \
  --motif-index resources/motif_sources/motif_index.tsv
```

Convert DeepPBS fold files with the untrimmed motif index:

```bash
python scripts/import_deeppbs_source_manifest.py \
  --fold-file valid0.txt \
  --output metadata/deeppbs_valid0_sources.tsv \
  --structure-format mmcif \
  --motif-index resources/motif_sources/motif_index.tsv
```

Then prepare from the generated source manifest:

```bash
python scripts/prepare_source_manifest.py \
  --source-manifest metadata/deeppbs_valid0_sources.tsv \
  --out-root data/deeppbs_valid0_sources \
  --download-structures \
  --device cuda
```
