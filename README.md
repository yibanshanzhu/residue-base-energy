# Residue-Base Energy

RBE 是一个从 protein monomer 预测 DNA motif/PWM 的模型。训练时使用 protein-DNA complex 生成监督信号；推理时只输入 protein structure 和 motif length。

## Current State

| 项 | 状态 |
|---|---|
| 数据源头 | PDB/mmCIF structure + motif database PWM + source manifest |
| 单样本处理 | 可用 |
| 批量数据准备 | 可用 |
| 训练/推理/评估 | 可用 |
| motif PWM | 只使用公共数据库的未裁剪完整 PWM |
| 结构监督 | `pwm_mask` 仅标记结构中可见的 motif columns |
| PWM MAE | 每个 sample 在 `pwm_mask=1` columns 上计算，再对 samples 求均值 |
| 严格 DeepPBS 对比 | 还没完全闭合 |

当前状态、做完/没做完的事见 [`docs/current_status.md`](docs/current_status.md)。

## Read First

| 目的 | 文档 |
|---|---|
| 当前状态和下一步 | [`docs/current_status.md`](docs/current_status.md) |
| 方法定义 | [`docs/method.md`](docs/method.md) |
| 代码结构 | [`docs/code_structure.md`](docs/code_structure.md) |
| 数据 manifest | [`metadata/README.md`](metadata/README.md) |
| DeepPBS vendored resources | [`resources/deeppbs_curated/README.md`](resources/deeppbs_curated/README.md) |

## Install

```bash
git clone https://github.com/yibanshanzhu/residue-base-energy.git
cd residue-base-energy
conda env create -f environment.gpu.yml
conda activate rbe_gpu
pip install -e .
```

已有合适 GPU 环境时：

```bash
pip install -e .
```

## Data

样本由 source manifest 定义：

```text
PDB/mmCIF structure + motif database PWM + chain selection + split
```

从 manifest 生成 training cache：

```bash
python scripts/prepare_source_manifest.py \
  --source-manifest metadata/samples.example.tsv \
  --out-root data/example_from_sources \
  --download-structures \
  --device cuda
```

先按 DeepPBS motif ID 下载未裁剪 PWM：

```bash
python scripts/download_motif_sources.py \
  --out-root resources/motif_sources \
  --motif-index resources/motif_sources/motif_index.tsv
```

再将 DeepPBS fold 转成 source manifest：

```bash
python scripts/import_deeppbs_source_manifest.py \
  --fold-file valid0.txt \
  --output metadata/deeppbs_valid0_sources.tsv \
  --structure-format mmcif \
  --motif-index resources/motif_sources/motif_index.tsv
```

再生成 training cache：

```bash
python scripts/prepare_source_manifest.py \
  --source-manifest metadata/deeppbs_valid0_sources.tsv \
  --out-root data/deeppbs_valid0_sources \
  --download-structures \
  --device cuda
```

## Single-Sample Smoke Test

```bash
mkdir -p /tmp/rbe_smoke/train
python -m rbe.data.process_complex \
  --pdb examples/smad3_1ozj/1ozj.pdb \
  --pwm examples/smad3_1ozj/smad3_hocomoco_pwm.txt \
  --protein-chains A \
  --dna-chains C,D \
  --output /tmp/rbe_smoke/train/smad3_1ozj_A.npz \
  --device cuda
```

## Train

```bash
python -m rbe.train \
  --manifest data/train_manifest.txt \
  --valid-manifest data/valid_manifest.txt \
  --config configs/dna_v1.yaml \
  --out-dir runs/dna_v1 \
  --device cuda
```

## Predict

```bash
python -m rbe.predict \
  --pdb path/to/protein_monomer.pdb \
  --motif-length 10 \
  --checkpoint runs/dna_v1/best.pt \
  --output predictions/protein_pwm.npz \
  --device cuda
```

## Evaluate

```bash
python -m rbe.eval.evaluate_manifest \
  --manifest data/test_manifest.txt \
  --pred-dir runs/dna_v1/preds \
  --checkpoint runs/dna_v1/best.pt \
  --device cuda
```

Outputs:

```text
runs/dna_v1/preds/eval_per_sample.tsv
runs/dna_v1/preds/eval_summary.tsv
```
