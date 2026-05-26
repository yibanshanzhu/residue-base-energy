# Residue-Base Energy Model 


## 目标

| 阶段 | 输入 | 输出 |
|---|---|---|
| 训练 | protein-DNA complex PDB + aligned PWM | `A_base/A_backbone/A_contact`、PWM、protein-site |
| 推理 | protein monomer PDB + motif length `M` | `A_base/A_backbone/A_contact`、`E(i,j,b)`、PWM、protein-site |

核心对象：

```text
A_base(i,j): residue i 是否读取 motif slot j 的碱基
A_backbone(i,j): residue i 是否接触 motif slot j 的糖-磷酸骨架
A_contact(i,j) = max(A_base(i,j), A_backbone(i,j))
E(i,j,b): residue i 对 slot j 上 base b 的能量/偏好
PWM[j,b] = softmax_b Σ_i A_base(i,j) * E(i,j,b)
```

思想来源和模型定义见：

| 文档 | 内容 |
|---|---|
| [`docs/idea_from_prior_work.md`](docs/idea_from_prior_work.md) | 说明本项目如何从 rCLAMPS、DeepPBS、EquiPPIS/EquiPNAS、MegSite 等方法抽象出来 |

## 安装

GPU 服务器推荐：

```bash
git clone https://github.com/yibanshanzhu/residue-base-energy.git
cd residue-base-energy
conda env create -f environment.gpu.yml
conda activate rbe_gpu
pip install -e .
```

如果服务器已有可用 GPU 环境，也可以只执行：

```bash
conda activate 你的GPU环境
pip install -e .
```

## GPU smoke test

```bash
python scripts/create_toy_npz.py --out-dir /tmp/rbe_toy
python -m rbe.train \
  --data-dir /tmp/rbe_toy \
  --config configs/dna_v1.yaml \
  --out-dir /tmp/rbe_run \
  --epochs 1 \
  --device cuda
```

## 数据处理

默认会自动把 PWM 对齐到选中 DNA chain 的 sequence：每条 DNA chain 的正向和反向互补都会尝试，选择 **IC-weighted log-likelihood** 最高的 `start/strand`，然后生成 `slot_to_dna_index`。

```bash
python -m rbe.data.process_complex \
  --pdb path/to/complex.pdb \
  --pwm path/to/pwm.txt \
  --protein-chains A \
  --dna-chains B \
  --output data/train/sample.npz \
  --device cuda
```

如需对比朴素 log-likelihood：

```bash
python -m rbe.data.process_complex \
  --pdb complex.pdb \
  --pwm pwm.txt \
  --protein-chains A \
  --dna-chains B \
  --alignment-score log_likelihood \
  --output data/train/sample.ll.npz
```

也可以手动覆盖 motif slot 到 DNA residue index：

```bash
python -m rbe.data.process_complex \
  --pdb complex.pdb \
  --pwm pwm.txt \
  --protein-chains A \
  --dna-chains B,C \
  --slot-to-dna-index 3,4,5,6,7,8,9,10 \
  --output data/train/sample.npz
```

或者手动指定连续 DNA 起点：

```bash
python -m rbe.data.process_complex \
  --pdb complex.pdb \
  --pwm pwm.txt \
  --protein-chains A \
  --dna-chains B \
  --dna-start-index 3 \
  --output data/train/sample.npz
```

输出 `npz`：

| 字段 | 形状 | 定义 |
|---|---:|---|
| `residue_ids` | `N` | chain/residue 编号 |
| `residue_aa` | `N` | one-letter AA |
| `residue_xyz` | `N,3` | Cα 坐标 |
| `residue_edges` | `2,E` | Cα 距离 `<14Å` 的 residue graph |
| `edge_attr` | `E,17` | 16 个 distance RBF + 1 个 sequence separation |
| `esm2_repr` | `N,1280` | frozen ESM2-t33 layer 33 hidden representation |
| `pwm_target` | `M,4` | A/C/G/T PWM |
| `A_base_label` | `N,M` | residue 是否接触第 `j` 个 nucleotide 的 base heavy atoms |
| `A_backbone_label` | `N,M` | residue 是否接触第 `j` 个 nucleotide 的 sugar/phosphate heavy atoms |
| `A_contact_label` | `N,M` | `max(A_base_label, A_backbone_label)` |
| `A_label` | `N,M` | 兼容旧字段，等同于 `A_base_label` |
| `site_label` | `N` | `max_j A_contact_label(i,j)` |
| `slot_to_dna_index` | `M` | motif slot 对应 DNA residue index |
| `alignment_*` | scalar | 自动/手动 PWM-DNA 对齐信息 |

训练 loss：

| loss | 作用 |
|---|---|
| `L_pwm` | 用预测 `A_base(i,j)` 门控 `E(i,j,b)` 还原 PWM |
| `L_pwm_teacher` | 用真实 `A_base_label(i,j)` 门控 `E(i,j,b)` 还原 PWM，让 `E` 被真实 base contact 锚住 |
| `L_A_base` | 监督 `A_base(i,j)` 接近真实 base contact |
| `L_A_backbone` | 监督 `A_backbone(i,j)` 接近真实 backbone contact |
| `L_site` | 监督 protein-side binding site |
| `L_sparse` | 防止 `A_contact(i,j)` 到处变大 |
| `L_noncontact` | 惩罚非 base-contact residue 对 PWM 的贡献 |

## 训练

```bash
python -m rbe.train \
  --data-dir data/train \
  --config configs/dna_v1.yaml \
  --out-dir runs/dna_v1 \
  --device cuda
```

单样本 overfit：

```bash
python -m rbe.train \
  --manifest data/one_sample.txt \
  --config configs/dna_v1.yaml \
  --out-dir runs/overfit \
  --epochs 200 \
  --device cuda
```

训练完成后，用 checkpoint 在同一个处理后样本上生成 `pred.npz`：

```bash
python -m rbe.eval.predict_npz \
  --sample data/train/sample.npz \
  --checkpoint runs/overfit/best.pt \
  --output runs/overfit/pred.npz \
  --device cuda
```

## 推理

推理脚本只接收 monomer PDB，不接收 DNA 坐标或 complex。

```bash
python -m rbe.predict \
  --pdb path/to/protein_monomer.pdb \
  --motif-length 10 \
  --checkpoint runs/dna_v1/best.pt \
  --output predictions/protein_pwm.npz \
  --device cuda
```

## 评估

```bash
python -m rbe.eval.evaluate_pwm \
  --target data/test/sample.npz \
  --pred predictions/sample_ours.npz \
  --baseline DeepPBS=predictions/sample_deeppbs.npz \
  --baseline rCLAMPS=predictions/sample_rclamps.npz
```

评估分三条线：

| 线 | 指标 |
|---|---|
| PWM | MAE、KL、IC-weighted PCC、RC-aware KL |
| protein site | AP、MCC、F1 |
| A map | AP、top-L precision |
