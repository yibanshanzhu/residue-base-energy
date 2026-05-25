# Residue-Base Energy Model 


## 目标

| 阶段 | 输入 | 输出 |
|---|---|---|
| 训练 | protein-DNA complex PDB + aligned PWM | `A(i,j)`、PWM、protein-site |
| 推理 | protein monomer PDB + motif length `M` | `A(i,j)`、`E(i,j,b)`、PWM、protein-site |

核心对象：

```text
A(i,j): residue i 和 motif slot j 的可微 contact/alignment
E(i,j,b): residue i 对 slot j 上 base b 的能量/偏好
PWM[j,b] = softmax_b Σ_i A(i,j) * E(i,j,b)
```

## 安装

```bash
cd /Users/qd/code/xbind/residue-base-energy
conda activate dl_hw
pip install -e .
```

## 数据处理

输入 PWM 必须已经和 DNA motif strand 对齐。默认把 PWM 第 `j` 行映射到选中 DNA residue list 的第 `dna_start_index + j` 个 nucleotide。

```bash
python -m rbe.data.process_complex \
  --pdb path/to/complex.pdb \
  --pwm path/to/aligned_pwm.txt \
  --protein-chains A \
  --dna-chains B \
  --dna-start-index 0 \
  --output data/train/sample.npz \
  --device cuda
```

也可以显式指定 motif slot 到 DNA residue index：

```bash
python -m rbe.data.process_complex \
  --pdb complex.pdb \
  --pwm aligned_pwm.txt \
  --protein-chains A \
  --dna-chains B,C \
  --slot-to-dna-index 3,4,5,6,7,8,9,10 \
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
| `A_label` | `N,M` | residue-base contact label |
| `site_label` | `N` | residue 是否接触 DNA |
| `slot_to_dna_index` | `M` | motif slot 对应 DNA residue index |

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

