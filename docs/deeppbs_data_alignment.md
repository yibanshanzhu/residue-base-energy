# DeepPBS-Style Data Alignment

## 核心结论

DeepPBS 不是人工给每个 PWM column 找 DNA base。

它分两步：

| 步骤 | DeepPBS 怎么做 | 我们怎么复用 |
|---|---|---|
| PDB-chain 到 PWM | 用 curated mapping，例如 `run/folds/*.txt` 里的 `pdb_chain_pwmid.npz` | 已内置到 `resources/deeppbs_curated/folds/` |
| PWM 到 complex DNA | 在 DNA 正反链上滑窗，选分数最高的 start/strand | 先要求 window 有 contact，再按 sequence score 选 |

所以真正需要人工确认的只有：

```text
这个 PDB protein chain 对应哪个 TF/PWM
```

这一步 DeepPBS 已经整理过；我们把需要的 fold 文件和 PWM 数值直接 vendored 到本仓库。

## DeepPBS 的对齐逻辑

DeepPBS 代码位置：

| 文件 | 作用 |
|---|---|
| `DeepPBS/run/process_co_crystal.py` | 读取 `pdb_file,pwm_id`，处理 complex |
| `DeepPBS/deeppbs/load_PWM.py` | 从 `pwms.pickle` 读 PWM，并裁掉低信息量两端 |
| `DeepPBS/deeppbs/compute_Y_and_mask.py` | 对两条 DNA strand 分别对齐 PWM |
| `DeepPBS/deeppbs/align_PWM_seq.py` | ungapped alignment，最大化 IC-weighted PCC |

简化成一句话：

```text
trimmed PWM vs one-hot DNA sequence
        ↓
forward strand / reverse-complement strand 都试
        ↓
所有 start 都试
        ↓
选 IC-weighted PCC 最高的 start/strand
```

这个步骤本身不是 contact-aware。RBE V1 在此基础上加了一条结构约束：

```text
candidate start/strand
        ↓
先要求 A_contact_pos >= 1 且 site_pos >= 1
        ↓
再选 sequence score 最高的候选
```

也就是说，当前 RBE 的默认 `alignment_mode` 是：

```text
contact_constrained_pwm_dna
```

## 我们的自动流程

服务器上只需要本仓库：

```bash
cd ~/residue-base-energy
git pull
pip install -e .
```

用内置 curated fold 文件直接准备 20 个样本：

```bash
python scripts/prepare_deeppbs_smoke.py \
  --fold-file valid0.txt \
  --out-root data/deeppbs_smoke \
  --limit 20 \
  --min-contact-pairs 1 \
  --min-site-residues 1 \
  --device cuda
```

这个脚本会自动做：

| 自动步骤 | 输出 |
|---|---|
| 解析 `pdb_chain_pwmid.npz` | `sample_id / pdb_id / protein_chain / pwm_id` |
| 下载 RCSB PDB | `data/deeppbs_smoke/raw/pdb/*.pdb` |
| 读取内置 trimmed PWM | `data/deeppbs_smoke/raw/pwm/*.txt` |
| 调用 contact-constrained `process_complex` | `data/deeppbs_smoke/train/*.npz` |
| 过滤空 contact/site 标签 | 默认要求 `A_contact_pos >= 1` 且 `site_pos >= 1` |
| 写 manifest | `data/deeppbs_smoke/train_manifest.txt` |
| 记录失败样本 | `data/deeppbs_smoke/failed.tsv` |

然后训练：

```bash
python -m rbe.train \
  --manifest data/deeppbs_smoke/train_manifest.txt \
  --config configs/dna_v1.yaml \
  --out-dir runs/deeppbs_smoke \
  --epochs 50 \
  --device cuda
```

## 看结果是否可信

处理时每个成功样本都会打印：

```text
A_base_pos=...
A_backbone_pos=...
A_contact_pos=...
site_pos=...
alignment=contact_constrained_pwm_dna
chain=...
start=...
rc=...
score_mode=deeppbs_ic_pcc
contact_candidates=...
```

快速判断：

| 现象 | 意义 |
|---|---|
| `A_base_pos > 0` | 有 residue 读到 base，可用于 PWM/E 训练 |
| `A_contact_pos > 0` | 有 motif 区域接触，可用于 site 训练 |
| `site_pos > 0` | protein binding site 标签不是空的 |
| 很多样本失败 | PDB chain、PDB 格式、DNA 长度或 PWM 映射不适合当前 V1 |

## 和 DeepPBS 的差别

DeepPBS 允许部分 overlap，并用 mask 训练。

当前 RBE V1 更简单：

| 项 | DeepPBS | RBE V1 |
|---|---|---|
| PWM-DNA 对齐 | 可 partial overlap + mask | 需要处理后的 PWM 能完整落在某条 DNA chain 上 |
| PWM trimming | 有 | 内置 PWM 已按 IC `0.5` 阈值裁剪 |
| 对齐分数 | IC-weighted PCC | 默认先过滤无 contact window，再按 score 选 |
| DNA 坐标 | 训练和推理都用 complex | 只训练用 complex，推理不用 DNA |

所以这个流程不是完整复刻 DeepPBS 训练集处理，而是复用它最关键的 **PWM 映射**，并把 PWM-DNA 自动对齐升级成 contact-aware。
