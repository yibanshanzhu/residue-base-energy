# SMAD3 1OZJ Example

这是一个真实 protein-DNA complex 小测试样本。

| 文件 | 来源 | 说明 |
|---|---|---|
| `1ozj.pdb` | RCSB PDB | SMAD3/DNA complex |
| `smad3_hocomoco_pwm.txt` | DeepPBS bundled HOCOMOCO PWM key `SMAD3_HUMAN.H11MO.0.B` | A/C/G/T 四列 PWM |

推荐先用 chain A 作为 protein，chain C,D 作为 DNA：

```bash
mkdir -p data/train

python -m rbe.data.process_complex \
  --pdb examples/smad3_1ozj/1ozj.pdb \
  --pwm examples/smad3_1ozj/smad3_hocomoco_pwm.txt \
  --protein-chains A \
  --dna-chains C,D \
  --output data/train/smad3_1ozj_A.npz \
  --device cuda
```

预期预处理输出应类似：

```text
alignment=contact_constrained_pwm_dna
chain=C
start=2
rc=True
score_mode=ic_log_likelihood
contact_candidates=16/16
A_base_pos=9
A_backbone_pos=13
A_contact_pos=19
site_pos=13
```

然后单样本 overfit：

```bash
python -m rbe.train \
  --manifest examples/smad3_1ozj/one_sample_manifest.txt \
  --config configs/dna_v1.yaml \
  --out-dir runs/smad3_1ozj_overfit \
  --epochs 200 \
  --device cuda
```

如果没有 manifest，也可以直接：

```bash
python -m rbe.train \
  --data-dir data/train \
  --config configs/dna_v1.yaml \
  --out-dir runs/smad3_1ozj_overfit \
  --epochs 200 \
  --device cuda
```
