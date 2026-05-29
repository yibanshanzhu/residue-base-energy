# Prepare 失败原因统计计划

## 目标

先做统计，不急着救样本。

我们要回答三个问题：

| 问题 | 目的 |
|---|---|
| 失败样本一共有多少 | 知道数据损失规模 |
| 每类失败各占多少 | 判断主要瓶颈在哪里 |
| 哪些失败值得救 | 避免把标签不成立的样本硬塞进训练 |

当前统计对象：

```text
data/deeppbs_train*_contactalign/failed.tsv
data/deeppbs_valid*_contactalign/failed.tsv
data/deeppbs_id_contactalign/failed.tsv
```

## 失败分类

| 类别 | 典型报错 | 本质 | 是否优先救 |
|---|---|---|---|
| `missing_dna_base_atoms` | `has no base heavy atoms` | DNA residue 缺碱基原子，无法算 `A_base` | 是 |
| `missing_dna_backbone_atoms` | `has no backbone heavy atoms` | DNA residue 缺骨架原子，无法算 `A_backbone` | 是 |
| `dna_shorter_than_pwm` | `No selected DNA chain is long enough for PWM length` | PDB 可见 DNA 短于 PWM | 暂缓 |
| `no_contact_alignment` | `No PWM-DNA alignment candidate passed contact constraints` | 找不到既匹配 PWM 又接触蛋白的 window | 部分可救 |
| `no_protein_or_dna` | `No protein residues found` / `No DNA residues found` | PDB chain 或分子识别失败 | 可救 |
| `pdb_download_fail` | `HTTPError` / `URL Error` | 下载问题 | 可救 |
| `pwm_missing` | `Curated PWM not found` | 本地 PWM 缺失 | 可救 |
| `skip_multi_char_chain` | `skip_multi_char_chain` | chain ID 解析限制 | 可救 |
| `esm_failure` | ESM 下载/加载/显存错误 | 特征提取问题 | 可救 |
| `other` | 其他 traceback | 需要单独看 | 待定 |

## 先跑总量统计

在服务器仓库根目录运行：

```bash
for d in data/deeppbs_*_contactalign; do
  [ -f "$d/train_manifest.txt" ] || continue
  [ -f "$d/failed.tsv" ] || continue
  ok=$(grep -cv '^[[:space:]]*$' "$d/train_manifest.txt")
  fail=$(tail -n +2 "$d/failed.tsv" | grep -cv '^[[:space:]]*$')
  echo -e "$(basename "$d")\tsuccess=${ok}\tfailure=${fail}"
done
```

这个只回答：

```text
每个 split 成功多少，失败多少
```

2026-05-29 当前统计结果：

| split | success | failure |
|---|---:|---:|
| `deeppbs_id_contactalign` | 112 | 18 |
| `deeppbs_smoke_contactalign` | 20 | 3 |
| `deeppbs_train0_contactalign` | 403 | 16 |
| `deeppbs_train1_contactalign` | 387 | 32 |
| `deeppbs_train2_contactalign` | 394 | 25 |
| `deeppbs_train3_contactalign` | 392 | 27 |
| `deeppbs_train4_contactalign` | 391 | 28 |
| `deeppbs_valid0_contactalign` | 89 | 15 |

注意：如果用 `wc -l failed.tsv - 1` 统计，可能因为文件最后一行没有换行而少算 1。这里以 `failure_details.tsv` 明细为准，`id.txt` 是 `success=112, failure=18`，合计 130。

## 失败原因自动分类

先不写进仓库代码，用临时 Python 命令统计即可。

```bash
mkdir -p reports/prepare_failures

python - <<'PY'
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

failed_files = sorted(Path("data").glob("deeppbs_*_contactalign/failed.tsv"))

patterns = [
    ("missing_dna_base_atoms", ["has no base heavy atoms"]),
    ("missing_dna_backbone_atoms", ["has no backbone heavy atoms"]),
    ("dna_shorter_than_pwm", ["No selected DNA chain is long enough for PWM length"]),
    ("no_contact_alignment", ["No PWM-DNA alignment candidate passed contact constraints"]),
    ("no_protein_or_dna", ["No protein residues found", "No DNA residues found"]),
    ("pdb_download_fail", ["HTTPError", "URLError", "urlopen", "download"]),
    ("pwm_missing", ["Curated PWM not found"]),
    ("skip_multi_char_chain", ["skip_multi_char_chain"]),
    ("esm_failure", ["esm2", "PytorchStreamReader", "load_state_dict_from_url"]),
]


def classify(reason: str) -> str:
    for label, needles in patterns:
        if any(needle in reason for needle in needles):
            return label
    return "other"


summary = []
details = []
global_counts = Counter()

for path in failed_files:
    split = path.parent.name
    counts = Counter()
    examples = defaultdict(list)
    with path.open() as handle:
        header = next(handle, "")
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            entry, reason = line.split("\t", 1)
            label = classify(reason)
            counts[label] += 1
            global_counts[label] += 1
            if len(examples[label]) < 5:
                examples[label].append(entry)
            details.append((split, entry, label, reason))

    for label, n in sorted(counts.items()):
        summary.append((split, label, n, ";".join(examples[label])))

Path("reports/prepare_failures/failure_summary.tsv").write_text(
    "split\tlabel\tcount\texamples\n"
    + "\n".join(f"{split}\t{label}\t{n}\t{examples}" for split, label, n, examples in summary)
    + "\n"
)

Path("reports/prepare_failures/failure_details.tsv").write_text(
    "split\tentry\tlabel\treason\n"
    + "\n".join(f"{split}\t{entry}\t{label}\t{reason}" for split, entry, label, reason in details)
    + "\n"
)

print("GLOBAL")
for label, n in global_counts.most_common():
    print(f"{label}\t{n}")

print("wrote reports/prepare_failures/failure_summary.tsv")
print("wrote reports/prepare_failures/failure_details.tsv")
PY
```

看汇总：

```bash
column -t -s $'\t' reports/prepare_failures/failure_summary.tsv | less -S
```

看全局失败类别：

```bash
cut -f3 reports/prepare_failures/failure_details.tsv | tail -n +2 | sort | uniq -c | sort -nr
```

2026-05-29 当前全局分类结果：

| failure label | count | 解读 |
|---|---:|---|
| `dna_shorter_than_pwm` | 121 | 最大瓶颈；需要 DeepPBS-style partial PWM/mask 才能救 |
| `missing_dna_base_atoms` | 37 | 第二大瓶颈；可以通过跳过缺原子的 alignment candidate 来救一部分 |
| `pdb_download_fail` | 6 | 工程问题；可通过重试、缓存或手动下载修复 |

结论：

```text
当前主要损失来自 DNA 可见片段短于 PWM，而不是模型或 ESM。
如果要显著提高 prepare 成功率，最终绕不开 partial PWM supervision。
```

## 重点看 id benchmark

我们最终要和 DeepPBS benchmark 比，所以先单独看：

```bash
column -t -s $'\t' reports/prepare_failures/failure_details.tsv | \
  awk '$1=="deeppbs_id_contactalign" {print}' | less -S
```

也可以看每类例子：

```bash
for label in missing_dna_base_atoms dna_shorter_than_pwm no_contact_alignment other; do
  echo "===== $label ====="
  awk -F'\t' -v label="$label" '$1=="deeppbs_id_contactalign" && $3==label {print $2}' \
    reports/prepare_failures/failure_details.tsv | head -n 20
done
```

2026-05-29 `deeppbs_id_contactalign` 失败明细：

| label | count | entries |
|---|---:|---|
| `pdb_download_fail` | 2 | `2evi_A_MA0343.1.jaspar.npz`, `2evf_A_MA0343.1.jaspar.npz` |
| `missing_dna_base_atoms` | 3 | `5vpf_B_JUND_HUMAN.H11MO.0.A.npz`, `6dfy_C_MA0468.1.jaspar.npz`, `6dfy_D_DUX4_HUMAN.H11MO.0.A.npz` |
| `dna_shorter_than_pwm` | 13 | `7dcj_A_HSF1_HUMAN.H11MO.0.A.npz`, `5hdn_A_HSF1_HUMAN.H11MO.0.A.npz`, `2ady_A_MA0106.3.jaspar.npz`, `5yef_B_MA1930.1.jaspar.npz`, `5und_A_MA1930.1.jaspar.npz`, `5und_B_MA1930.1.jaspar.npz`, `5yef_G_MA1930.1.jaspar.npz`, `6a8r_A_DUX4_HUMAN.H11MO.0.A.npz`, `5kl4_A_WT1_HUMAN.H11MO.0.C.npz`, `4r2p_A_WT1_HUMAN.H11MO.0.C.npz`, `5kl4_D_WT1_HUMAN.H11MO.0.C.npz`, `5kl3_A_WT1_HUMAN.H11MO.0.C.npz`, `6b0q_A_WT1_HUMAN.H11MO.0.C.npz` |

结论：`id` benchmark 的 18 个失败里，13 个是 `dna_shorter_than_pwm`，说明要提高 benchmark 覆盖率，主要要解决 partial PWM/masked supervision。

建议核对命令：

```bash
awk -F'\t' '$1=="deeppbs_id_contactalign" {print $3}' reports/prepare_failures/failure_details.tsv | \
  sort | uniq -c | sort -nr

awk -F'\t' '$1=="deeppbs_id_contactalign" && $3=="dna_shorter_than_pwm" {print $2}' \
  reports/prepare_failures/failure_details.tsv | wc -l
```

## 可救优先级

| 优先级 | 类别 | 方案 | 原因 |
|---|---|---|---|
| P0 | `missing_dna_base_atoms` / `missing_dna_backbone_atoms` | 对齐候选 window 缺关键原子就跳过候选，不跳过整个样本 | 不改变监督定义 |
| P1 | `pdb_download_fail` / `pwm_missing` / `no_protein_or_dna` | 工程修复 | 标签定义不变 |
| P2 | `no_contact_alignment` | top-k sequence candidates 中选 contact 最大者，并记录 confidence | 有改变 alignment 策略风险 |
| P3 | `dna_shorter_than_pwm` | 引入 DeepPBS-style `pwm_mask/dna_mask` partial supervision | 改变数据 schema 和 loss |

## 为什么 `dna_shorter_than_pwm` 暂缓

DeepPBS 的做法是 partial alignment：

```text
完整 PWM 长度 M
PDB 可见 DNA 长度 K
只在重叠区域算 loss/metric
```

也就是：

```text
PWM[pwm_mask] vs prediction[dna_mask]
```

如果 RBE 要支持它，需要新增：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `pwm_mask` | `M` | 哪些 PWM columns 有结构监督 |
| `slot_to_dna_index` | `M` | 无 DNA 对应的 slot 记为 `-1` |
| `A_mask` | `N,M` | 哪些 residue-slot pair 可以算 contact loss |

这不是小修，会影响：

```text
dataset -> model output -> loss -> evaluation
```

所以先统计，不立刻救。

## 最终要形成的表

统计结束后，在报告里放这张表：

| split | success | failure | missing atoms | DNA shorter | no contact alignment | engineering | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke | 20 | 3 | 1 | 2 | 0 | 0 | 0 |
| train0 | 403 | 16 | 7 | 9 | 0 | 0 | 0 |
| train1 | 387 | 32 | 8 | 23 | 0 | 1 | 0 |
| train2 | 394 | 25 | 7 | 18 | 0 | 0 | 0 |
| train3 | 392 | 27 | 3 | 21 | 0 | 3 | 0 |
| train4 | 391 | 28 | 7 | 21 | 0 | 0 | 0 |
| valid0 | 89 | 15 | 1 | 14 | 0 | 0 | 0 |
| id benchmark | 112 | 18 | 3 | 13 | 0 | 2 | 0 |
| global | - | 164 | 37 | 121 | 0 | 6 | 0 |

## 当前原则

| 原则 | 说明 |
|---|---|
| 先统计，后救样本 | 避免为指标污染标签 |
| 只救监督仍成立的样本 | 不把 unknown 当 negative |
| 每类失败单独处理 | 不写混乱的大 if-else |
| 所有 rescue 都要单独报告 | benchmark subset 变化必须透明 |
