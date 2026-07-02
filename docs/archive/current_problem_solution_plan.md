# 当前问题与解决方案

## 一句话结论

当前 RBE 的主要问题不是模型完全不行，而是训练数据监督还不够完整。

最核心的洞是：

```text
DeepPBS 可以处理 DNA 片段短于 PWM 的样本；
RBE 现在不能。
```

所以大量样本在 prepare 阶段失败。

## 现在到底卡在哪里

我们统计了所有 contactalign prepare 失败样本：

| failure label | count | 含义 |
|---|---:|---|
| `dna_shorter_than_pwm` | 121 | PDB 里可见 DNA 比 PWM 短 |
| `missing_dna_base_atoms` | 37 | DNA residue 缺 base heavy atoms |
| `pdb_download_fail` | 6 | RCSB 下载失败 |

`id` benchmark 也一样：

| failure label | count |
|---|---:|
| `dna_shorter_than_pwm` | 13 |
| `missing_dna_base_atoms` | 3 |
| `pdb_download_fail` | 2 |

这说明：

```text
最大问题 = dna_shorter_than_pwm
```

不是 ESM，不是 EGNN，不是训练 loop。

## 为什么这个问题会发生

RBE 现在的数据定义是：

```text
PWM 第 j 列
    必须对应 complex 里一个真实 DNA base
    才能生成 A(i,j), E(i,j,b), site label
```

也就是默认要求：

```text
PDB DNA length >= PWM length
```

但真实结构经常不是这样。

很多 PDB 只解析了结合核心附近的 DNA，而数据库 PWM 可能更长：

```text
PWM length = 21
PDB visible DNA = 12
```

在这种情况下，当前 RBE 直接失败。

## DeepPBS 是怎么解决的

DeepPBS 不要求完整 PWM 都落到 DNA 上。

它做 partial alignment：

```text
完整 PWM 长度 M
PDB 可见 DNA 长度 K
只找二者最匹配的 overlap 区域
```

然后用 mask 控制训练和评估：

```text
PWM[pwm_mask] vs prediction[dna_mask]
```

所以 DeepPBS 可以处理：

```text
DNA length < PWM length
```

而 RBE 目前不能。

## contact-aware alignment 验证给了什么信息

你在 DeepPBS 里验证了 contact-aware alignment。

结果是：

| 实验 | baseline | contact-aware | 结论 |
|---|---:|---:|---|
| internal 5-fold MAE | 0.6887 | 0.6693 | 小幅变好 |
| independent benchmark mean MAE | 0.5896 | 0.5870 | 极小幅变好 |
| independent benchmark median MAE | 0.5498 | 0.5652 | 反而略差 |
| better count | - | 61/130 | 不到一半样本变好 |

这个验证说明两件事：

| 结论 | 对 RBE 的意义 |
|---|---|
| contact-aware bound motif 定义是合理的 | RBE 继续使用 contact-constrained alignment 没问题 |
| hard contact filter 不是大杀器 | 不能指望只靠 contact-aware alignment 大幅超过 DeepPBS |

所以 contact-aware alignment 是：

```text
数据监督清洗策略
```

不是：

```text
最终主要创新点
```

RBE 的主要创新仍然应该是：

```text
monomer protein -> latent residue-slot-base A/E -> PWM + binding site
```

## 解决原则

| 原则 | 解释 |
|---|---|
| 不把 unknown 当 negative | 没有 DNA 原子或没有对应 base，不等于没有接触 |
| 不做补丁式兜底 | 不能失败了就回退到 PWM-only，否则 label 会混 |
| 先解决最大失败源 | `dna_shorter_than_pwm` 占 121/164 |
| 保持和 DeepPBS 可比 | PWM MAE、mask 逻辑、benchmark 口径要对齐 |
| 每一步单独验收 | 避免 schema/loss/eval 一起乱改 |

## 总体解决路线

```text
Step 0: 统一 all sample pool
Step 1: 引入 DeepPBS-style partial PWM mask
Step 2: 让 RBE loss/eval 支持 mask
Step 3: 再处理 missing DNA atoms
Step 4: 重建 train/valid/id manifests
Step 5: 重新训练 5-fold ensemble
Step 6: 和 DeepPBS 做 same-subset comparison
```

下面分开说。

## Step 0. 统一 all sample pool

目的：

```text
同一个 sample 只 prepare 一次
不同 train/valid/id 只通过 manifest 或 symlink 引用
```

原因：

| 现在的问题 | 后果 |
|---|---|
| 每个 fold 单独 prepare | 重复下载 PDB、重复算 ESM、重复 alignment |
| 同一 entry 在不同 fold 里重复失败 | 失败统计膨胀 |
| 网络下载随机失败 | fold 数据不稳定 |

正确生成 all entries：

```bash
awk 'NF {print}' \
  resources/deeppbs_curated/folds/train{0..4}.txt \
  resources/deeppbs_curated/folds/valid{0..4}.txt \
  resources/deeppbs_curated/folds/id.txt \
  | sort -u \
  > data/deeppbs_all_contactalign/all_entries.txt
```

注意：

```text
不要用 cat 拼接 fold 文件。
```

因为有些 fold 文件末尾没有换行，`cat file1 file2` 会把相邻文件的两条 entry 粘成假 entry。

当前正确数量：

| 集合 | unique |
|---|---:|
| cross-validation `train0-4 + valid0-4` | 523 |
| independent benchmark `id.txt` | 130 |
| overlap | 0 |
| all pool | 653 |

## Step 1. 引入 partial PWM schema

目标：

让 RBE 支持：

```text
PWM 长度 M
但只有其中 K 个 slot 有 complex DNA supervision
```

新增字段：

| 字段 | shape | 定义 |
|---|---:|---|
| `pwm_mask` | `[M]` | 哪些 PWM columns 有可见 DNA 对应 |
| `slot_to_dna_index` | `[M]` | 有对应 DNA 的 slot 写 DNA index；没有则写 `-1` |
| `A_mask` | `[N,M]` | 哪些 residue-slot pair 可以计算 contact loss |
| `site_label` | `[N]` | 从有监督的 visible slots 聚合得到 |

原来：

```text
slot_to_dna_index[j] 必须是 DNA index
```

改成：

```text
slot_to_dna_index[j] = -1 表示这个 PWM column 没有 PDB DNA 对应
```

这样：

```text
PWM 可以保持完整 M 列
A/E 也保持完整 N x M
但只有 pwm_mask=True 的位置参与结构监督
```

## Step 2. 修改 loss 和 eval

### PWM loss

现在：

```text
L_pwm = KL(PWM_target[M,4], PWM_pred[M,4])
```

masked 后：

```text
L_pwm = KL(PWM_target[pwm_mask], PWM_pred[pwm_mask])
```

如果 `pwm_mask` 全 True，行为和现在一致。

### A loss

现在：

```text
L_A = BCE(A_pred[N,M], A_label[N,M])
```

masked 后：

```text
L_A = BCE(A_pred[A_mask], A_label[A_mask])
```

也就是没有 DNA 对应的 slot 不参与 contact loss。

### site loss

site label 只从可见 DNA slots 得到：

```text
site_label[i] = max_j A_contact_label[i,j]
where pwm_mask[j] = True
```

site loss 仍然是：

```text
BCE(site_prob, site_label)
```

但要记住：

```text
site_label 只表示 residue 是否接触可见 bound motif region
```

### evaluation

PWM benchmark 也要支持 mask：

```text
MAE/KL/IC-PCC 只在 pwm_mask=True 的 columns 上算
```

这样才和 DeepPBS 的 `pwm_mask/dna_mask` 逻辑一致。

## Step 3. missing DNA atoms 怎么救

`missing_dna_base_atoms` 的数量也不少：

```text
37 global
3 id benchmark
```

解决方式：

```text
alignment candidate 如果包含缺 base/backbone atoms 的 DNA residue
    跳过这个 candidate
如果还有其他 candidate
    继续选 contact-aware best candidate
如果所有 candidate 都不合法
    样本失败
```

为什么这么做：

| 做法 | 是否正确 |
|---|---|
| 缺原子当作 no contact | 不正确 |
| 整个样本直接失败 | 太保守 |
| 只跳过坏 candidate | 合理 |

这里必须写干净：

```text
结构层抛明确错误类型
alignment 层只捕获这个类型
```

不要用字符串匹配 traceback。

## Step 4. download fail 怎么救

`pdb_download_fail` 是工程问题：

```text
6 global
2 id benchmark
```

解决方式：

| 方法 | 说明 |
|---|---|
| 重跑 prepare | 很多 SSL EOF 是瞬时问题 |
| 下载加 retry | 最多重试 3 次 |
| 优先使用已存在 cache | 避免重复网络请求 |

这个不影响模型定义，优先级低于 partial PWM，但实现很简单。

## Step 5. 重建 split manifests / symlinks

当 all pool prepare 完成后：

```text
data/deeppbs_all_contactalign/train/*.npz
```

每个 split 不再重新 prepare。

只做：

```text
读取 train0.txt / valid0.txt / id.txt
如果 all pool 里有对应 .npz
    写入 split manifest
    或建立 symlink
否则
    记录 split missing
```

推荐结构：

```text
data/deeppbs_all_contactalign/
  train/
    sample.npz
  train_manifest.txt
  failed.tsv
  sample_table.tsv

data/deeppbs_train0_contactalign/
  train/
    sample.npz -> ../deeppbs_all_contactalign/train/sample.npz
  train_manifest.txt
```

这样目录仍然直观，但不重复 prepare。

## Step 6. 重新训练和 benchmark

重新训练：

```text
fold0 model
fold1 model
fold2 model
fold3 model
fold4 model
```

benchmark：

```text
id.txt contact-valid / mask-valid subset
5-fold ensemble
DeepPBS-style PWM MAE
site AP/MCC/F1
A_contact AP/top-L precision
```

最终要形成两张表。

### 表 1. 与 DeepPBS 量级比较

| 方法 | benchmark | 输入 | PWM MAE |
|---|---|---|---:|
| DeepPBS official | full id 130 | protein-DNA complex | 0.4796 median |
| RBE | id mask-valid subset | monomer + motif length | 待跑 |

### 表 2. same-subset 严格比较

| 方法 | subset | 输入 | MAE | KL | IC-PCC |
|---|---|---|---:|---:|---:|
| DeepPBS rerun | same subset | protein-DNA complex | 待跑 | 待跑 | 待跑 |
| RBE | same subset | monomer + motif length | 待跑 | 待跑 | 待跑 |

第二张表才是真正公平比较。

## 为什么先做 partial PWM，而不是继续调模型

因为失败统计已经说明：

```text
121/164 failures = dna_shorter_than_pwm
```

如果不支持 partial PWM：

| 后果 | 影响 |
|---|---|
| 大量样本进不了训练 | 数据少 |
| id benchmark 从 130 变 112 | 和 DeepPBS 不完全可比 |
| 长 PWM TF family 被系统性丢掉 | 数据分布偏 |
| RBE 的 slot 学习更困难 | 长 motif 信息缺失 |

所以先补数据监督，比盲目加模型层更合理。

## 为什么不是直接学 DeepPBS 全部逻辑

我们只借 DeepPBS 的：

```text
masked partial supervision
```

不借它的核心输入设定：

```text
protein-DNA complex geometry
```

RBE 仍然保持：

```text
推理只输入 protein monomer + motif length
```

这是和 DeepPBS 的根本区别。

## 最短落地顺序

| 顺序 | 改什么 | 验收 |
|---|---|---|
| 1 | all pool 去重 prepare 流程 | all entries = 653 |
| 2 | `pwm_mask/slot_to_dna_index=-1/A_mask` schema | toy/sample npz shape 正确 |
| 3 | masked loss/eval | mask 全 True 时结果等于旧逻辑 |
| 4 | partial PWM prepare | `dna_shorter_than_pwm` failure 明显下降 |
| 5 | missing atom candidate skip | `missing_dna_base_atoms` failure 下降 |
| 6 | 5-fold RBE retrain | train/valid 不崩 |
| 7 | id benchmark ensemble | 输出新 `eval_summary.tsv` |
| 8 | DeepPBS same-subset rerun | 形成公平表 |

## 当前不做

| 不做 | 原因 |
|---|---|
| 不把 missing DNA 当作 no contact | 标签会错 |
| 不对 failed sample 做 PWM-only 回退 | 会混入非 bound motif |
| 不直接加复杂模型 | 当前主要瓶颈是数据监督 |
| 不声称超过 DeepPBS | same-subset baseline 还没完成 |

## 最小成功标准

| 标准 | 目标 |
|---|---|
| all pool 数量 | 653 entries |
| `dna_shorter_than_pwm` failure | 明显下降 |
| id benchmark coverage | 从 112 提高，目标接近 130 |
| masked eval | 能和 DeepPBS mask 口径一致 |
| PWM MAE | 仍接近 DeepPBS 量级 |
| site/AP/A map | 不因为 partial PWM 明显崩 |
