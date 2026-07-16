# ETS Family Mechanism Benchmark v1

## Protocol

| Item | Value |
|---|---:|
| Included complexes | 26 |
| UniProt groups | 12 |
| PWM window | fixed 8 bp |
| Core alignment | `GGAA/GGAT` at zero-based slots `3:7` |
| Structural visibility | 8/8 columns for every sample |
| Split | leave-one-UniProt-out |
| Model selection | one independent validation UniProt per fold |
| Aggregation | structures within UniProt, then 12 UniProt groups equally |

The model uses a group-balanced training-fold PWM prior and predicts residue-base
residual logits. Residual scale is selected from `0.0, 0.1, ..., 1.0` using only
the fold's validation UniProt. The 12 outer test groups are never used for model
or scale selection.

Server verification used branch `family-mechanism-benchmark` at commit
`86986cf`; all 24 training logs reached epoch 100 and 44 tests passed on node5.

## Results

| Method | PWM MAE | A-base AP | Interpretation |
|---|---:|---:|---|
| full RBE | 0.3202 | 0.6093 | calibrated residue-base model |
| PWM-only RBE | 0.3143 | 0.0289 | same architecture, no structural losses |
| nearest ESM | 0.3086 | - | training-fold nearest neighbor |
| family mean | **0.3062** | - | group-balanced training-fold prior |
| uniform gate | 0.4678 | - | residue localization removed |
| shuffled energy | 0.4799 | - | residue-energy pairing destroyed |

Relative to full RBE, uniform-gate and shuffled-energy ablations increase mean
MAE by `+0.1476` and `+0.1597`. Full RBE beats each ablation in 9 groups, loses
in 2, and ties in the fold whose validation-selected residual scale is zero.

Full RBE does **not** beat family mean, nearest ESM, or PWM-only on mean PWM MAE.
The model beats family mean in 6 groups, loses in 5, and ties in 1, but several
large errors make its group-equal mean worse by `0.0140`.

## Claim Boundary

This benchmark supports the following claim:

> RBE learns structurally meaningful residue-base contact maps, and its PWM
> output causally depends on the learned residue-energy pairing.

It does not support this stronger claim:

> RBE improves held-out ETS-family PWM accuracy over a training-family prior.

Therefore the demonstrated value is mechanistic localization and coupling, not
superior family-level specificity accuracy. The small 12-group benchmark and
high between-group variance make this an exploratory mechanism result.

## Artifacts

| File | Content |
|---|---|
| `summary.tsv` | group-equal metric summary |
| `paired_pwm_mae.tsv` | paired method-minus-full MAE deltas |
| `per_group.tsv` | one row per method and held-out UniProt |
| `full_residual_scales.tsv` | full-model validation scale curves |
| `pwm_only_residual_scales.tsv` | PWM-only validation scale curves |
