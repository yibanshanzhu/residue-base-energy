# DeepPBS Curated Mapping Resources

This directory vendors the small DeepPBS curated resources needed by RBE data prep.

| Path | Content |
|---|---|
| `folds/*.txt` | DeepPBS curated `pdb_chain_pwmid.npz` fold entries |
| `pwms/*.txt` | A/C/G/T PWM matrices exported from DeepPBS `pwms.pickle` |
| `summary.tsv` | Resource counts |

PWM files are trimmed with the DeepPBS rule: remove low-information columns from both ends until information content is `> 0.5`.

Runtime data prep does not import or execute the DeepPBS repository.
