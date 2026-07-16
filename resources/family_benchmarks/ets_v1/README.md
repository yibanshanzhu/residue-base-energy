# ETS Family Benchmark v1

This benchmark asks whether RBE predicts specificity differences within the ETS
DNA-binding-domain family, rather than identifying unrelated TF families.

## Contract

- Every protein is grouped by UniProt accession.
- All structures from one accession stay in the same held-out fold.
- Every included PWM is oriented to the `GGAA`/`GGAT` family reference.
- Every target is cropped to a fixed 8-bp window with the ETS core at zero-based
  slots `3:7`.
- Database duplicates, mismatched targets, and tandem FLI1 motifs are excluded.
- Results are first averaged within UniProt groups, then across groups.

`samples.tsv` is the auditable membership and alignment table derived from the
canonical DeepPBS cache and mmCIF chain metadata.
