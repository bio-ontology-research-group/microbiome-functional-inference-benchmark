# Reproducibility notebooks

The notebooks are the primary usage guide for the repository. They import the
metric and plotting functions from `picrust_benchmark_metrics` and use only the
processed files committed under `data/`.

| Notebook | What it demonstrates |
| --- | --- |
| `01_established_profile_and_group_recovery.ipynb` | PICRUSt2 paper-derived profile comparison and KO coverage; Matchado et al.-derived recovery and randomization dumbbells; Sun et al.-derived observed inference correlations and WGS-abundance controls. PICRUSt2, basic-MLP adaptation, and Tax4Fun2 are compared. |
| `02_exploratory_effect_ranking_and_deciles.ipynb` | WGS-versus-predicted effects, directional precision-recall and ROC curves, AP/ROC-AUC summaries, and signed outcome composition by predicted-effect decile. |

The full profile overview contains 11 benchmark strata plus 15 expansion
strata. The matrix-based biological-group analyses use the ten accepted labeled
contrasts included under `data/studies/`.

From the repository root:

```bash
python -m pip install -e ".[notebooks]"
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  notebooks/01_established_profile_and_group_recovery.ipynb

python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  notebooks/02_exploratory_effect_ranking_and_deciles.ipynb
```

Both notebooks display the input hashes, numerical comparison tolerances, and
any reference result that is unavailable on the exact evaluated surface.

## Attribution

Notebook 1 reimplements published metric families and credits the corresponding
papers in its opening section and beside the relevant analyses. Notebook 2 is
explicitly labeled as project-developed extensions. A complete bibliography
and metric-ownership table is maintained in `docs/REFERENCES.md`.

In particular, Sun et al. (2020) evaluated the original Tax4Fun, not Tax4Fun2;
the notebook applies their metric definition to the current Tax4Fun2
predictions. Matchado et al. (2024) did include Tax4Fun2. Figure layouts marked
“inspired by” use the cited paper's visual grammar on this repository's data
and are not presented as copies of the published figures.

Notebook 1's Matchado-derived replay uses the paper's unadjusted *p* < 0.05
rule. Notebook 2's recovery, effect, ROC/PR, and decile targets instead use the
project's BH-adjusted WGS reference. BH was added to address simultaneous
testing across thousands of KOs and the family was fixed from WGS so every
method is judged against the same target. The notebook labels these outputs as
a project extension; they are not reported as the original Matchado metric.
