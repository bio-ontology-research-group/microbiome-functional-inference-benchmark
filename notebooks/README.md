# Reproducibility notebooks

The notebooks are the primary usage guide for the repository. They import the
metric and plotting functions from `picrust_benchmark_metrics` and use only the
processed files committed under `data/`.

| Notebook | What it demonstrates |
| --- | --- |
| `01_established_profile_and_group_recovery.ipynb` | PICRUSt2-style profile comparison and KO coverage; Matchado-style recovery and randomization dumbbells; Sun-style observed inference correlations and WGS-abundance controls. PICRUSt2, basic-MLP adaptation, and Tax4Fun2 are compared. |
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
