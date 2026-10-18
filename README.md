# Functional-inference benchmark metrics

This repository consolidates metrics used to benchmark methods that predict
microbial community functional profiles from amplicon profiles. Its purpose is
to bring established evaluation approaches into one reproducible place,
document what each metric compares, and demonstrate their use against paired
WGS functional profiles. It contains tested metric implementations,
paper-inspired plotting functions, compact processed benchmark inputs, and
executed multi-study notebooks. The current benchmark examples compare
PICRUSt2, PICRUSt2 plus a basic-MLP adaptation, and Tax4Fun2.

The first notebook brings together established profile-level and
biological-group benchmarking approaches from the reference literature. The
second notebook develops additional diagnostics from this project to examine
the same predictions in greater depth: the relationship between observed and
predicted biological effects, directional ROC and precision-recall behavior,
and the composition of observed outcomes across prediction-ranked deciles.
These views help reveal what a summary metric is actually rewarding, where a
method succeeds or fails, and whether its apparent performance supports the
biological question being asked. The goal is to help users interpret the
metrics and choose an evaluation strategy deliberately, rather than select a
score without understanding the behavior behind it.

## Install and run

```bash
git clone <repository-url>
cd <repository-directory>
python -m pip install -e ".[notebooks]"
jupyter lab notebooks/
```

To run the checks:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Included notebooks

- `01_established_profile_and_group_recovery.ipynb` covers profile agreement,
  KO coverage, Matchado-style differential-KO recovery and label
  randomizations, and Sun-style observed inference correlation with WGS
  abundance controls. It compares PICRUSt2, the basic-MLP adaptation, and
  Tax4Fun2.
- `02_exploratory_effect_ranking_and_deciles.ipynb` covers WGS-versus-predicted
  KO effects, directional precision-recall and ROC curves, AUC summaries, and
  signed outcome composition across predicted-effect deciles.

The full overview contains **11 benchmark strata plus 15 expansion strata**,
not 15 studies total. Biological-group analyses use the ten contrasts with
accepted labels.

## Included processed data

`data/studies/` contains the ten labeled contrasts used by the group-level
notebook analyses. Each compressed file contains only:

- a processed sample-by-KO WGS truth matrix;
- processed PICRUSt2, basic-MLP adaptation, and Tax4Fun2 predictions;
- sample identifiers, KO identifiers, and the audited two-group labels.

`data/reference/` contains compact completed result tables used for regression
checks and the complete 26-stratum profile overview. No reads, assemblies,
amplicon tables, model weights, or training data are included. See
`data/README.md` and `data/manifest.tsv` for the exact contents and hashes.

## Use individual functions

```python
from picrust_benchmark_metrics import (
    load_processed_study,
    matchado_differential_abundance_metrics,
    samplewise_spearman_on_universe,
)

wgs, predictions, labels, metadata = load_processed_study(
    "data/studies/cameroon.npz"
)

profile = samplewise_spearman_on_universe(
    wgs, predictions["PICRUSt2"], wgs.columns
)

recovery, per_ko = matchado_differential_abundance_metrics(
    wgs,
    predictions,
    labels,
    group_a=metadata["group_a"],
    group_b=metadata["group_b"],
    n_permutations=20,
)
```

Input requirements, metric definitions, plot functions, and validation results
are documented in `docs/INPUT_FORMATS.md` and `docs/METRICS_LIBRARY.md`.
