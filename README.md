# Functional-inference benchmark metrics

Reusable Python functions and complete notebooks for comparing WGS functional
profiles with PICRUSt2, PICRUSt2 plus a basic-MLP adaptation, and Tax4Fun2.

The repository is intended to be cloned and explored directly. The two
notebooks are the primary usage guides: they call the same metric and plotting
functions available under `src/picrust_benchmark_metrics/` and compare their
outputs with completed benchmark results.

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
