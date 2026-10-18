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

## Attribution

This repository brings published benchmark metrics into a reusable Python
implementation; it does not claim those metrics as original contributions.
Profile-correlation and differential-KO precedents are credited to the
[PICRUSt](https://doi.org/10.1038/nbt.2676) and
[PICRUSt2](https://doi.org/10.1038/s41587-020-0548-6) papers. The raw-*p*
differential-KO precision/recall/F1 analysis and 20-round prediction-label
randomization are credited to
[Matchado et al. (2024)](https://doi.org/10.1099/mgen.0.001203). The signed
group-inference correlation and 100-round WGS-abundance shuffle are credited to
[Sun et al. (2020)](https://doi.org/10.1186/s40168-020-00815-y).

The effect-size, directional ROC/precision-recall, and prediction-ranked decile
analyses are extensions developed in this project. The complete ownership map,
method citations, and distinctions between paper definitions and repository
implementation choices are in [docs/REFERENCES.md](docs/REFERENCES.md).

### Important: raw *p* and BH results are different analyses

The repository deliberately provides both versions, and their scores should
not be treated as the same metric:

- **Published Matchado replay:** significant KOs use unadjusted *p* < 0.05,
  following Matchado et al. (2024). No multiple-testing correction is applied.
- **Project BH extension:** significant KOs use Benjamini-Hochberg-adjusted
  *q* < 0.05 within each study's fixed KO family. We introduced this version to
  limit the expected false-discovery proportion when thousands of KOs are
  tested simultaneously and to give every prediction method the same
  WGS-defined reference family.

The BH extension also zero-fills prediction KOs absent from that WGS family and
does not apply a prediction-side prevalence filter. These are project choices,
not parts of the original Matchado metric. Consequently, raw-*p* and BH
precision, recall, and F1 values are labeled separately and are not directly
comparable.

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
  KO coverage, Matchado et al.-derived differential-KO recovery and label
  randomizations, and Sun et al.-derived observed inference correlation with WGS
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
are documented in `docs/INPUT_FORMATS.md`, `docs/METRICS_LIBRARY.md`, and
`docs/REFERENCES.md`.
