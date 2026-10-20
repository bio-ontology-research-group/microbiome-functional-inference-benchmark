# Combined benchmark and environment summary

This folder consolidates the repository's original benchmark sources and the
completed expansion benchmark into one analysis layer. It compares **PICRUSt2**
and **Tax4Fun2** only; no model is fitted. The original notebooks, reference
tables, and `expansion_benchmark/` are read-only inputs to this layer.

The goal is to answer two different questions without confusing them:

1. **Profile agreement:** how closely does each inferred sample profile agree
   with its paired WGS profile, and how does that agreement vary by biological
   environment?
2. **Biological-group recovery:** where labels support a two-group contrast,
   how well do inferred profiles recover WGS differential functions, signed
   group effects, and directional rankings?

## Census and analysis units

- **1,717 paired samples:** 621 benchmark-source samples plus 1,096 completed
  expansion samples.
- **24 unique source projects.** An accession contributes once to this count,
  even when it contains several environments or technical analysis strata.
- **29 source-project × environment scopes.** This is the primary unit for
  profile summaries. HMP contributes two environments; the five EMP500
  environments remain separate; Brooks forward/reverse reads belong to one
  biological environment scope.
- **17 accepted biological contrasts:** eight benchmark-source contrasts and
  nine expansion contrasts. Paper-style metrics that require unpaired groups
  are not applied to the three paired contrasts.

The environment-level macro summary gives every source-project × environment
scope equal weight. It therefore does not allow a large study to dominate a
small study. Sample counts remain visible in all source tables. Blueberry is
retained as `Unknown (contradictory)` because that is the frozen source
annotation; this layer does not invent an environment assignment.

## Reproduce

From the repository root:

```bash
python combined_benchmark/build_combined_benchmark.py --overwrite
python combined_benchmark/validate_outputs.py
jupyter nbconvert --to notebook --execute --inplace \
  combined_benchmark/notebooks/01_combined_environment_summary.ipynb
```

`--overwrite` replaces only `combined_benchmark/results/` and
`combined_benchmark/figures/`. It does not write into the original result
folders. The builder reads the small frozen environment registry under
`inputs/`; the old research workspace is needed only if that derived registry
must be regenerated.

## Outputs

| File | Meaning |
|---|---|
| `results/combined_profile_sample_long.tsv.gz` | Both methods for every paired sample, including explicit undefined statuses. |
| `results/combined_sample_environment_registry.tsv.gz` | Sample-to-project and environment provenance. |
| `results/scope_profile_summary.tsv` | Means, medians, SDs, and deterministic sample-bootstrap intervals for 29 scopes × 2 methods. |
| `results/scope_method_difference.tsv` | Paired Tax4Fun2 − PICRUSt2 sample-correlation differences by scope. |
| `results/environment_profile_summary.tsv` | Equal-scope-weight environment macro summary. |
| `results/combined_contrast_registry.tsv` | The 17 biological contrasts and their environment/provenance. |
| `results/combined_matchado.tsv` | Matchado-style raw-*p* recovery and label-randomization results. Brooks reverse reads are retained as a technical stratum; the main plot uses the forward primary stratum. |
| `results/combined_sun.tsv` | Sun-style observed inference correlations and WGS-abundance controls. |
| `results/combined_bh.tsv` | Repository BH extension on a fixed WGS-defined KO family. |
| `results/combined_effect.tsv` | WGS-versus-prediction rank-biserial effect relationships. |
| `results/combined_directional.tsv` | Positive/negative average precision and ROC AUC. |
| `results/input_manifest.tsv` | SHA-256 provenance for every read-only input. |

The eight numbered PNGs are the compact visual summary. The notebook is the
human-readable walkthrough of the same tables and figures.

## Attribution and repository modifications

Sample-wise profile agreement follows benchmarking precedent from
[PICRUSt](https://doi.org/10.1038/nbt.2676) and
[PICRUSt2](https://doi.org/10.1038/s41587-020-0548-6). Tax4Fun2 predictions are
from [Wemheuer et al. (2020)](https://doi.org/10.1186/s40793-020-00358-7).

The raw-*p* differential-KO precision, recall, F1, 20-round phenotype-label
randomization, and dumbbell presentation are credited to
[Matchado et al. (2024)](https://doi.org/10.1099/mgen.0.001203). They use the
paper's **unadjusted *p* < 0.05** definition. The repository uses one shared
KO/prevalence surface for direct comparison; this operational choice is not
claimed as part of the paper.

The signed group-inference correlation and 100 WGS KO-abundance shuffles are
credited to [Sun et al. (2020)](https://doi.org/10.1186/s40168-020-00815-y).
Sun et al. evaluated the original Tax4Fun; applying their metric definition to
Tax4Fun2 here is a documented repository application, not a claim that the
paper evaluated Tax4Fun2.

The following are **repository extensions**, not metrics attributed to those
papers:

- BH adjustment within each complete KO family, with one WGS-defined family,
  prediction zero-fill, and *q* < 0.05 selection. BH is used to limit the
  expected false-discovery proportion across the many simultaneous KO tests.
- WGS-versus-predicted rank-biserial effect relationships.
- Directional ROC/precision-recall curves, ROC AUC, average precision, and
  prediction-ranked outcome composition.
- Equal-scope environment aggregation and paired Tax4Fun2 − PICRUSt2
  differences.

See [`docs/REFERENCES.md`](../docs/REFERENCES.md) and
[`docs/METRICS_LIBRARY.md`](../docs/METRICS_LIBRARY.md) for the complete metric
definitions and citation boundaries.

## Explicit unavailable values

Four Tax4Fun2 sample-wise correlations are undefined because the native
prediction has zero total abundance on the WGS evaluation universe; they are
retained as rows with status `zero_method_total_on_wgs_universe`. Tax4Fun2
group-level metrics for olive season 1 are likewise unavailable because two
samples have zero native totals. The tables do not impute or silently drop
these cases.

## Scientific choices still open

- Whether the final manuscript should use the frozen EMPO-style hierarchy or
  a separately reviewed environment ontology.
- Whether environments should be summarized with equal scope weights (current
  default), sample weights, or a hierarchical model. These answer different
  questions and should not be mixed.
- Whether provisional/control expansion contrasts belong in the primary
  biological figure or a supplement. Their status is retained in the contrast
  registry so this can be changed without recomputing metrics.
- Whether Brooks reverse-read technical results should remain supplemental;
  the table retains them, while the primary Matchado plot uses forward reads.
