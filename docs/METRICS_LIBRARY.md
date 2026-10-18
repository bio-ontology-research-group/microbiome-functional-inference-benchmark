# Metrics and plotting functions

All calculations operate on processed sample-by-KO tables. The package does
not contain model training, raw-data processing, or study-selection code.

## Metric families

| Family | Main functions | Implemented behavior |
| --- | --- | --- |
| Profile agreement | `samplewise_spearman`, `samplewise_spearman_on_universe`, `samplewise_profile_metrics`, `per_ko_spearman`, `per_sample_richness_overlap` | Shared or declared WGS-positive KO surface, prediction zero-fill where requested, per-sample L1 normalization, and sample-wise profile agreement. |
| Matchado-style recovery | `matchado_differential_abundance_metrics`, `classification_set_metrics` | Shared 5% prevalence surface, two-sided per-KO rank-sum tests, WGS-significant KOs as the reference set, precision/recall/F1, ranking metrics, and prediction-label randomizations. |
| Shared-WGS BH recovery | `matchado_bh_shared_wgs_reference_metrics`, `benjamini_hochberg`, `r_default_wilcoxon_by_ko` | One WGS-defined KO family, BH-corrected tests, prediction zero-fill, and significant-KO recovery. |
| Sun-style group inference | `sun_signed_wilcoxon_by_ko`, `sun_inference_correlation_abundance_randomization`, `permute_ko_abundances_across_samples` | Signed `log10(p)` group statistic, WGS-versus-prediction Spearman correlation, and independent WGS KO-abundance shuffles with labels fixed. |
| Effect relationship | `rank_biserial_effect_by_ko`, `biological_group_effect_concordance` | Signed rank-biserial effects and WGS-versus-predicted Pearson and Spearman relationships. |
| Directional ROC/PR | `directional_ranking_curves` | Separate positive and negative WGS targets, exact sklearn ROC and PR points, average precision, and ROC-AUC. |
| Decile outcome composition | `decile_outcome_composition` | Ascending predicted-effect ranking, lexical KO tie-breaks, equal-count bins, and positive/negative/nonsignificant WGS outcome shares. |

## Plot functions

| Plot | Function |
| --- | --- |
| PICRUSt2-style study profile panels | `plot_profile_study_panels` |
| Study-stratum mean ranges | `plot_stratum_mean_ranges` |
| Differences from PICRUSt2 | `plot_method_differences` |
| KO coverage | `plot_ko_coverage` |
| Matchado-style metric rows | `plot_metric_row_grid` |
| Matchado-style randomization dumbbells | `plot_matchado_randomization_dumbbells` |
| Sun-style observed and shuffled controls | `plot_sun_observed_controls` |
| Directional AP or ROC-AUC summaries | `plot_directional_metric_summary` |
| Full directional ROC grid | `plot_directional_roc_grid` |
| Signed outcome composition by decile | `plot_signed_decile_grid` |

Each plotting function returns its Matplotlib figure and axes. Notebook 1 shows
the paper-style profile, Matchado, and Sun figures; Notebook 2 shows the effect,
ROC/PR, and decile figures.

## Included evidence and validation

The 26-stratum profile overview distinguishes 11 benchmark strata from 15
expansion strata. The distributed study matrices cover the ten accepted
contrasts with biological labels.

| Notebook | Local inputs | Validation outcome |
| --- | --- | --- |
| `01_established_profile_and_group_recovery.ipynb` | `data/studies/*.npz` plus profile, coverage, registry, Tax4Fun2, and legacy Matchado tables under `data/reference/` | PICRUSt2 and adaptation retain 1,639 profile rows each; Tax4Fun2 retains 1,637. Completed comparable profile outputs agree within `1e-12`. The three-method Matchado and Sun calculations are rerun over all ten bundled contrasts. The earlier Matchado result has a different KO surface and is labeled as such. |
| `02_exploratory_effect_ranking_and_deciles.ipynb` | The accepted ten-study per-KO and completed comparison tables under `data/reference/` | Effect relationships, directional AP/ROC-AUC, and decile composition agree with completed results within `1e-12`. The notebook displays full ROC trajectories and signed decile panels with whole-study reference lines. |

`data/manifest.tsv` gives the SHA-256 digest and byte size of every distributed
input. The notebooks read only these repository files and write nothing outside
the repository.

## Reference basis

The implementations follow the PICRUSt2 profile comparisons, Matchado et al.
differential-KO recovery metrics, and Sun et al. signed group-inference
correlation and abundance-shuffle controls. Exact source notebook and artifact
identifiers are preserved in `data/source_artifact_manifest.tsv` and the
notebook provenance sections.

Unavailable reference surfaces remain labeled unavailable; the notebooks do
not manufacture comparisons for results that were never completed.
