# Metrics and plotting functions

All calculations operate on processed sample-by-KO tables. The package does
not contain model training, raw-data processing, or study-selection code.

## Metric families

| Family | Main functions | Implemented behavior |
| --- | --- | --- |
| Profile agreement | `samplewise_spearman`, `samplewise_spearman_on_universe`, `samplewise_profile_metrics`, `per_ko_spearman`, `per_sample_richness_overlap` | Published precedent: Langille et al. (2013) and Douglas et al. (2020). The declared WGS-positive KO surface, prediction zero-fill, row normalization, and aggregation are repository choices. |
| Matchado published recovery family | `matchado_differential_abundance_metrics`, `classification_set_metrics` | Derived from Matchado et al. (2024): low-prevalence filtering, two-sided per-KO rank-sum tests, unadjusted *p* < 0.05, WGS-significant KOs as the reference set, precision/recall/F1, and 20 prediction-label randomizations. The library fixes one KO intersection and 5% prevalence surface across WGS and all supplied methods for direct comparison; this all-method surface is a repository choice. |
| Raw-*p* ranking extension | `matchado_differential_abundance_metrics` (`auroc`, `aupr` rows) | Project addition calculated on the Matchado raw-*p* surface; AUROC/AUPR are not attributed to Matchado et al. |
| Shared-WGS BH recovery | `matchado_bh_shared_wgs_reference_metrics`, `benjamini_hochberg`, `r_default_wilcoxon_by_ko` | Project implementation combining published differential-KO set-recovery precedent with one WGS-defined family, BH correction, and prediction zero-fill. It is not an exact Matchado or PICRUSt2 paper metric. |
| Sun published group-inference family | `sun_signed_wilcoxon_by_ko`, `sun_inference_correlation_abundance_randomization`, `permute_ko_abundances_across_samples` | Sun et al. (2020): signed `log10(p)` group statistic, WGS-versus-prediction Spearman correlation, and 100 independent WGS gene/KO-abundance shuffles with labels and predictions fixed. |
| Effect relationship | `rank_biserial_effect_by_ko`, `biological_group_effect_concordance` | Signed rank-biserial effects and WGS-versus-predicted Pearson and Spearman relationships. |
| Directional ROC/PR | `directional_ranking_curves` | Separate positive and negative WGS targets, exact sklearn ROC and PR points, average precision, and ROC-AUC. |
| Decile outcome composition | `decile_outcome_composition` | Ascending predicted-effect ranking, lexical KO tie-breaks, equal-count bins, and positive/negative/nonsignificant WGS outcome shares. |

## Plot functions

| Plot | Function |
| --- | --- |
| PICRUSt2 Fig. 2a-inspired study profile panels | `plot_profile_study_panels` |
| Study-stratum mean ranges | `plot_stratum_mean_ranges` |
| Differences from PICRUSt2 | `plot_method_differences` |
| KO coverage | `plot_ko_coverage` |
| Matchado Fig. 6-inspired metric rows | `plot_metric_row_grid` |
| Matchado Fig. 7-inspired randomization dumbbells | `plot_matchado_randomization_dumbbells` |
| Sun Fig. 4-inspired observed and shuffled controls | `plot_sun_observed_controls` |
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

## Reference basis and ownership

The implementations use published precedents from Langille et al. (2013) and
Douglas et al. (2020), the differential-KO recovery metrics of Matchado et al.
(2024), and the signed group-inference correlation and abundance-shuffle
controls of Sun et al. (2020). Exact source notebook and artifact identifiers
are preserved in `data/source_artifact_manifest.tsv` and the notebook
provenance sections.

Sun et al. evaluated PICRUSt, PICRUSt2, and Tax4Fun; this repository applies
their metric family to PICRUSt2, the basic-MLP adaptation, and Tax4Fun2.
Matchado et al. evaluated PICRUSt2, Tax4Fun2, PanFP, and MetGEM; this repository
applies their metric family to the three distributed comparators and accepted
contrasts. “Inspired” or “style” describes metric or visual provenance, not an
exact reproduction of the paper's datasets or full pipelines.

See `docs/REFERENCES.md` for complete citations and the line-by-line ownership
map separating published metrics, repository operational choices, and project
extensions.

Unavailable reference surfaces remain labeled unavailable; the notebooks do
not manufacture comparisons for results that were never completed.
