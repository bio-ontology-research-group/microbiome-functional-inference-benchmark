#!/usr/bin/env python3
"""Build the two lightweight, executable consolidation notebooks."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def notebook(cells):
    book = nbf.v4.new_notebook(cells=cells)
    book.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    book.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return book


COMMON = r'''
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks": ROOT = ROOT.parent
DATA = ROOT / "data"
STUDIES = DATA / "studies"
REFERENCE = DATA / "reference"
ANALYSIS = REFERENCE
FIGREF = REFERENCE
CONTRACT = REFERENCE
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from picrust_benchmark_metrics import (
    plot_directional_metric_summary,
    plot_directional_roc_grid,
    plot_ko_coverage,
    plot_matchado_randomization_dumbbells,
    plot_method_differences,
    plot_metric_row_grid,
    plot_profile_study_panels,
    plot_signed_decile_grid,
    plot_stratum_mean_ranges,
    plot_sun_observed_controls,
)
sns.set_theme(style="whitegrid", context="notebook")
METHODS = ["PICRUSt2", "PICRUSt2 + adaptation"]
COLORS = {"PICRUSt2": "#4C78A8", "PICRUSt2 + adaptation": "#E45756"}
COLORS["Tax4Fun2"] = "#F2A900"
assert STUDIES.is_dir() and REFERENCE.is_dir(), "Run from a complete repository checkout"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
'''


nb1 = notebook([
    md('''
    # 01 — Established functional-inference benchmarking metrics

    This notebook consolidates the established PICRUSt/PICRUSt2 profile benchmark and the published
    Matchado/Sun biological-group metric families for **PICRUSt2**, **PICRUSt2 + adaptation** (the cached
    basic-MLP `picrust2_teacher_only_residual` arm), and **Tax4Fun2**. It covers the complete accepted surface:

    - 23 independent biological outer groups, 26 study strata, and 1,639 paired samples;
    - **11 benchmark strata plus 15 expansion strata** (not 15 studies total);
    - 10 pre-audited biological contrasts where labels/phenotypes exist.

    No training, LOSO fitting, raw-read processing, or model selection occurs here. Numerical metrics
    are recomputed from the accepted sample/per-KO evaluation tables produced from frozen WGS and cached
    predictions. The source files remain read-only.
    '''),
    code(COMMON + r'''
METHODS = ["PICRUSt2", "PICRUSt2 + adaptation", "Tax4Fun2"]
'''),
    md('''
    ## Provenance and accepted surfaces

    The exact reference is `unified5_loso23_e3_six_method_comparison_r1`, whose validation report is
    PASS. The cells below fail closed on the study/sample counts and method identities.
    '''),
    code(r'''
sample = pd.read_csv(ANALYSIS / "sample_spearman_long.tsv.gz", sep="\t")
strata = pd.read_csv(ANALYSIS / "stratum_summary.tsv", sep="\t")
groups = pd.read_csv(ANALYSIS / "group_summary.tsv", sep="\t")
coverage = pd.read_csv(ANALYSIS / "coverage_by_stratum_method.tsv", sep="\t")
surface_ref = pd.read_csv(ANALYSIS / "surface_summary.tsv", sep="\t")
registry = pd.read_csv(CONTRACT / "study_strata.tsv", sep="\t")
cohort = pd.read_csv(ANALYSIS / "effect_cohort.tsv", sep="\t")
from replay_bundled_group_metrics import replay_bundled_group_metrics

(bundled_profile, published_matchado, published_sun,
 sun_randomizations, raw_sources) = replay_bundled_group_metrics(
    STUDIES, matchado_permutations=20, sun_randomizations=100
)
tax4fun_profile = pd.read_csv(REFERENCE / "tax4fun2_sample_spearman.tsv.gz", sep="\t")
tax4fun_coverage = pd.read_csv(REFERENCE / "tax4fun2_coverage.tsv", sep="\t")
tax4fun_reference_path = REFERENCE / "tax4fun2_completed_reference.tsv.gz"
tax4fun_reference = pd.read_csv(tax4fun_reference_path, sep="\t")
tax4fun_reference = tax4fun_reference[tax4fun_reference.prediction_method.eq("Tax4Fun2")].copy()

inputs_01 = [
    ANALYSIS / "sample_spearman_long.tsv.gz", ANALYSIS / "stratum_summary.tsv",
    ANALYSIS / "group_summary.tsv", ANALYSIS / "coverage_by_stratum_method.tsv",
    ANALYSIS / "surface_summary.tsv", ANALYSIS / "effect_cohort.tsv",
    CONTRACT / "study_strata.tsv", REFERENCE / "tax4fun2_sample_spearman.tsv.gz",
    REFERENCE / "tax4fun2_coverage.tsv", REFERENCE / "matchado_label_metric_results_long.tsv",
    tax4fun_reference_path,
    *raw_sources,
]
inputs_01 = list(dict.fromkeys(inputs_01))
provenance_01 = pd.DataFrame({"path": [p.relative_to(ROOT).as_posix() for p in inputs_01], "bytes": [p.stat().st_size for p in inputs_01],
                              "sha256": [sha256(p) for p in inputs_01]})

sample = pd.concat([
    sample[sample.method_label.isin(["PICRUSt2", "PICRUSt2 + adaptation"])].copy(),
    tax4fun_profile[sample.columns],
], ignore_index=True)
strata = strata[strata.method_label.isin(METHODS)].copy()
groups = groups[groups.method_label.isin(METHODS)].copy()
coverage = pd.concat([
    coverage[coverage.method.isin(["PICRUSt2", "PICRUSt2 + adaptation"])].copy(),
    tax4fun_coverage,
], ignore_index=True)
surface_ref = surface_ref[surface_ref.method.isin(METHODS)].copy()
assert sample.groupby("method_label").size().to_dict() == {
    "PICRUSt2": 1639, "PICRUSt2 + adaptation": 1639, "Tax4Fun2": 1637,
}
assert sample.dataset.nunique() == 26 and sample.heldout_group.nunique() == 23
assert registry.is_expansion_group.value_counts().to_dict() == {True: 15, False: 11}
assert cohort.dataset.nunique() == 10
assert published_matchado.groupby("prediction_method").dataset.nunique().eq(10).all()
assert published_sun.groupby("prediction_method").dataset.nunique().eq(10).all()
assert sun_randomizations.groupby(["dataset", "prediction_method"]).size().eq(100).all()
print(f"Local replay used {len(raw_sources)} bundled processed studies")
display(tax4fun_coverage.loc[tax4fun_coverage.n_zero_total_samples.gt(0),
                             ["dataset", "n_samples", "n_zero_total_samples"]])
display(cohort[["dataset", "group_a", "group_b", "n_group_a", "n_group_b", "n_eligible_kos"]])
display(provenance_01)
'''),
    md('''
    ## Paper-level metric audit

    The local papers and downloaded author code were checked in addition to the completed project results.
    This prevents conflating a metric with a dataset name and makes missing input surfaces explicit.
    '''),
    code(r'''
paper_audit = pd.DataFrame([
    {"source": "PICRUSt2 Fig. 2a", "metric": "sample KO-abundance Spearman", "status": "replayed: 26 strata, three methods; paper-style faceted box/point panels"},
    {"source": "PICRUSt2 Fig. 2b", "metric": "BH/FDR differential-KO precision, recall, F1", "status": "accepted BH-family extension replayed in Notebook 02; exact universal-single-copy normalization/ASV shuffle unavailable"},
    {"source": "PICRUSt2 Fig. 3a", "metric": "MetaCyc pathway-abundance Spearman", "status": "unavailable: accepted surface is KO, not pathway abundance"},
    {"source": "PICRUSt2 Fig. 3b", "metric": "reference-genome binary phenotype precision, recall, F1", "status": "out of scope: genome phenotype prediction, not paired community functional profiles"},
    {"source": "Matchado Figs. 3/4/6/7", "metric": "raw-p differential-KO precision, recall, F1 + 20 label permutations", "status": "replayed below: 10 accepted contrasts, three methods; bar grid plus dumbbells"},
    {"source": "Sun Figs. 3/4", "metric": "signed log10(p) inference Spearman + 100 WGS abundance permutations", "status": "replayed below: 10 accepted contrasts, three methods; observed points over control boxplots"},
    {"source": "Sun Fig. 1", "metric": "sample profile Spearman + abundance-permutation control", "status": "observed Spearman replayed; separate composition-null analysis intentionally not added"},
    {"source": "Sun supplementary", "metric": "5+5 sample-size sensitivity and KEGG-category stratification", "status": "unavailable on frozen accepted result surface; secondary sensitivity, not invented"},
])
display(paper_audit)
'''),
    md('''
    ## PICRUSt2 profile metrics

    The accepted rule fixes the WGS-positive KO universe within each stratum, zero-fills KOs absent from
    a prediction, L1-closes each sample, and computes Spearman correlation across KOs. Below we recompute
    every stratum/group/surface summary from 4,915 valid sample-method rows and compare to the frozen tables.
    Tax4Fun2 contributes 1,637 rows because two olive-xylem samples are zero-total on the fixed WGS universe.
    '''),
    code(r'''
recomputed_strata = (sample.groupby(["dataset", "heldout_group", "method_label"], as_index=False)
                     .agg(n_samples=("sample_id", "nunique"), mean_sample_spearman=("sample_spearman", "mean"),
                          sd_sample_spearman=("sample_spearman", "std")))
cmp = recomputed_strata.merge(strata[["dataset", "method_label", "mean_sample_spearman"]],
                             on=["dataset", "method_label"], suffixes=("_new", "_reference"), validate="one_to_one")
cmp["absolute_error"] = (cmp.mean_sample_spearman_new - cmp.mean_sample_spearman_reference).abs()

recomputed_surface = (sample.groupby("method_label", as_index=False)
                      .agg(pooled_sample_mean_spearman=("sample_spearman", "mean"), n_samples=("sample_id", "size")))
surface_cmp = recomputed_surface[recomputed_surface.method_label.ne("Tax4Fun2")].merge(
                                       surface_ref[["method", "pooled_sample_mean_spearman"]],
                                       left_on="method_label", right_on="method", suffixes=("_new", "_reference"))
surface_cmp["absolute_error"] = (surface_cmp.pooled_sample_mean_spearman_new - surface_cmp.pooled_sample_mean_spearman_reference).abs()

tax4fun_cmp = tax4fun_profile.merge(
    tax4fun_reference[["dataset", "sample_id", "spearman_r"]],
    on=["dataset", "sample_id"], suffixes=("_new", "_reference"), validate="one_to_one",
)
tax4fun_cmp["absolute_error"] = (tax4fun_cmp.spearman_r_new - tax4fun_cmp.spearman_r_reference).abs()
bundle_profile_cmp = bundled_profile.merge(
    sample[["dataset", "sample_id", "method_label", "sample_spearman"]],
    left_on=["dataset", "sample_id", "prediction_method"],
    right_on=["dataset", "sample_id", "method_label"],
    validate="one_to_one",
)
bundle_profile_cmp["absolute_error"] = (
    bundle_profile_cmp.spearman_r - bundle_profile_cmp.sample_spearman
).abs()
assert cmp.absolute_error.max() <= 1e-12
assert surface_cmp.absolute_error.max() <= 1e-12
assert tax4fun_cmp.absolute_error.max() <= 1e-12 and len(tax4fun_cmp) == 1218
assert bundle_profile_cmp.absolute_error.max() <= 1e-12
display(recomputed_surface)
print("Maximum 26-stratum mean absolute error:", f"{cmp.absolute_error.max():.3g}")
print("Maximum Tax4Fun2 completed-reference absolute error:", f"{tax4fun_cmp.absolute_error.max():.3g}",
      f"across {len(tax4fun_cmp)} samples / 19 completed-reference strata")
print("Maximum bundled-matrix profile error:", f"{bundle_profile_cmp.absolute_error.max():.3g}",
      f"across {len(bundle_profile_cmp)} study-method-sample rows")
'''),
    code(r'''
# Plot 1: PICRUSt2-Fig.-2a-style per-study distributions for all three methods.
study_order = registry.dataset.tolist()
fig, axes = plot_profile_study_panels(sample, registry, METHODS, colors=COLORS)
'''),
    code(r'''
# Plot 2: compact three-method stratum-mean summary.
from matplotlib.lines import Line2D
means = recomputed_strata.merge(registry[["dataset", "is_expansion_group"]], on="dataset", validate="many_to_one")
fig, ax, wide = plot_stratum_mean_ranges(means, METHODS, colors=COLORS)
'''),
    code(r'''
# Plot 3: both alternative methods relative to PICRUSt2 across the 23 outer groups.
group_means = (sample.groupby(["heldout_group", "method_label"], as_index=False)
               .agg(group_mean_sample_spearman=("sample_spearman", "mean")))
fig, axes, gdelta = plot_method_differences(
    group_means, ["PICRUSt2 + adaptation", "Tax4Fun2"], baseline_method="PICRUSt2"
)
display(gdelta.describe())
'''),
    code(r'''
# Plot 4: WGS-universe coverage/zero-fill audit for all three methods.
fig, ax = plot_ko_coverage(coverage, METHODS, colors=COLORS)
'''),
    md('''
    ## Matchado-style differential-KO recovery

    The library implements both (a) the paper-faithful legacy family—shared 5% prevalence, unadjusted
    two-sided rank-sum p < 0.05, precision/recall/F1, prediction-label permutations—and (b) the later
    project extension using a WGS-only family with BH q < 0.05 and zero-fill. The completed paper-faithful
    reference has PICRUSt2 and Tax4Fun2 results for the earlier labeled surface; it predates the basic-MLP
    adaptation and the two expansion contrasts, so there is no honest adaptation reference to compare on
    that surface. Notebook 02 evaluates the later BH extension for its deliberately fixed two-method scope.
    '''),
    code(r'''
legacy_path = REFERENCE / "matchado_label_metric_results_long.tsv"
legacy = pd.read_csv(legacy_path, sep="\t")
legacy_core = legacy[
    legacy.prediction_method.isin(["PICRUSt2", "Tax4Fun2"]) &
    legacy.metric.isin(["precision", "recall", "f1"]) & legacy.dataset.isin(cohort.dataset)
].copy()
legacy_cmp = published_matchado.merge(
    legacy_core[["dataset", "prediction_method", "metric", "estimate", "group_a", "group_b"]],
    on=["dataset", "prediction_method", "metric", "group_a", "group_b"],
    suffixes=("_new", "_reference"), validate="one_to_one",
)
legacy_cmp["absolute_error"] = (legacy_cmp.estimate_new - legacy_cmp.estimate_reference).abs()
print(f"Label-matched legacy Matchado rows: {len(legacy_cmp)}; "
      f"max difference across the non-equivalent earlier KO surface: {legacy_cmp.absolute_error.max():.3g}")

# Plot 5: completed-reference Matchado rows in the paper's metric-row grammar.
metric_order = ["f1", "recall", "precision"]
legacy_studies = list(dict.fromkeys(legacy_core.dataset))
fig, axes = plot_metric_row_grid(
    legacy_core, legacy_studies, ["PICRUSt2", "Tax4Fun2"], metric_order=metric_order,
    colors=COLORS, title="Completed Matchado reference - F1, recall, and precision rows"
)
display(legacy_cmp)
'''),
    md('''
    ## Full paper-faithful Matchado and Sun replay

    The replay loaded above reads frozen WGS/native/cached-adaptation/Tax4Fun2 matrices and audited labels,
    then calls the clean library for every eligible study. Matchado uses its published 20 prediction-label
    permutations. Sun uses signed `log10(p)` inference correlation and 100 independent WGS-KO abundance
    shuffles with labels fixed. The figures below deliberately reuse each paper's visual grammar.
    '''),
    code(r'''
assert published_matchado.dataset.nunique() == published_sun.dataset.nunique() == 10
assert set(published_matchado.prediction_method) == set(published_sun.prediction_method) == set(METHODS)
display(published_matchado[["dataset", "prediction_method", "metric", "estimate", "permutation_mean"]].head(10))
display(published_sun[["dataset", "prediction_method", "observed_inference_correlation", "randomization_mean"]].head(10))
'''),
    code(r'''
# Plot 6: Matchado-Fig.-6-style set recovery for three methods and all 10 contrasts.
set_metrics = published_matchado[published_matchado.metric.isin(["precision", "recall", "f1"])]
MATCHADO_ORDER = ["cameroon", "hmp", "blueberry", "indian", "primate", "sun_human_kw",
                  "matchado", "brooks_forward", "activated_sludge", "han_river"]
fig, axes = plot_metric_row_grid(
    set_metrics, MATCHADO_ORDER, METHODS, metric_order=["f1", "recall", "precision"],
    colors=COLORS, title="Paper-faithful Matchado differential-KO recovery"
)
'''),
    code(r'''
# Plot 7: Matchado-Fig.-7-style observed-versus-randomized dumbbells (lollipops).
fig, axes = plot_matchado_randomization_dumbbells(
    published_matchado, MATCHADO_ORDER, METHODS,
    metric_order=["f1", "precision", "recall"]
)
'''),
    code(r'''
# Plot 8: additional threshold-free ranking metrics returned by the reusable implementation.
ranking_metrics = published_matchado[published_matchado.metric.isin(["auroc", "aupr"])].copy()
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for ax, metric in zip(axes, ["auroc", "aupr"]):
    block = ranking_metrics[ranking_metrics.metric.eq(metric)]
    sns.barplot(data=block, x="dataset", y="estimate", hue="prediction_method",
                hue_order=METHODS, palette=COLORS, ax=ax)
    ax.set(title=metric.upper(), xlabel="", ylabel="Observed ranking metric")
    ax.tick_params(axis="x", rotation=65)
    if ax is axes[1]: ax.legend_.remove()
axes[0].legend(title="", fontsize=8)
fig.suptitle("Matchado-family WGS-significant-KO ranking (library extension)")
fig.tight_layout()
'''),
    code(r'''
# Plot 9: Sun-Fig.-4-style red observed points over blue 100-randomization controls.
fig, axes = plot_sun_observed_controls(
    published_sun, sun_randomizations, MATCHADO_ORDER, METHODS
)
'''),
    md('''
    The raw 10-study Matchado and Sun results above have no completed source table on this exact
    three-method/accepted-contrast surface. Their full-surface comparison status is therefore
    **unavailable**, not PASS. Directly comparable legacy Matchado rows and the completed 19-study
    Tax4Fun2 profile surface are checked where possible.
    '''),
    code(r'''
metric_contract = pd.DataFrame([
    {"family": "KO profile", "studies": 26, "methods": 3, "reference": "accepted two-method table + completed 19-study Tax4Fun2 table", "validation": f"two-method error <=1e-12; Tax4Fun2 max error {tax4fun_cmp.absolute_error.max():.3g} on 1,218 samples"},
    {"family": "Matchado paper-faithful", "studies": 10, "methods": 3, "reference": "completed legacy PICRUSt2/Tax4Fun2 rows where labels match", "validation": f"{len(legacy_cmp)} label-matched rows are surface-non-equivalent (max difference {legacy_cmp.absolute_error.max():.3g}); full three-method reference unavailable"},
    {"family": "Matchado BH extension", "studies": 10, "methods": 2, "reference": "accepted unified5 tables", "validation": "retained in Notebook 02's fixed two-method scope"},
    {"family": "Sun inference correlation", "studies": 10, "methods": 3, "reference": "none on exact accepted surface", "validation": "full 100-randomization replay complete; comparison unavailable"},
])
display(metric_contract)
'''),
])


nb2 = notebook([
    md('''
    # 02 — Project-learned effect, directional ranking, and decile diagnostics

    This notebook consolidates the analyses developed after the published benchmark metrics. It uses the
    **frozen, already-audited BH-corrected WGS reference**; it does not rediscover labels, select studies,
    or rerun multiple-testing choices. Every panel covers all 10 eligible biological contrasts and compares
    PICRUSt2 with the cached basic-MLP adaptation.

    WGS-derived labels are target-informed, cohort-conditional, post-hoc diagnostics and are not deployable
    prediction targets.
    '''),
    code(COMMON + r'''
from picrust_benchmark_metrics import directional_ranking_curves, decile_outcome_composition
'''),
    md('''
    ## Frozen biological contrasts and per-KO surface

    The study/label selection and BH corrections were completed in the accepted source run. Topsoil is not
    included because it had zero BH-selected WGS KOs; the other 16 strata have no accepted binary phenotype.
    '''),
    code(r'''
cohort = pd.read_csv(ANALYSIS / "effect_cohort.tsv", sep="\t")
selection = pd.read_csv(ANALYSIS / "effect_selection_fixed.tsv", sep="\t")
assignments = pd.read_csv(ANALYSIS / "quantile_per_ko_assignments.tsv.gz", sep="\t")
assignments = assignments[assignments.prediction_method.isin(METHODS)].copy()
effect_ref = pd.read_csv(ANALYSIS / "effect_long.tsv", sep="\t")
effect_ref = effect_ref[effect_ref.method_label.isin(METHODS)].copy()
class_ref = pd.read_csv(ANALYSIS / "classification_summary.tsv", sep="\t")
class_ref = class_ref[class_ref.prediction_method.isin(METHODS)].copy()
decile_ref = pd.read_csv(ANALYSIS / "quantile_composition_by_bin.tsv", sep="\t")
decile_ref = decile_ref[decile_ref.prediction_method.isin(METHODS)].copy()
decile_baselines = pd.read_csv(ANALYSIS / "quantile_study_baselines.tsv", sep="\t")
inputs_02 = [
    ANALYSIS / "effect_cohort.tsv", ANALYSIS / "effect_selection_fixed.tsv",
    ANALYSIS / "quantile_per_ko_assignments.tsv.gz", ANALYSIS / "effect_long.tsv",
    ANALYSIS / "classification_summary.tsv", ANALYSIS / "quantile_composition_by_bin.tsv",
    ANALYSIS / "quantile_study_baselines.tsv",
    FIGREF / "pr_roc_metric_summary.tsv",
]
provenance_02 = pd.DataFrame({"path": [p.relative_to(ROOT).as_posix() for p in inputs_02], "bytes": [p.stat().st_size for p in inputs_02],
                              "sha256": [sha256(p) for p in inputs_02]})
assert cohort.dataset.nunique() == selection.dataset.nunique() == assignments.dataset.nunique() == 10
assert assignments.groupby(["dataset", "prediction_method"]).ngroups == 20
display(cohort[["dataset", "group_a", "group_b", "n_group_a", "n_group_b", "n_eligible_kos"]])
display(provenance_02)
'''),
    md('''
    ## WGS-versus-predicted effect-size relation

    Rank-biserial effects use `2U/(nA*nB)-1`. Spearman and Pearson correlations are recomputed directly
    from every eligible per-KO row and checked against the accepted `effect_long.tsv` values.
    '''),
    code(r'''
effect_rows = []
for (dataset, method), block in assignments.groupby(["dataset", "prediction_method"], sort=False):
    selected = block[block.wgs_bh_qvalue < 0.05]
    effect_rows += [
        {"dataset": dataset, "method_label": method, "metric": "spearman",
         "effect_correlation_new": selected.wgs_rank_biserial_effect.corr(selected.prediction_effect, method="spearman")},
        {"dataset": dataset, "method_label": method, "metric": "pearson",
         "effect_correlation_new": selected.wgs_rank_biserial_effect.corr(selected.prediction_effect, method="pearson")},
    ]
effect_new = pd.DataFrame(effect_rows)
effect_cmp = effect_new.merge(effect_ref[["dataset", "method_label", "metric", "effect_correlation"]],
                              on=["dataset", "method_label", "metric"], validate="one_to_one")
effect_cmp["absolute_error"] = (effect_cmp.effect_correlation_new - effect_cmp.effect_correlation).abs()
assert effect_cmp.absolute_error.max() <= 1e-12
print("Maximum effect-correlation absolute error:", f"{effect_cmp.absolute_error.max():.3g}")
pivot = effect_cmp[effect_cmp.metric.eq("spearman")].pivot(index="dataset", columns="method_label", values="effect_correlation_new")
display(pivot)
'''),
    code(r'''
# Plot 1: all-study effect-correlation comparison.
fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(data=effect_cmp[effect_cmp.metric.eq("spearman")], x="dataset", y="effect_correlation_new",
            hue="method_label", hue_order=METHODS, palette=COLORS, ax=ax)
ax.axhline(0, color="black", lw=.8)
ax.set(xlabel="Biological contrast", ylabel="Spearman(WGS effect, predicted effect)",
       title="WGS-versus-predicted KO effects: all 10 eligible studies")
ax.tick_params(axis="x", rotation=55)
ax.legend(title="")
fig.tight_layout()
'''),
    code(r'''
# Plot 2: per-KO effect scatter for every study and both methods.
fig, axes = plt.subplots(5, 4, figsize=(15, 18), sharex=True, sharey=True)
for ax, ((dataset, method), block) in zip(axes.flat, assignments.groupby(["dataset", "prediction_method"], sort=False)):
    ax.hexbin(block.wgs_rank_biserial_effect, block.prediction_effect, gridsize=28, mincnt=1, cmap="viridis")
    ax.axline((0, 0), slope=1, color="white", lw=.7, alpha=.8)
    ax.set_title(f"{dataset}\n{method}", fontsize=9)
for ax in axes[-1]: ax.set_xlabel("WGS effect")
for ax in axes[:, 0]: ax.set_ylabel("Predicted effect")
fig.suptitle("Eligible-KO effect relation (20 study-method panels)", y=.998)
fig.tight_layout()
'''),
    md('''
    ## Directional PR and ROC

    Positive targets are WGS BH q ≤ 0.05 with positive effect; negative targets use q ≤ 0.05 with negative
    effect and score `−prediction_effect`. AP is sklearn average precision—not trapezoidal PR area.
    Every summary is independently recomputed with the clean library and compared to the completed figure
    source table.
    '''),
    code(r'''
curve_ref = pd.read_csv(FIGREF / "pr_roc_metric_summary.tsv", sep="\t")
curve_ref = curve_ref[curve_ref.prediction_method.isin(METHODS)].copy()
curve_rows, pr_tables, roc_tables = [], [], []
for (dataset, method), block in assignments.groupby(["dataset", "prediction_method"], sort=False):
    pr, roc, summary = directional_ranking_curves(block.wgs_bh_qvalue, block.wgs_rank_biserial_effect, block.prediction_effect)
    summary.insert(0, "prediction_method", method); summary.insert(0, "dataset", dataset)
    pr.insert(0, "prediction_method", method); pr.insert(0, "dataset", dataset)
    roc.insert(0, "prediction_method", method); roc.insert(0, "dataset", dataset)
    curve_rows.append(summary); pr_tables.append(pr); roc_tables.append(roc)
curve_new = pd.concat(curve_rows, ignore_index=True)
pr_values = pd.concat(pr_tables, ignore_index=True); roc_values = pd.concat(roc_tables, ignore_index=True)
curve_cmp = curve_new.merge(curve_ref[["dataset", "prediction_method", "direction", "ap_sklearn", "roc_auc_sklearn"]],
                            on=["dataset", "prediction_method", "direction"], validate="one_to_one")
curve_cmp["ap_error"] = (curve_cmp.ap - curve_cmp.ap_sklearn).abs()
curve_cmp["roc_error"] = (curve_cmp.roc_auc - curve_cmp.roc_auc_sklearn).abs()
assert curve_cmp[["ap_error", "roc_error"]].max().max() <= 1e-12
print("Maximum AP/ROC-AUC absolute error:", f"{curve_cmp[['ap_error','roc_error']].max().max():.3g}")
'''),
    code(r'''
# Plot 3: directional average precision across all studies.
fig, axes = plot_directional_metric_summary(curve_new, "ap", METHODS, colors=COLORS)
'''),
    code(r'''
# Plot 4: directional ROC-AUC across all studies.
fig, axes = plot_directional_metric_summary(curve_new, "roc_auc", METHODS, colors=COLORS)
'''),
    code(r'''
# Plot 5: full directional ROC curves, with the independently recomputed AUC in every panel.
from matplotlib.lines import Line2D

STUDY_ORDER = ["cameroon", "hmp", "blueberry", "indian", "primate", "sun_human_kw",
               "matchado", "brooks_forward", "activated_sludge", "han_river"]
display_names = (effect_ref[["dataset", "display_name"]].drop_duplicates("dataset")
                 .set_index("dataset").display_name.to_dict())
fig, axes = plot_directional_roc_grid(
    roc_values, curve_new, STUDY_ORDER, METHODS,
    display_names=display_names, colors=COLORS, benchmark_count=8
)
'''),
    md('''
    ## Matchado-inspired BH recovery

    This is the project adaptation of Matchado’s set-recovery idea: the WGS-only detectable family is tested
    once, BH-corrected once, and shared by both methods. The plot uses the completed 400-permutation summary;
    it is not a new null analysis in this notebook.
    '''),
    code(r'''
assert class_ref.groupby(["dataset", "prediction_method"]).ngroups == 20
fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(data=class_ref, x="dataset", y="f1", hue="prediction_method", hue_order=METHODS, palette=COLORS, ax=ax)
ax.set(xlabel="Biological contrast", ylabel="BH significant-KO recovery F1",
       title="Shared-WGS-family Matchado extension (all 10 studies)")
ax.tick_params(axis="x", rotation=55)
ax.legend(title="")
fig.tight_layout()
display(class_ref[["dataset", "prediction_method", "precision", "recall", "f1", "n_wgs_positive", "n_prediction_positive_seed_mean"]])
'''),
    md('''
    ## Decile outcome composition

    KOs are sorted by predicted effect (ties broken lexically by KO ID), divided into ten equal-count rank
    bins, and summarized by WGS class: significant negative, nonsignificant, or significant positive. This
    provides the directional WGS outcome composition in each rank bin.
    '''),
    code(r'''
new_assignments, new_bins = [], []
for (dataset, method), block in assignments.groupby(["dataset", "prediction_method"], sort=False):
    assigned, bins = decile_outcome_composition(block.KO, block.prediction_effect,
                                                block.wgs_bh_qvalue, block.wgs_rank_biserial_effect)
    assigned.insert(0, "prediction_method", method); assigned.insert(0, "dataset", dataset)
    bins.insert(0, "prediction_method", method); bins.insert(0, "dataset", dataset)
    new_assignments.append(assigned); new_bins.append(bins)
decile_new = pd.concat(new_bins, ignore_index=True)
keys = ["dataset", "prediction_method", "bin"]
check_cols = ["n_kos", "p_sig_negative", "p_nonsignificant", "p_sig_positive", "p_any_significant", "balance_pos_minus_neg"]
decile_cmp = decile_new.merge(decile_ref[keys + check_cols], on=keys, suffixes=("_new", "_reference"), validate="one_to_one")
errors = []
for column in check_cols:
    errors.append(np.nanmax(np.abs(decile_cmp[f"{column}_new"].to_numpy(float) - decile_cmp[f"{column}_reference"].to_numpy(float))))
assert max(errors) <= 1e-12
print("Maximum decile-summary absolute error:", f"{max(errors):.3g}")
'''),
    code(r'''
# Plot 6: signed decile composition, matching the consolidated all-study layout.

STUDY_ORDER = ["cameroon", "hmp", "blueberry", "indian", "primate", "sun_human_kw",
               "matchado", "brooks_forward", "activated_sludge", "han_river"]
display_names = (effect_ref[["dataset", "display_name"]].drop_duplicates("dataset")
                 .set_index("dataset").display_name.to_dict())
fig, axes = plot_signed_decile_grid(
    decile_new, decile_baselines, cohort, STUDY_ORDER, METHODS,
    display_names=display_names, colors=COLORS, benchmark_count=8
)
'''),
    code(r'''
# Plot 7: compact directional gradient (positive minus negative outcome fraction).
fig, axes = plt.subplots(5, 2, figsize=(12, 16), sharex=True, sharey=True)
for ax, (dataset, block) in zip(axes.flat, decile_new.groupby("dataset", sort=False)):
    sns.lineplot(data=block, x="bin", y="balance_pos_minus_neg", hue="prediction_method",
                 hue_order=METHODS, palette=COLORS, marker="o", ax=ax)
    ax.axhline(0, color="black", lw=.6)
    ax.set_title(dataset)
    if ax is not axes.flat[0] and ax.legend_: ax.legend_.remove()
axes.flat[0].legend(title="", fontsize=8)
fig.suptitle("Decile direction gradient: WGS positive minus negative fraction", y=.998)
fig.tight_layout()
'''),
    md('''
    ## Validation summary and scientific boundaries

    All feasible families agree numerically with completed reference tables at 1e−12 absolute tolerance.
    The source provides no completed accepted-surface Sun randomization table, so that comparison remains
    explicitly unavailable rather than inferred. Remaining scientific choices include whether to treat the
    23 outer groups or 26 strata as the reporting unit, whether q < 0.05 or q ≤ 0.05 is desired (the frozen
    accepted selection is retained), and whether future work should average multiple adaptation seeds.
    '''),
    code(r'''
validation = pd.DataFrame([
    {"family": "effect-size correlation", "studies": 10, "methods": 2, "max_abs_error": effect_cmp.absolute_error.max(), "tolerance": 1e-12, "status": "PASS"},
    {"family": "directional AP / ROC-AUC", "studies": 10, "methods": 2, "max_abs_error": curve_cmp[["ap_error", "roc_error"]].max().max(), "tolerance": 1e-12, "status": "PASS"},
    {"family": "decile composition", "studies": 10, "methods": 2, "max_abs_error": max(errors), "tolerance": 1e-12, "status": "PASS"},
    {"family": "BH set recovery", "studies": 10, "methods": 2, "max_abs_error": np.nan, "tolerance": np.nan, "status": "accepted completed table displayed"},
    {"family": "Sun abundance randomization", "studies": 10, "methods": 2, "max_abs_error": np.nan, "tolerance": np.nan, "status": "reference unavailable"},
])
display(validation)
'''),
])


NB.mkdir(exist_ok=True)
nbf.write(nb1, NB / "01_established_profile_and_group_recovery.ipynb")
nbf.write(nb2, NB / "02_exploratory_effect_ranking_and_deciles.ipynb")
print("Wrote", NB / "01_established_profile_and_group_recovery.ipynb")
print("Wrote", NB / "02_exploratory_effect_ranking_and_deciles.ipynb")
