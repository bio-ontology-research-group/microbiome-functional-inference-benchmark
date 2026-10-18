"""Reusable figures shown in the two benchmark notebooks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_COLORS = {
    "PICRUSt2": "#4C78A8",
    "PICRUSt2 + adaptation": "#E45756",
    "Tax4Fun2": "#F2A900",
}
POSITIVE_COLOR = "#B2182B"
NEGATIVE_COLOR = "#2166AC"


def _palette(methods: Sequence[str], colors: Mapping[str, str] | None) -> dict[str, str]:
    selected = dict(DEFAULT_COLORS)
    if colors:
        selected.update(colors)
    fallback = sns.color_palette("colorblind", n_colors=len(methods))
    return {method: selected.get(method, fallback[index]) for index, method in enumerate(methods)}


def plot_profile_study_panels(
    sample_metrics: pd.DataFrame,
    registry: pd.DataFrame,
    methods: Sequence[str],
    *,
    colors: Mapping[str, str] | None = None,
    ncols: int = 7,
):
    """PICRUSt2 Fig. 2a-inspired panels for every study stratum."""
    palette = _palette(methods, colors)
    study_order = registry.dataset.astype(str).tolist()
    nrows = math.ceil(len(study_order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.15 * ncols, 3.2 * nrows), sharey=True, squeeze=False)
    expansion = registry.set_index("dataset").is_expansion_group.to_dict()
    for ax, dataset in zip(axes.flat, study_order):
        block = sample_metrics[sample_metrics.dataset.eq(dataset)]
        sns.boxplot(data=block, x="method_label", y="sample_spearman", hue="method_label",
                    order=methods, hue_order=methods, palette=palette, dodge=False, legend=False,
                    width=.62, showfliers=False, linewidth=.8, ax=ax)
        sns.stripplot(data=block, x="method_label", y="sample_spearman", order=methods,
                      color="black", size=1.7, jitter=.18, alpha=.45, ax=ax)
        title = block.display_name.iloc[0] if "display_name" in block and len(block) else dataset
        ax.set_title(f"{title}\n(n={block.sample_id.nunique()})", fontsize=9,
                     color="#B45F06" if expansion.get(dataset, False) else "#222222",
                     bbox={"facecolor": "#E8E8E8", "edgecolor": "none", "pad": 3})
        ax.set(xlabel="", ylabel="")
        ax.tick_params(axis="x", rotation=55, labelsize=7)
        ax.grid(False)
    for ax in axes.flat[len(study_order):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Spearman correlation")
    fig.suptitle("Sample-level KO profile agreement\nblack: benchmark strata; orange: expansion strata", y=.998)
    fig.tight_layout(rect=(0, 0, 1, .975))
    return fig, axes


def plot_stratum_mean_ranges(
    stratum_metrics: pd.DataFrame,
    methods: Sequence[str],
    *,
    colors: Mapping[str, str] | None = None,
):
    """Compact ranges and method points for study-stratum mean correlations."""
    palette = _palette(methods, colors)
    wide = stratum_metrics.pivot(index=["dataset", "is_expansion_group"], columns="method_label",
                                 values="mean_sample_spearman").reset_index()
    wide = wide.sort_values(methods[0])
    fig, ax = plt.subplots(figsize=(10, max(5, .34 * len(wide))))
    for index, (_, row) in enumerate(wide.iterrows()):
        values = [row[method] for method in methods]
        ax.plot([min(values), max(values)], [index, index], color="#C8C8C8", lw=1)
        for method in methods:
            ax.scatter(row[method], index, color=palette[method], s=38,
                       marker="s" if row.is_expansion_group else "o")
    ax.set(yticks=range(len(wide)), yticklabels=wide.dataset,
           xlabel="Mean sample-wise Spearman",
           title="Study-stratum means (circles: benchmark; squares: expansion)")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[m],
                              markeredgecolor="none", label=m) for m in methods], frameon=False)
    fig.tight_layout()
    return fig, ax, wide


def plot_method_differences(
    group_metrics: pd.DataFrame,
    comparison_methods: Sequence[str],
    *,
    baseline_method: str = "PICRUSt2",
):
    """Horizontal paired differences from one baseline method."""
    wide = group_metrics.pivot(index="heldout_group", columns="method_label",
                               values="group_mean_sample_spearman")
    difference = wide[list(comparison_methods)].subtract(wide[baseline_method], axis=0)
    difference = difference.sort_values(comparison_methods[0])
    fig, axes = plt.subplots(1, len(comparison_methods), figsize=(7.5 * len(comparison_methods), 8),
                             sharey=True, squeeze=False)
    for ax, method in zip(axes.flat, comparison_methods):
        values = difference[method]
        ax.barh(values.index, values, color=np.where(values >= 0, "#59A14F", "#E15759"))
        ax.axvline(0, color="black", lw=1)
        ax.set(title=f"{method} - {baseline_method}", xlabel="Difference in mean Spearman")
    axes.flat[0].set_ylabel("Independent outer group")
    fig.suptitle(f"Paired outer-group profile differences (N = {len(difference)})")
    fig.tight_layout()
    return fig, axes, difference


def plot_ko_coverage(coverage: pd.DataFrame, methods: Sequence[str], *, colors=None):
    """Plot the fraction of the WGS-positive KO universe present for each method."""
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.scatterplot(data=coverage, x="dataset", y="fraction_universe_in_head", hue="method", style="method",
                    hue_order=methods, palette=_palette(methods, colors), s=65, ax=ax)
    ax.set(xlabel="Study stratum", ylabel="Fraction of WGS-positive KOs present",
           title=f"KO coverage across {coverage.dataset.nunique()} study strata")
    ax.tick_params(axis="x", rotation=75)
    ax.legend(title="")
    fig.tight_layout()
    return fig, ax


def plot_metric_row_grid(
    metrics: pd.DataFrame,
    study_order: Sequence[str],
    methods: Sequence[str],
    *,
    metric_order: Sequence[str] = ("f1", "recall", "precision"),
    colors=None,
    title: str = "Differential-KO recovery",
):
    """Matchado Fig. 6-inspired metric rows with one study per column."""
    palette = _palette(methods, colors)
    fig, axes = plt.subplots(len(metric_order), len(study_order),
                             figsize=(2.5 * len(study_order), 2.65 * len(metric_order)),
                             sharex=True, sharey=True, squeeze=False)
    for row, metric in enumerate(metric_order):
        for col, dataset in enumerate(study_order):
            ax = axes[row, col]
            block = metrics[(metrics.metric.eq(metric)) & (metrics.dataset.eq(dataset))].set_index("prediction_method")
            ax.bar(np.arange(len(methods)), block.loc[list(methods), "estimate"],
                   color=[palette[method] for method in methods], width=.72)
            ax.set(ylim=(0, 1), xticks=range(len(methods)), xticklabels=methods)
            ax.tick_params(axis="x", rotation=60, labelsize=7)
            ax.grid(False)
            if row == 0:
                ax.set_title(dataset, fontsize=8, weight="bold")
            if col == 0:
                ax.set_ylabel(metric.upper())
    fig.suptitle(title, y=.995)
    fig.tight_layout()
    return fig, axes


def plot_matchado_randomization_dumbbells(
    metrics: pd.DataFrame,
    study_order: Sequence[str],
    methods: Sequence[str],
    *,
    metric_order: Sequence[str] = ("f1", "precision", "recall"),
):
    """Matchado Fig. 7-inspired observed-versus-randomized dumbbells."""
    fig, axes = plt.subplots(len(metric_order), len(study_order),
                             figsize=(2.5 * len(study_order), 2.65 * len(metric_order)),
                             sharex=True, sharey=True, squeeze=False)
    for row, metric in enumerate(metric_order):
        for col, dataset in enumerate(study_order):
            ax = axes[row, col]
            block = metrics[(metrics.metric.eq(metric)) & metrics.dataset.eq(dataset)].set_index("prediction_method").loc[list(methods)]
            x = np.arange(len(methods))
            for index, (_, value) in enumerate(block.iterrows()):
                ax.plot([index, index], [value.permutation_mean, value.estimate], color="#AEB6BF", lw=1.2)
            ax.scatter(x, block.permutation_mean, color="#574B90", s=24, zorder=3, label="randomization mean")
            ax.scatter(x, block.estimate, color="#F17CB0", s=28, zorder=4, label="observed")
            ax.set(ylim=(0, 1), xticks=x, xticklabels=methods)
            ax.tick_params(axis="x", rotation=60, labelsize=7)
            ax.grid(False)
            if row == 0:
                ax.set_title(dataset, fontsize=8, weight="bold")
            if col == 0:
                ax.set_ylabel(metric.upper())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .96))
    fig.suptitle("Observed recovery versus label-randomization mean", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return fig, axes


def plot_sun_observed_controls(
    observed: pd.DataFrame,
    controls: pd.DataFrame,
    study_order: Sequence[str],
    methods: Sequence[str],
):
    """Sun Fig. 4-inspired observed correlations over abundance controls."""
    fig, axes = plt.subplots(len(methods), 1, figsize=(17, 3.6 * len(methods)), sharex=True, sharey=True, squeeze=False)
    for ax, method in zip(axes.flat, methods):
        control = controls[controls.prediction_method.eq(method)]
        point = observed[observed.prediction_method.eq(method)].set_index("dataset")
        sns.boxplot(data=control, x="dataset", y="inference_correlation", order=study_order,
                    color="#DCE6FF", width=.55, fliersize=0, linewidth=.9, ax=ax)
        sns.stripplot(data=control, x="dataset", y="inference_correlation", order=study_order,
                      color="#1646FF", size=2.1, jitter=.22, alpha=.5, ax=ax)
        ax.scatter(np.arange(len(study_order)), point.loc[list(study_order), "observed_inference_correlation"],
                   color="#F13B35", s=38, zorder=5)
        ax.axhline(0, color="black", lw=.7)
        if len(study_order) > 8:
            ax.axvline(7.5, color="#666666", ls=(0, (4, 3)), lw=.9)
        ax.set(title=method, xlabel="", ylabel="Inference rho")
        ax.grid(False)
    axes.flat[-1].tick_params(axis="x", rotation=55)
    fig.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="#F13B35", label="observed"),
                        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1646FF", label="WGS-abundance controls")],
               loc="upper right", bbox_to_anchor=(.98, .98), frameon=False)
    fig.suptitle("Sun et al.-derived inference correlation: observed versus WGS-abundance controls", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .975))
    return fig, axes


def plot_directional_metric_summary(curves: pd.DataFrame, metric: str, methods: Sequence[str], *, colors=None):
    """Plot positive and negative AP or ROC-AUC summaries."""
    if metric not in {"ap", "roc_auc"}:
        raise ValueError("metric must be 'ap' or 'roc_auc'")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, direction in zip(axes, ["negative", "positive"]):
        block = curves[curves.direction.eq(direction)]
        sns.barplot(data=block, x="dataset", y=metric, hue="prediction_method",
                    hue_order=methods, palette=_palette(methods, colors), ax=ax)
        if metric == "ap":
            ax.scatter(np.arange(block.dataset.nunique()), block.groupby("dataset").baseline_target_prevalence.first(),
                       marker="_", color="black", label="WGS prevalence")
        else:
            ax.axhline(.5, color="black", ls="--", lw=.8)
        ax.set(title=f"{direction.title()} direction", xlabel="",
               ylabel="Average precision" if metric == "ap" else "ROC-AUC")
        ax.tick_params(axis="x", rotation=65)
        if ax is axes[1] and ax.legend_:
            ax.legend_.remove()
    axes[0].legend(title="", fontsize=8)
    fig.suptitle("Directional precision-recall summary" if metric == "ap" else "Directional ROC summary")
    fig.tight_layout()
    return fig, axes


def plot_directional_roc_grid(
    roc_points: pd.DataFrame,
    summaries: pd.DataFrame,
    study_order: Sequence[str],
    methods: Sequence[str],
    *,
    display_names: Mapping[str, str] | None = None,
    colors=None,
    benchmark_count: int = 8,
):
    """Full two-direction ROC grid with AUC values in panel legends."""
    palette = _palette(methods, colors)
    display_names = display_names or {}
    fig, axes = plt.subplots(2, len(study_order), figsize=(2.7 * len(study_order), 6.8), sharex=True, sharey=True)
    for row, direction in enumerate(["negative", "positive"]):
        for col, dataset in enumerate(study_order):
            ax = axes[row, col]
            for method in methods:
                points = roc_points[(roc_points.dataset.eq(dataset)) &
                                    (roc_points.prediction_method.eq(method)) &
                                    (roc_points.direction.eq(direction))]
                auc = summaries[(summaries.dataset.eq(dataset)) &
                                (summaries.prediction_method.eq(method)) &
                                (summaries.direction.eq(direction))].roc_auc.iloc[0]
                ax.plot(points.fpr, points.tpr, color=palette[method], lw=1.5, label=f"{method} (AUC={auc:.2f})")
            ax.plot([0, 1], [0, 1], color="#777777", ls=(0, (3, 2)), lw=.8)
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
            ax.grid(False)
            if row == 0:
                ax.set_title(display_names.get(dataset, dataset), fontsize=9)
            if row == 1:
                ax.set_xlabel("FPR", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{direction.title()} target\nTPR")
            ax.tick_params(labelsize=7)
            ax.legend(loc="lower right", fontsize=5.7, frameon=False, handlelength=1.8)
    fig.tight_layout(rect=(.02, .05, 1, .84), w_pad=.6)
    fig.canvas.draw()
    if 0 < benchmark_count < len(study_order):
        separator_x = (axes[0, benchmark_count - 1].get_position().x1 + axes[0, benchmark_count].get_position().x0) / 2
        fig.add_artist(Line2D([separator_x, separator_x], [.06, .87], transform=fig.transFigure,
                              color="#555555", ls=(0, (4, 3)), lw=1))
        benchmark_center = (axes[0, 0].get_position().x0 + axes[0, benchmark_count - 1].get_position().x1) / 2
        expansion_center = (axes[0, benchmark_count].get_position().x0 + axes[0, -1].get_position().x1) / 2
        fig.text(benchmark_center, .895, f"Benchmark-source studies (n={benchmark_count})", ha="center", weight="bold")
        fig.text(expansion_center, .895, f"Expansion studies (n={len(study_order) - benchmark_count})", ha="center", weight="bold")
    fig.suptitle("Directional ROC curves for all accepted contrasts (AUC shown per method)", y=.98)
    return fig, axes


def plot_signed_decile_grid(
    summaries: pd.DataFrame,
    baselines: pd.DataFrame,
    cohort: pd.DataFrame,
    study_order: Sequence[str],
    methods: Sequence[str],
    *,
    display_names: Mapping[str, str] | None = None,
    colors=None,
    benchmark_count: int = 8,
):
    """Signed positive/negative WGS outcome shares by predicted-effect decile."""
    palette = _palette(methods, colors)
    display_names = display_names or {}
    cohort_lookup = cohort.set_index("dataset")
    baseline_lookup = baselines.set_index("dataset")
    fig, axes = plt.subplots(len(methods), len(study_order), figsize=(2.7 * len(study_order), 4.1 * len(methods)),
                             sharex=True, sharey=True, squeeze=False)
    for row, method in enumerate(methods):
        for col, dataset in enumerate(study_order):
            ax = axes[row, col]
            block = summaries[(summaries.dataset.eq(dataset)) &
                              (summaries.prediction_method.eq(method))].sort_values("bin")
            baseline = baseline_lookup.loc[dataset]
            ax.bar(block.bin, block.p_sig_positive, color=POSITIVE_COLOR, width=.78)
            ax.bar(block.bin, -block.p_sig_negative, color=NEGATIVE_COLOR, width=.78)
            ax.axhline(baseline.p_sig_positive, color=POSITIVE_COLOR, ls=(0, (3, 2)), lw=.9, alpha=.75)
            ax.axhline(-baseline.p_sig_negative, color=NEGATIVE_COLOR, ls=(0, (3, 2)), lw=.9, alpha=.75)
            ax.axhline(0, color="#303438", lw=.65)
            ax.set(xlim=(.4, 10.6), ylim=(-1, 1), xticks=[1, 5, 10], yticks=[-1, -.5, 0, .5, 1])
            ax.grid(False)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            if row == 0:
                info = cohort_lookup.loc[dataset]
                ax.set_title(f"{display_names.get(dataset, dataset)}\nA: {info.group_a}\nB: {info.group_b}\n"
                             f"baseline P(sig)={baseline.p_any_significant:.2f}", fontsize=7.2)
            if row == len(methods) - 1:
                ax.set_xlabel("prediction decile", fontsize=8)
            if col == 0:
                ax.set_ylabel(method, color=palette[method], weight="bold", fontsize=9)
    legend_handles = [
        Patch(facecolor=POSITIVE_COLOR, label="WGS significant-positive share — bars up"),
        Patch(facecolor=NEGATIVE_COLOR, label="WGS significant-negative share — bars down"),
        Patch(facecolor="#D6DADD", label="nonsignificant — remainder"),
        Line2D([0], [0], color="#555555", ls=(0, (3, 2)), lw=1,
               label="whole-study positive/negative baseline shares"),
    ]
    fig.tight_layout(rect=(.02, .06, 1, .80), w_pad=.7, h_pad=1.0)
    fig.canvas.draw()
    if 0 < benchmark_count < len(study_order):
        separator_x = (axes[0, benchmark_count - 1].get_position().x1 + axes[0, benchmark_count].get_position().x0) / 2
        fig.add_artist(Line2D([separator_x, separator_x], [.06, .84], transform=fig.transFigure,
                              color="#555555", ls=(0, (4, 3)), lw=1))
        left_center = (axes[0, 0].get_position().x0 + axes[0, benchmark_count - 1].get_position().x1) / 2
        right_center = (axes[0, benchmark_count].get_position().x0 + axes[0, -1].get_position().x1) / 2
        fig.text(left_center, .845, f"Benchmark-source studies (n={benchmark_count})", ha="center", weight="bold")
        fig.text(right_center, .845, f"Expansion studies (n={len(study_order) - benchmark_count})", ha="center", weight="bold")
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(.5, .915),
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Prediction-ranked decile composition across all accepted contrasts "
                 "(D1 = lowest predicted effect; D10 = highest; fixed y-axis [-1, 1])", y=.985)
    return fig, axes
