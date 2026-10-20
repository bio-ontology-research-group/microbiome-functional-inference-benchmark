#!/usr/bin/env python3
"""Build the environment-oriented summary of benchmark and expansion results.

This script only reads the repository's frozen reference tables, compact study
bundles, and the completed expansion outputs. It trains no model and does not
modify either source result tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REFERENCE = REPO / "data" / "reference"
STUDIES = REPO / "data" / "studies"
EXPANSION = REPO / "expansion_benchmark"
INPUTS = HERE / "inputs"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
SOURCE_ROOT = Path(
    os.environ.get(
        "PICRUST_BENCHMARK_SOURCE_ROOT",
        "/home/hernan_melmoth/Documents/phd_work/functions_inference_benchmark_EQ",
    )
)
SOURCE_ENVIRONMENT = SOURCE_ROOT / (
    "functional_inference_generalization/results/"
    "otu_taxa_independent_empo_token_strict_grouped_loso_r1/preflight/"
    "annotations_r1/strict_621_source_only_empo_annotations.tsv"
)
METHODS = ["PICRUSt2", "Tax4Fun2"]
COLORS = {"PICRUSt2": "#d55e00", "Tax4Fun2": "#0072b2"}
BOOTSTRAP_SEED = 20261020


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, compression="gzip" if path.suffix == ".gz" else None)


def bootstrap_mean(values: pd.Series, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    array = values.dropna().to_numpy(float)
    if not len(array):
        return np.nan, np.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for index in range(n_boot):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def deterministic_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def prepare_benchmark_environment_input() -> pd.DataFrame:
    """Freeze only the lightweight sample/environment fields needed here."""
    INPUTS.mkdir(parents=True, exist_ok=True)
    target = INPUTS / "benchmark_source_sample_environment.tsv.gz"
    if not target.exists():
        if not SOURCE_ENVIRONMENT.exists():
            raise FileNotFoundError(
                f"Missing {target} and source annotation {SOURCE_ENVIRONMENT}. "
                "Regenerate the lightweight input on a machine with the source workspace."
            )
        source = pd.read_csv(SOURCE_ENVIRONMENT, sep="\t")
        keep = [
            "study_id", "dataset", "sample_id", "annotation_scope", "empo_1", "empo_2", "empo_3",
            "source_metadata", "mapping_status", "mapping_note", "evidence_sha256", "vocabulary_sha256",
        ]
        missing = sorted(set(keep) - set(source.columns))
        if missing:
            raise ValueError(f"source environment annotation lacks {missing}")
        write_table(source[keep].copy(), target)
    result = pd.read_csv(target, sep="\t")
    if len(result) != 621 or result[["dataset", "sample_id"]].duplicated().any():
        raise ValueError("benchmark environment registry must contain 621 unique dataset/sample pairs")
    return result


def expansion_environment(status: pd.DataFrame) -> pd.DataFrame:
    def hierarchy(value: str) -> tuple[str, str]:
        low = value.lower()
        if low.startswith("animal"):
            return "Host-associated", "Animal"
        if low.startswith("plant"):
            return "Host-associated", "Plant"
        if low.startswith("fungus"):
            return "Host-associated", "Fungus"
        if "non-saline" in low:
            return "Free-living", "Non-saline"
        if "saline" in low or "hypersaline" in low:
            return "Free-living", "Saline"
        if "pending" in low:
            return "Engineered", "Not assigned"
        return "Free-living", "Non-saline"

    rows = []
    for item in status.itertuples(index=False):
        empo_1, empo_2 = hierarchy(str(item.empo_level_3))
        rows.append(
            {
                "dataset": item.dataset,
                "source_project_id": f"expansion_{item.accession}",
                "empo_1": empo_1,
                "empo_2": empo_2,
                "empo_3": item.empo_level_3,
                "environment_mapping_status": "frozen_expansion_registry",
            }
        )
    return pd.DataFrame(rows)


def build_profiles(benchmark_env: pd.DataFrame) -> dict[str, pd.DataFrame]:
    strata = pd.read_csv(REFERENCE / "study_strata.tsv", sep="\t")
    base_ids = set(strata.loc[~strata.is_expansion_group, "dataset"])
    pic = pd.read_csv(REFERENCE / "sample_spearman_long.tsv.gz", sep="\t")
    pic = pic.loc[pic.dataset.isin(base_ids) & pic.method_label.eq("PICRUSt2")].copy()
    pic["status"] = np.where(pic.sample_spearman.notna(), "defined", "undefined")
    pic = pic[["dataset", "sample_id", "method_label", "sample_spearman", "status", "display_name"]]
    tax = pd.read_csv(REFERENCE / "tax4fun2_sample_spearman.tsv.gz", sep="\t")
    tax = tax.loc[tax.dataset.isin(base_ids)].copy()
    tax["status"] = tax.spearman_status
    tax = tax[["dataset", "sample_id", "method_label", "sample_spearman", "status", "display_name"]]
    base = pd.concat([pic, tax], ignore_index=True)
    base["provenance"] = "benchmark_source"

    env = benchmark_env.copy()
    env["source_project_id"] = "benchmark_" + env.study_id.astype(str)
    env["environment_mapping_status"] = env.mapping_status
    env = env[["dataset", "sample_id", "source_project_id", "empo_1", "empo_2", "empo_3", "environment_mapping_status"]]
    base = base.merge(env, on=["dataset", "sample_id"], how="left", validate="many_to_one")
    if base.empo_3.isna().any():
        raise ValueError("benchmark profile rows failed environment mapping")

    exp = pd.read_csv(EXPANSION / "results/profile_sample_spearman_wgs_universe.tsv.gz", sep="\t")
    exp = exp.rename(columns={"spearman_status": "status"})
    exp = exp[["dataset", "sample_id", "method_label", "sample_spearman", "status", "display_name"]]
    status = pd.read_csv(EXPANSION / "results/study_status.tsv", sep="\t")
    exp = exp.merge(expansion_environment(status), on="dataset", how="left", validate="many_to_one")
    exp["provenance"] = "expansion"

    combined = pd.concat([base, exp], ignore_index=True)
    combined = combined.loc[combined.method_label.isin(METHODS)].copy()
    combined["sample_id"] = combined.sample_id.astype(str)
    combined["sample_key"] = combined.provenance + "::" + combined.dataset + "::" + combined.sample_id
    combined["environment_scope_id"] = combined.source_project_id + "::" + combined.empo_3.astype(str)
    combined["method_label"] = pd.Categorical(combined.method_label, METHODS, ordered=True)
    combined = combined.sort_values(["provenance", "source_project_id", "empo_3", "sample_id", "method_label"])
    combined["method_label"] = combined.method_label.astype(str)

    if len(combined) != 3434:
        raise ValueError(f"expected 3434 method/sample rows; observed {len(combined)}")
    unique = combined.drop_duplicates("sample_key")
    if len(unique) != 1717 or unique.source_project_id.nunique() != 24 or unique.environment_scope_id.nunique() != 29:
        raise ValueError("combined census does not match 1717 samples / 24 projects / 29 scopes")
    if combined.groupby("sample_key").method_label.nunique().ne(2).any():
        raise ValueError("every sample must retain both methods, including explicit undefined values")

    sample_registry = unique[
        ["sample_key", "provenance", "source_project_id", "dataset", "sample_id", "display_name",
         "empo_1", "empo_2", "empo_3", "environment_scope_id", "environment_mapping_status"]
    ].sort_values(["provenance", "source_project_id", "empo_3", "sample_id"])

    scope_rows = []
    for (scope, method), group in combined.groupby(["environment_scope_id", "method_label"], sort=True):
        defined = group.sample_spearman.dropna()
        lo, hi = bootstrap_mean(defined, BOOTSTRAP_SEED + deterministic_seed(scope + method))
        first = group.iloc[0]
        scope_rows.append(
            {
                "environment_scope_id": scope,
                "source_project_id": first.source_project_id,
                "provenance": first.provenance,
                "display_name": first.display_name,
                "empo_1": first.empo_1,
                "empo_2": first.empo_2,
                "empo_3": first.empo_3,
                "method_label": method,
                "n_samples": group.sample_key.nunique(),
                "n_defined": int(defined.size),
                "n_undefined": int(group.sample_spearman.isna().sum()),
                "mean_sample_spearman": defined.mean(),
                "median_sample_spearman": defined.median(),
                "sd_sample_spearman": defined.std(ddof=1),
                "mean_bootstrap_ci_low": lo,
                "mean_bootstrap_ci_high": hi,
                "aggregation_unit": "source_project_x_environment_scope",
            }
        )
    scope_summary = pd.DataFrame(scope_rows)

    wide = combined.pivot(index="sample_key", columns="method_label", values="sample_spearman")
    metadata = sample_registry.set_index("sample_key")
    delta = metadata.join(wide)
    delta["tax4fun2_minus_picrust2"] = delta["Tax4Fun2"] - delta["PICRUSt2"]
    delta_rows = []
    for scope, group in delta.groupby("environment_scope_id", sort=True):
        values = group.tax4fun2_minus_picrust2.dropna()
        lo, hi = bootstrap_mean(values, BOOTSTRAP_SEED + deterministic_seed(scope + "delta"))
        first = group.iloc[0]
        delta_rows.append(
            {
                "environment_scope_id": scope,
                "source_project_id": first.source_project_id,
                "provenance": first.provenance,
                "display_name": first.display_name,
                "empo_1": first.empo_1,
                "empo_2": first.empo_2,
                "empo_3": first.empo_3,
                "n_paired_defined": len(values),
                "mean_tax4fun2_minus_picrust2": values.mean(),
                "median_tax4fun2_minus_picrust2": values.median(),
                "mean_bootstrap_ci_low": lo,
                "mean_bootstrap_ci_high": hi,
            }
        )
    difference = pd.DataFrame(delta_rows)

    environment_rows = []
    for (level1, level2, method), group in scope_summary.groupby(["empo_1", "empo_2", "method_label"], sort=True):
        values = group.mean_sample_spearman.dropna()
        environment_rows.append(
            {
                "empo_1": level1,
                "empo_2": level2,
                "method_label": method,
                "n_source_project_environment_scopes": len(values),
                "n_samples": group.n_samples.sum(),
                "median_scope_mean_spearman": values.median(),
                "q1_scope_mean_spearman": values.quantile(0.25),
                "q3_scope_mean_spearman": values.quantile(0.75),
                "mean_scope_mean_spearman": values.mean(),
                "aggregation_policy": "equal_weight_per_source_project_x_environment_scope",
            }
        )
    environment = pd.DataFrame(environment_rows)
    return {
        "sample": combined,
        "registry": sample_registry,
        "scope": scope_summary,
        "difference": difference,
        "environment": environment,
    }


def group_environment_lookup(sample_registry: pd.DataFrame) -> pd.DataFrame:
    def collapse(series: pd.Series) -> str:
        return "; ".join(sorted(series.astype(str).unique()))
    return sample_registry.groupby("dataset", as_index=False).agg(
        source_project_id=("source_project_id", collapse),
        empo_1=("empo_1", collapse),
        empo_2=("empo_2", collapse),
        empo_3=("empo_3", collapse),
    )


def build_group_results(sample_registry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sys.path.insert(0, str(REPO / "src"))
    from picrust_benchmark_metrics import (  # pylint: disable=import-outside-toplevel
        directional_ranking_curves,
        load_processed_study,
        matchado_bh_shared_wgs_reference_metrics,
        normalize_rows,
        rank_biserial_effect_by_ko,
        sun_inference_correlation_abundance_randomization,
    )
    from scipy.stats import pearsonr, spearmanr  # pylint: disable=import-outside-toplevel

    environment = group_environment_lookup(sample_registry)
    cohort = pd.read_csv(REFERENCE / "effect_cohort.tsv", sep="\t")
    base_ids = set(sample_registry.loc[sample_registry.provenance.eq("benchmark_source"), "dataset"])
    cohort = cohort.loc[cohort.dataset.isin(base_ids)].copy()
    cohort["contrast_id"] = "benchmark_" + cohort.dataset
    cohort["design"] = "unpaired"
    cohort["status"] = "accepted"
    cohort["provenance"] = "benchmark_source"
    cohort = cohort.merge(environment, on="dataset", how="left", validate="many_to_one")

    exp_contrast = pd.read_csv(EXPANSION / "results/contrast_manifest.tsv", sep="\t")
    exp_contrast["provenance"] = "expansion"
    exp_contrast = exp_contrast.merge(environment, on="dataset", how="left", validate="many_to_one")
    contrast_cols = [
        "contrast_id", "dataset", "design", "group_a", "group_b", "status", "n_group_a", "n_group_b",
        "n_pairs", "provenance", "source_project_id", "empo_1", "empo_2", "empo_3",
    ]
    for column in contrast_cols:
        if column not in cohort:
            cohort[column] = np.nan
        if column not in exp_contrast:
            exp_contrast[column] = np.nan
    contrasts = pd.concat([cohort[contrast_cols], exp_contrast[contrast_cols]], ignore_index=True)
    if len(contrasts) != 17:
        raise ValueError(f"expected 17 accepted biological contrasts, found {len(contrasts)}")

    # The original directional/effect table did not include Tax4Fun2. Re-run both
    # methods on its exact frozen KO family and WGS BH/effect vectors. PICRUSt2 is
    # later validated against the completed reference to machine precision.
    quantiles = pd.read_csv(REFERENCE / "quantile_per_ko_assignments.tsv.gz", sep="\t")
    base_effect_rows, base_directional_rows = [], []
    for item in cohort.itertuples(index=False):
        frozen = quantiles.loc[
            quantiles.dataset.eq(item.dataset) & quantiles.prediction_method.eq("PICRUSt2"),
            ["KO", "wgs_bh_qvalue", "wgs_rank_biserial_effect"],
        ].drop_duplicates("KO").sort_values("KO")
        truth, predictions, labels, _ = load_processed_study(STUDIES / f"{item.dataset}.npz")
        selected = frozen.wgs_bh_qvalue.to_numpy(float) < 0.05
        wgs_effect = frozen.wgs_rank_biserial_effect.to_numpy(float)
        for method in METHODS:
            relative, totals = normalize_rows(predictions[method])
            if (totals <= 0).any():
                raise ValueError(f"{item.dataset}/{method} has a zero-total sample")
            aligned = relative.reindex(columns=frozen.KO, fill_value=0.0).fillna(0.0)
            prediction_effect = rank_biserial_effect_by_ko(
                aligned, labels, item.group_a, item.group_b
            ).set_index("KO").loc[frozen.KO, "rank_biserial_effect"].to_numpy(float)
            spearman_all = float(spearmanr(wgs_effect, prediction_effect).statistic)
            pearson_all = float(pearsonr(wgs_effect, prediction_effect).statistic)
            spearman_selected = float(spearmanr(wgs_effect[selected], prediction_effect[selected]).statistic)
            pearson_selected = float(pearsonr(wgs_effect[selected], prediction_effect[selected]).statistic)
            base_effect_rows.append(
                {
                    "contrast_id": item.contrast_id,
                    "dataset": item.dataset,
                    "design": "unpaired",
                    "prediction_method": method,
                    "n_kos": len(frozen),
                    "n_wgs_bh_selected": int(selected.sum()),
                    "spearman_effect_all": spearman_all,
                    "pearson_effect_all": pearson_all,
                    "spearman_effect_wgs_bh_selected": spearman_selected,
                    "pearson_effect_wgs_bh_selected": pearson_selected,
                    "provenance": "benchmark_source",
                    "metric_surface": "frozen_original_wgs_bh_and_eligible_ko_surface",
                }
            )
            _, _, summary = directional_ranking_curves(
                frozen.wgs_bh_qvalue, wgs_effect, prediction_effect
            )
            summary.insert(0, "prediction_method", method)
            summary.insert(0, "dataset", item.dataset)
            summary.insert(0, "contrast_id", item.contrast_id)
            summary["provenance"] = "benchmark_source"
            base_directional_rows.append(summary)
    base_effect = pd.DataFrame(base_effect_rows)
    exp_effect = pd.read_csv(EXPANSION / "results/effect_summary.tsv", sep="\t")
    exp_effect["provenance"] = "expansion"
    exp_effect["metric_surface"] = "wgs_bh_selected_and_all_kos"
    effect = pd.concat([base_effect, exp_effect], ignore_index=True, sort=False)

    base_directional = pd.concat(base_directional_rows, ignore_index=True)
    exp_directional = pd.read_csv(EXPANSION / "results/directional_metric_summary.tsv", sep="\t")
    exp_directional = exp_directional.rename(columns={"dataset": "contrast_id"})
    exp_directional = exp_directional.merge(
        exp_contrast[["contrast_id", "dataset"]], on="contrast_id", how="left", validate="many_to_one"
    )
    exp_directional["provenance"] = "expansion"
    directional = pd.concat([base_directional, exp_directional], ignore_index=True, sort=False)

    base_map = dict(zip(cohort.dataset, cohort.contrast_id))
    base_map["brooks_reverse_only"] = "benchmark_brooks_forward"
    base_matchado = pd.read_csv(REFERENCE / "matchado_label_metric_results_long.tsv", sep="\t")
    base_matchado = base_matchado.loc[base_matchado.prediction_method.isin(METHODS)].copy()
    base_matchado["contrast_id"] = base_matchado.dataset.map(base_map)
    base_matchado = base_matchado.loc[base_matchado.contrast_id.notna()].copy()
    base_matchado["analysis_stratum"] = base_matchado.dataset
    base_matchado["provenance"] = "benchmark_source"
    base_matchado["multiple_testing_method"] = "none_unadjusted_pvalue"
    base_matchado["significance_rule"] = "pvalue < 0.05"
    base_matchado["metric_provenance"] = np.where(
        base_matchado.metric.isin(["precision", "recall", "f1"]), "published_matchado", "project_ranking_extension"
    )
    exp_matchado = pd.read_csv(EXPANSION / "results/matchado_raw_p_summary.tsv", sep="\t")
    exp_matchado = exp_matchado.rename(columns={"dataset": "contrast_id"})
    exp_matchado = exp_matchado.merge(
        exp_contrast[["contrast_id", "dataset"]], on="contrast_id", how="left", validate="many_to_one"
    )
    exp_matchado["analysis_stratum"] = exp_matchado.dataset
    exp_matchado["provenance"] = "expansion"
    matchado = pd.concat([base_matchado, exp_matchado], ignore_index=True, sort=False)

    # Re-run project BH and Sun metrics for the eight benchmark biological contrasts.
    bh_rows, sun_rows, random_rows = [], [], []
    for item in cohort.itertuples(index=False):
        path = STUDIES / f"{item.dataset}.npz"
        truth, predictions, labels, _ = load_processed_study(path)
        selected = {method: predictions[method] for method in METHODS}
        bh, _ = matchado_bh_shared_wgs_reference_metrics(
            truth, selected, labels, group_a=item.group_a, group_b=item.group_b
        )
        bh.insert(0, "dataset", item.dataset)
        bh.insert(0, "contrast_id", item.contrast_id)
        bh["provenance"] = "benchmark_source"
        bh_rows.append(bh)
        sun, random, _ = sun_inference_correlation_abundance_randomization(
            truth, selected, labels, group_a=item.group_a, group_b=item.group_b,
            n_randomizations=100, random_seed=20260901,
        )
        sun.insert(0, "dataset", item.dataset)
        sun.insert(0, "contrast_id", item.contrast_id)
        sun["provenance"] = "benchmark_source"
        random.insert(0, "dataset", item.dataset)
        random.insert(0, "contrast_id", item.contrast_id)
        random["provenance"] = "benchmark_source"
        sun_rows.append(sun)
        random_rows.append(random)

    base_bh = pd.concat(bh_rows, ignore_index=True)
    exp_bh = pd.read_csv(EXPANSION / "results/project_bh_summary.tsv", sep="\t")
    exp_bh = exp_bh.rename(columns={"dataset": "contrast_id"}).merge(
        exp_contrast[["contrast_id", "dataset"]], on="contrast_id", how="left", validate="many_to_one"
    )
    exp_bh["provenance"] = "expansion"
    bh = pd.concat([base_bh, exp_bh], ignore_index=True, sort=False)

    base_sun = pd.concat(sun_rows, ignore_index=True)
    exp_sun = pd.read_csv(EXPANSION / "results/sun_summary.tsv", sep="\t")
    exp_sun = exp_sun.rename(columns={"dataset": "contrast_id"}).merge(
        exp_contrast[["contrast_id", "dataset"]], on="contrast_id", how="left", validate="many_to_one"
    )
    exp_sun["provenance"] = "expansion"
    sun = pd.concat([base_sun, exp_sun], ignore_index=True, sort=False)

    base_random = pd.concat(random_rows, ignore_index=True)
    exp_random = pd.read_csv(EXPANSION / "results/sun_randomizations.tsv", sep="\t")
    exp_random = exp_random.rename(columns={"dataset": "contrast_id"}).merge(
        exp_contrast[["contrast_id", "dataset"]], on="contrast_id", how="left", validate="many_to_one"
    )
    exp_random["provenance"] = "expansion"
    randomizations = pd.concat([base_random, exp_random], ignore_index=True, sort=False)
    return {
        "contrast_registry": contrasts,
        "effect": effect,
        "directional": directional,
        "matchado": matchado,
        "bh": bh,
        "sun": sun,
        "sun_randomizations": randomizations,
    }


def add_panel_label(axis: plt.Axes, text: str) -> None:
    axis.text(-0.08, 1.03, text, transform=axis.transAxes, fontsize=11, fontweight="bold", va="bottom")


def save_coverage(registry: pd.DataFrame) -> None:
    counts = registry.groupby(["empo_3", "provenance"]).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.34 * len(counts))))
    left = np.zeros(len(counts))
    for provenance, color in [("benchmark_source", "#666666"), ("expansion", "#56b4e9")]:
        values = counts.get(provenance, pd.Series(0, index=counts.index)).to_numpy()
        ax.barh(counts.index, values, left=left, color=color, label=provenance.replace("_", " "))
        left += values
    ax.set_xlabel("paired samples")
    ax.set_title("Environment coverage: benchmark sources plus completed expansions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_environment_coverage_census.png", dpi=180)
    plt.close(fig)


def scope_order(scope: pd.DataFrame) -> list[str]:
    pic = scope.loc[scope.method_label.eq("PICRUSt2")].sort_values(
        ["empo_1", "empo_2", "mean_sample_spearman", "environment_scope_id"]
    )
    return pic.environment_scope_id.tolist()


def scope_labels(scope: pd.DataFrame, order: list[str]) -> list[str]:
    lookup = scope.drop_duplicates("environment_scope_id").set_index("environment_scope_id")
    return [
        f"[{lookup.loc[key, 'provenance'][0].upper()}] {lookup.loc[key, 'empo_3']} | "
        f"{lookup.loc[key, 'display_name']}"
        for key in order
    ]


def save_scope_dumbbell(scope: pd.DataFrame) -> None:
    order = scope_order(scope)
    wide = scope.pivot(index="environment_scope_id", columns="method_label", values="mean_sample_spearman").loc[order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.hlines(y, wide.PICRUSt2, wide.Tax4Fun2, color="#bdbdbd", linewidth=1.2)
    for method in METHODS:
        ax.scatter(wide[method], y, s=38, color=COLORS[method], label=method, zorder=3)
    ax.set_yticks(y, scope_labels(scope, order), fontsize=7)
    ax.set_xlabel("mean sample-wise Spearman correlation on each WGS KO universe")
    ax.set_title("Profile agreement by source-project × environment scope")
    ax.grid(axis="x", color="#eeeeee")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_scope_profile_dumbbell.png", dpi=180)
    plt.close(fig)


def save_scope_difference(difference: pd.DataFrame, scope: pd.DataFrame) -> None:
    order = scope_order(scope)
    data = difference.set_index("environment_scope_id").loc[order]
    y = np.arange(len(order))
    x = data.mean_tax4fun2_minus_picrust2.to_numpy(float)
    lo = data.mean_bootstrap_ci_low.to_numpy(float)
    hi = data.mean_bootstrap_ci_high.to_numpy(float)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt="o", color="#4d4d4d", ecolor="#aaaaaa", capsize=2)
    ax.set_yticks(y, scope_labels(scope, order), fontsize=7)
    ax.set_xlabel("mean paired difference (Tax4Fun2 − PICRUSt2); 95% sample-bootstrap interval")
    ax.set_title("Within-sample method difference by environment scope")
    ax.grid(axis="x", color="#eeeeee")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_scope_method_difference.png", dpi=180)
    plt.close(fig)


def save_environment_macro(environment: pd.DataFrame) -> None:
    environment = environment.copy()
    environment["label"] = environment.empo_1 + " / " + environment.empo_2
    labels = sorted(environment.label.unique())
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.65 * len(labels))))
    y = np.arange(len(labels))
    offsets = {"PICRUSt2": -0.11, "Tax4Fun2": 0.11}
    for method in METHODS:
        block = environment.loc[environment.method_label.eq(method)].set_index("label").reindex(labels)
        value = block.median_scope_mean_spearman.to_numpy(float)
        low = block.q1_scope_mean_spearman.to_numpy(float)
        high = block.q3_scope_mean_spearman.to_numpy(float)
        ax.errorbar(value, y + offsets[method], xerr=np.vstack([value - low, high - value]), fmt="o", color=COLORS[method], label=method, capsize=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("median [IQR] of source-project × environment mean Spearman")
    ax.set_title("Environment-level macro summary (equal weight per scope)")
    ax.grid(axis="x", color="#eeeeee")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_environment_macro_profile.png", dpi=180)
    plt.close(fig)


def contrast_labels(registry: pd.DataFrame) -> dict[str, str]:
    repeated = set(registry.loc[registry.dataset.duplicated(keep=False), "dataset"])
    return {
        row.contrast_id: (
            f"[{row.provenance[0].upper()}] "
            f"{row.contrast_id if row.dataset in repeated else row.dataset}: {row.group_a} vs {row.group_b}"
        )
        for row in registry.itertuples(index=False)
    }


def save_matchado(matchado: pd.DataFrame, registry: pd.DataFrame) -> None:
    # Main biological view uses the primary Brooks forward stratum; reverse is retained in the table.
    data = matchado.loc[matchado.metric.eq("f1") & ~matchado.analysis_stratum.eq("brooks_reverse_only")].copy()
    labels = contrast_labels(registry)
    order = [value for value in registry.contrast_id if value in set(data.contrast_id)]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.48 * len(order))))
    y = np.arange(len(order))
    offsets = {"PICRUSt2": -0.1, "Tax4Fun2": 0.1}
    for method in METHODS:
        block = data.loc[data.prediction_method.eq(method)].drop_duplicates("contrast_id").set_index("contrast_id").reindex(order)
        ax.hlines(y + offsets[method], block.permutation_min, block.permutation_max, color=COLORS[method], alpha=0.35, linewidth=3)
        ax.scatter(block.permutation_mean, y + offsets[method], facecolors="white", edgecolors=COLORS[method], s=28)
        ax.scatter(block.estimate, y + offsets[method], color=COLORS[method], s=30, label=method)
    ax.set_yticks(y, [labels[item] for item in order], fontsize=7)
    ax.set_xlabel("F1 (filled: observed; open: permutation mean; line: permutation range)")
    ax.set_title("Matchado-style unadjusted-p recovery and phenotype-label control")
    ax.grid(axis="x", color="#eeeeee")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_matchado_raw_p_f1.png", dpi=180)
    plt.close(fig)


def save_sun(sun: pd.DataFrame, registry: pd.DataFrame) -> None:
    labels = contrast_labels(registry)
    order = [value for value in registry.contrast_id if value in set(sun.contrast_id)]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.48 * len(order))))
    y = np.arange(len(order))
    offsets = {"PICRUSt2": -0.1, "Tax4Fun2": 0.1}
    for method in METHODS:
        block = sun.loc[sun.prediction_method.eq(method)].drop_duplicates("contrast_id").set_index("contrast_id").reindex(order)
        ax.hlines(y + offsets[method], block.randomization_min, block.randomization_max, color=COLORS[method], alpha=0.35, linewidth=3)
        ax.scatter(block.randomization_mean, y + offsets[method], facecolors="white", edgecolors=COLORS[method], s=28)
        ax.scatter(block.observed_inference_correlation, y + offsets[method], color=COLORS[method], s=30, label=method)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [labels[item] for item in order], fontsize=7)
    ax.set_xlabel("inference correlation (filled: observed; open: WGS-abundance control mean; line: range)")
    ax.set_title("Sun-style observed inference correlation and WGS-abundance control")
    ax.grid(axis="x", color="#eeeeee")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_sun_inference_correlation.png", dpi=180)
    plt.close(fig)


def save_effect(effect: pd.DataFrame, registry: pd.DataFrame) -> None:
    labels = contrast_labels(registry)
    data = effect.drop_duplicates(["contrast_id", "prediction_method"]).copy()
    order = [value for value in registry.contrast_id if value in set(data.contrast_id)]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.48 * len(order))))
    y = np.arange(len(order))
    offsets = {"PICRUSt2": -0.1, "Tax4Fun2": 0.1}
    for method in METHODS:
        block = data.loc[data.prediction_method.eq(method)].set_index("contrast_id").reindex(order)
        ax.scatter(block.spearman_effect_wgs_bh_selected, y + offsets[method], color=COLORS[method], s=34, label=method)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [labels[item] for item in order], fontsize=7)
    ax.set_xlabel("Spearman correlation of WGS and predicted group-effect sizes (WGS BH-selected KOs)")
    ax.set_title("Project effect-size relation across accepted biological contrasts")
    ax.grid(axis="x", color="#eeeeee")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_effect_relationship.png", dpi=180)
    plt.close(fig)


def save_directional(directional: pd.DataFrame, registry: pd.DataFrame) -> None:
    labels = contrast_labels(registry)
    data = directional.groupby(["contrast_id", "prediction_method"], as_index=False).agg(
        mean_roc_auc=("roc_auc", "mean"), mean_ap=("ap", "mean")
    )
    order = [value for value in registry.contrast_id if value in set(data.contrast_id)]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, 0.48 * len(order))), sharey=True)
    y = np.arange(len(order))
    offsets = {"PICRUSt2": -0.1, "Tax4Fun2": 0.1}
    for panel, metric, xlabel in [
        (axes[0], "mean_roc_auc", "mean directional ROC AUC"),
        (axes[1], "mean_ap", "mean directional average precision"),
    ]:
        for method in METHODS:
            block = data.loc[data.prediction_method.eq(method)].set_index("contrast_id").reindex(order)
            panel.scatter(block[metric], y + offsets[method], color=COLORS[method], s=34, label=method)
        panel.set_xlabel(xlabel)
        panel.grid(axis="x", color="#eeeeee")
        panel.set_yticks(y, [labels[item] for item in order], fontsize=7)
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].legend(frameon=False, ncol=2)
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.suptitle("Directional recovery across accepted biological contrasts", y=1.002)
    fig.tight_layout()
    fig.savefig(FIGURES / "08_directional_roc_auc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def input_manifest(benchmark_input: Path) -> pd.DataFrame:
    files = [
        benchmark_input,
        REFERENCE / "study_strata.tsv",
        REFERENCE / "sample_spearman_long.tsv.gz",
        REFERENCE / "tax4fun2_sample_spearman.tsv.gz",
        REFERENCE / "effect_cohort.tsv",
        REFERENCE / "effect_long.tsv",
        REFERENCE / "pr_roc_metric_summary.tsv",
        REFERENCE / "matchado_label_metric_results_long.tsv",
        EXPANSION / "RUN_COMPLETE.json",
        *sorted((EXPANSION / "results").glob("*.tsv*")),
        *sorted(STUDIES.glob("*.npz")),
    ]
    return pd.DataFrame(
        {
            "path": [str(path.relative_to(REPO)) for path in files],
            "size_bytes": [path.stat().st_size for path in files],
            "sha256": [sha256(path) for path in files],
            "access": "read_only_input",
        }
    )


def output_manifest() -> pd.DataFrame:
    files = sorted(path for directory in (INPUTS, RESULTS, FIGURES) for path in directory.glob("*") if path.is_file())
    return pd.DataFrame(
        {
            "path": [str(path.relative_to(HERE)) for path in files],
            "size_bytes": [path.stat().st_size for path in files],
            "sha256": [sha256(path) for path in files],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="replace only combined_benchmark/results and figures")
    args = parser.parse_args()
    if (RESULTS.exists() or FIGURES.exists()) and not args.overwrite:
        raise SystemExit("outputs already exist; pass --overwrite to replace only this combined layer")
    for directory in (RESULTS, FIGURES):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    benchmark_environment = prepare_benchmark_environment_input()
    profiles = build_profiles(benchmark_environment)
    groups = build_group_results(profiles["registry"])

    write_table(profiles["sample"], RESULTS / "combined_profile_sample_long.tsv.gz")
    write_table(profiles["registry"], RESULTS / "combined_sample_environment_registry.tsv.gz")
    write_table(profiles["scope"], RESULTS / "scope_profile_summary.tsv")
    write_table(profiles["difference"], RESULTS / "scope_method_difference.tsv")
    write_table(profiles["environment"], RESULTS / "environment_profile_summary.tsv")
    write_table(groups["contrast_registry"], RESULTS / "combined_contrast_registry.tsv")
    write_table(groups["effect"], RESULTS / "combined_effect.tsv")
    write_table(groups["directional"], RESULTS / "combined_directional.tsv")
    write_table(groups["matchado"], RESULTS / "combined_matchado.tsv")
    write_table(groups["bh"], RESULTS / "combined_bh.tsv")
    write_table(groups["sun"], RESULTS / "combined_sun.tsv")
    write_table(groups["sun_randomizations"], RESULTS / "combined_sun_randomizations.tsv.gz")
    write_table(input_manifest(INPUTS / "benchmark_source_sample_environment.tsv.gz"), RESULTS / "input_manifest.tsv")

    save_coverage(profiles["registry"])
    save_scope_dumbbell(profiles["scope"])
    save_scope_difference(profiles["difference"], profiles["scope"])
    save_environment_macro(profiles["environment"])
    save_matchado(groups["matchado"], groups["contrast_registry"])
    save_sun(groups["sun"], groups["contrast_registry"])
    save_effect(groups["effect"], groups["contrast_registry"])
    save_directional(groups["directional"], groups["contrast_registry"])

    receipt = {
        "status": "PASS",
        "training_performed": False,
        "methods": METHODS,
        "paired_samples": int(profiles["registry"].sample_key.nunique()),
        "benchmark_source_samples": int(profiles["registry"].provenance.eq("benchmark_source").sum()),
        "expansion_samples": int(profiles["registry"].provenance.eq("expansion").sum()),
        "unique_source_projects": int(profiles["registry"].source_project_id.nunique()),
        "source_project_environment_scopes": int(profiles["registry"].environment_scope_id.nunique()),
        "accepted_biological_contrasts": int(len(groups["contrast_registry"])),
        "profile_aggregation_policy": "equal weight per source-project x environment scope for environment macro summaries",
        "blueberry_environment": "Unknown (contradictory), retained from frozen source annotation",
        "benchmark_metric_recomputation": "project BH and Sun metrics rerun from repository compact processed bundles",
        "source_trees_modified": False,
    }
    (HERE / "RUN_COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    write_table(output_manifest(), HERE / "OUTPUT_MANIFEST.tsv")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
