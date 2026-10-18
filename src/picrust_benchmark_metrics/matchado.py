"""Matchado-style WGS-referenced differential-abundance metrics.

This retains the legacy project's published-method implementation: unadjusted
two-sided Mann--Whitney tests, a shared 5% prevalence surface, WGS significant
KOs as the reference set, and prediction-label permutations as the control.
It is intentionally separate from the project's BH-selected effect extensions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.multitest import multipletests

from .profile import normalize_rows


def _labels(labels: pd.Series, sample_ids: Sequence[str], group_a: str, group_b: str) -> pd.Series:
    out = labels.copy()
    out.index = out.index.astype(str)
    out = out.reindex(pd.Index(sample_ids, dtype=str))
    if out.isna().any() or set(out.astype(str).unique()) != {group_a, group_b}:
        raise ValueError("labels must be complete and contain exactly the declared two groups")
    return out.astype(str)


def wilcoxon_rank_sum_by_ko(matrix: pd.DataFrame, labels: pd.Series, group_a: str, group_b: str) -> pd.DataFrame:
    """Two-sided Mann--Whitney U tests, one per KO, with medians retained."""
    numeric = matrix.astype(float)
    if not np.isfinite(numeric.to_numpy()).all() or numeric.index.duplicated().any() or numeric.columns.duplicated().any():
        raise ValueError("matrix must have unique IDs and finite values")
    numeric.index, numeric.columns = numeric.index.astype(str), numeric.columns.astype(str)
    fixed = _labels(labels, numeric.index, group_a, group_b)
    a, b = numeric.loc[fixed.eq(group_a)].to_numpy(), numeric.loc[fixed.eq(group_b)].to_numpy()
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each group needs at least two samples")
    result = mannwhitneyu(a, b, axis=0, alternative="two-sided", method="auto")
    return pd.DataFrame({"KO": numeric.columns, "wilcoxon_u": result.statistic, "pvalue": result.pvalue,
                         "median_group_a": np.median(a, axis=0), "median_group_b": np.median(b, axis=0)})


def classification_set_metrics(reference: set[str], predicted: set[str]) -> dict:
    """Legacy precision/recall/F1 with the project's empty-set convention."""
    reference, predicted = set(reference), set(predicted)
    tp, fp, fn = len(reference & predicted), len(predicted - reference), len(reference - predicted)
    precision = tp / (tp + fp) if tp + fp else (0.0 if reference else np.nan)
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else (0.0 if np.isfinite(precision) and np.isfinite(recall) else np.nan)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _ranking(y_true: np.ndarray, pvalues: np.ndarray) -> dict:
    score = -np.log10(np.clip(pvalues.astype(float), 1e-300, 1.0))
    pos = int(y_true.sum())
    neg = int((~y_true).sum())
    if pos == 0 or neg == 0:
        return {"auroc": np.nan, "aupr": np.nan}
    return {"auroc": float(roc_auc_score(y_true, score)), "aupr": float(average_precision_score(y_true, score))}


def matchado_differential_abundance_metrics(
    truth: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    labels: pd.Series,
    *,
    group_a: str,
    group_b: str,
    prevalence_threshold: float = 0.05,
    alpha: float = 0.05,
    n_permutations: int = 20,
    random_seed: int = 20260821,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Matchado-style observed-vs-label-permuted metrics and KO tests.

    The WGS test is fixed. For each prediction method, only the phenotype
    labels attached to that prediction are permuted for the requested rounds.
    """
    if not predictions or n_permutations < 1:
        raise ValueError("at least one prediction and one permutation are required")
    truth = truth.copy().astype(float); truth.index = truth.index.astype(str); truth.columns = truth.columns.astype(str)
    fixed = _labels(labels, truth.index, group_a, group_b)
    common = set(truth.columns)
    for matrix in predictions.values():
        if list(matrix.index.astype(str)) != list(truth.index):
            raise ValueError("prediction samples must be aligned to WGS")
        common &= set(matrix.columns.astype(str))
    kos = sorted(common)
    if not kos:
        raise ValueError("no shared KOs")
    matrices = {"WGS": truth, **{name: frame.copy().astype(float) for name, frame in predictions.items()}}
    relative: dict[str, pd.DataFrame] = {}
    eligible = []
    for name, matrix in matrices.items():
        matrix.index, matrix.columns = matrix.index.astype(str), matrix.columns.astype(str)
        normalized, totals = normalize_rows(matrix.loc[truth.index, kos])
        if (totals <= 0).any():
            raise ValueError(f"{name} has a zero-total sample")
        relative[name] = normalized.fillna(0.0)
        eligible.append(set(normalized.columns[(normalized > 0).mean(axis=0) >= prevalence_threshold]))
    tested = sorted(set.intersection(*eligible))
    if not tested:
        raise ValueError("no KO passes the shared prevalence filter")
    wgs_tests = wilcoxon_rank_sum_by_ko(relative["WGS"].loc[:, tested], fixed, group_a, group_b)
    reference = set(wgs_tests.loc[wgs_tests.pvalue < alpha, "KO"])
    reference_vector = wgs_tests.pvalue.to_numpy(float) < alpha
    rng, rows, ko_tables = np.random.default_rng(random_seed), [], []
    for method in predictions:
        pred_tests = wilcoxon_rank_sum_by_ko(relative[method].loc[:, tested], fixed, group_a, group_b)
        observed = classification_set_metrics(reference, set(pred_tests.loc[pred_tests.pvalue < alpha, "KO"]))
        observed.update(_ranking(reference_vector, pred_tests.pvalue.to_numpy(float)))
        permutations = []
        for repeat in range(1, n_permutations + 1):
            shuffled = pd.Series(rng.permutation(fixed.to_numpy()), index=fixed.index)
            perm = wilcoxon_rank_sum_by_ko(relative[method].loc[:, tested], shuffled, group_a, group_b)
            item = classification_set_metrics(reference, set(perm.loc[perm.pvalue < alpha, "KO"]))
            item.update(_ranking(reference_vector, perm.pvalue.to_numpy(float)))
            item["permutation"] = repeat
            permutations.append(item)
        null = pd.DataFrame(permutations)
        for metric in ("precision", "recall", "f1", "auroc", "aupr"):
            rows.append({"prediction_method": method, "metric": metric, "estimate": observed[metric],
                         "permutation_mean": null[metric].mean(), "permutation_sd": null[metric].std(ddof=1),
                         "permutation_min": null[metric].min(), "permutation_max": null[metric].max(),
                         "observed_minus_permutation_mean": observed[metric] - null[metric].mean(),
                         "n_permutations": n_permutations, "n_tested_kos": len(tested),
                         "n_wgs_significant": len(reference), "n_prediction_significant": int((pred_tests.pvalue < alpha).sum()),
                         "tp": observed["tp"], "fp": observed["fp"], "fn": observed["fn"],
                         "randomization_method": "matchado_2024_prediction_phenotype_label_permutation"})
        merged = wgs_tests.merge(pred_tests, on="KO", suffixes=("_wgs", "_prediction"), validate="one_to_one")
        merged.insert(0, "prediction_method", method)
        ko_tables.append(merged)
    return pd.DataFrame(rows), pd.concat(ko_tables, ignore_index=True)


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """BH-adjust a complete hypothesis family without dropping tests."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues must be a finite one-dimensional vector in [0, 1]")
    return multipletests(values, method="fdr_bh")[1] if len(values) else values.copy()


def r_default_wilcoxon_by_ko(
    matrix: pd.DataFrame, labels: pd.Series, group_a: str, group_b: str
) -> pd.DataFrame:
    """R-default-compatible unpaired Wilcoxon tests with explicit degeneracy.

    R selects an exact test only when both groups have fewer than 50 samples
    and the pooled values contain no ties. Otherwise this uses SciPy's
    tie-corrected normal approximation with continuity correction. Columns
    whose pooled values are all identical receive the frozen project
    convention ``p = 1``.
    """
    numeric = matrix.astype(float).copy()
    numeric.index, numeric.columns = numeric.index.astype(str), numeric.columns.astype(str)
    if numeric.index.duplicated().any() or numeric.columns.duplicated().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("matrix must have unique identifiers and finite values")
    fixed = _labels(labels, numeric.index, group_a, group_b)
    values = numeric.to_numpy(float)
    is_a = fixed.eq(group_a).to_numpy()
    x, y = values[is_a], values[~is_a]
    if not len(x) or not len(y):
        raise ValueError("both groups must contain at least one sample")
    ordered = np.sort(values, axis=0)
    ties = (np.diff(ordered, axis=0) == 0).any(axis=0)
    identical = values.max(axis=0) == values.min(axis=0)
    use_exact = ~ties if len(x) < 50 and len(y) < 50 else np.zeros(values.shape[1], dtype=bool)
    use_asymptotic = ~use_exact & ~identical
    u = np.full(values.shape[1], np.nan)
    p = np.full(values.shape[1], np.nan)
    method = np.full(values.shape[1], "normal_tie_corrected_continuity", dtype=object)
    if use_exact.any():
        result = mannwhitneyu(x[:, use_exact], y[:, use_exact], axis=0, alternative="two-sided", method="exact")
        u[use_exact], p[use_exact] = result.statistic, result.pvalue
        method[use_exact] = "exact_no_ties_both_n_lt_50"
    if use_asymptotic.any():
        result = mannwhitneyu(
            x[:, use_asymptotic], y[:, use_asymptotic], axis=0,
            alternative="two-sided", method="asymptotic", use_continuity=True,
        )
        u[use_asymptotic], p[use_asymptotic] = result.statistic, result.pvalue
    if identical.any():
        u[identical], p[identical] = len(x) * len(y) / 2.0, 1.0
        method[identical] = "degenerate_all_identical_p1_convention"
    return pd.DataFrame({"KO": numeric.columns, "wilcoxon_u": u, "pvalue": p,
                         "bh_qvalue": benjamini_hochberg(p), "test_method": method})


def matchado_bh_shared_wgs_reference_metrics(
    truth: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    labels: pd.Series,
    *,
    group_a: str,
    group_b: str,
    prevalence_threshold: float = 0.05,
    q_threshold: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project extension: BH recovery on one WGS-defined KO family.

    Samples are normalized over every source column before filtering. The
    family is KOs detected in at least ``prevalence_threshold`` of labeled WGS
    samples. Prediction KOs absent from that fixed family are zero-filled;
    there is no prediction-side prevalence filter.
    """
    if not predictions:
        raise ValueError("at least one prediction is required")
    truth = truth.astype(float).copy()
    truth.index, truth.columns = truth.index.astype(str), truth.columns.astype(str)
    fixed = _labels(labels, truth.index, group_a, group_b)
    truth_relative, totals = normalize_rows(truth)
    if (totals <= 0).any():
        raise ValueError("truth contains a zero-total sample")
    counts = (truth_relative.fillna(0.0) > 0).sum(axis=0)
    family = sorted(counts.index[counts.to_numpy() >= np.ceil(prevalence_threshold * len(truth))].astype(str))
    if not family:
        raise ValueError("the WGS prevalence family is empty")
    wgs_tests = r_default_wilcoxon_by_ko(truth_relative.loc[:, family], fixed, group_a, group_b)
    wgs_positive = wgs_tests.bh_qvalue.to_numpy(float) < q_threshold
    summaries, details = [], []
    for method, frame in predictions.items():
        candidate = frame.astype(float).copy()
        candidate.index, candidate.columns = candidate.index.astype(str), candidate.columns.astype(str)
        if list(candidate.index) != list(truth.index):
            raise ValueError(f"{method} samples are not identically aligned to WGS")
        relative, method_totals = normalize_rows(candidate)
        if (method_totals <= 0).any():
            raise ValueError(f"{method} contains a zero-total sample")
        aligned = relative.reindex(columns=family, fill_value=0.0).fillna(0.0)
        pred_tests = r_default_wilcoxon_by_ko(aligned, fixed, group_a, group_b)
        pred_positive = pred_tests.bh_qvalue.to_numpy(float) < q_threshold
        score = classification_set_metrics(set(np.asarray(family)[wgs_positive]), set(np.asarray(family)[pred_positive]))
        summaries.append({"prediction_method": method, **score, "n_tested_kos": len(family),
                          "n_wgs_significant": int(wgs_positive.sum()),
                          "n_prediction_significant": int(pred_positive.sum()),
                          "prevalence_threshold": prevalence_threshold, "q_threshold": q_threshold,
                          "ko_family_policy": "wgs_only_prevalence_then_prediction_zero_fill"})
        merged = wgs_tests.merge(pred_tests, on="KO", suffixes=("_wgs", "_prediction"), validate="one_to_one")
        merged.insert(0, "prediction_method", method)
        merged["wgs_positive"] = wgs_positive
        merged["prediction_positive"] = pred_positive
        details.append(merged)
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)
