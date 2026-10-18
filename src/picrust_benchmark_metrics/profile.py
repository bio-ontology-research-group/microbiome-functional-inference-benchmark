"""Sample-profile metrics with PICRUSt/PICRUSt2 benchmarking precedent.

References: Langille et al. (2013), doi:10.1038/nbt.2676, and Douglas et al.
(2020), doi:10.1038/s41587-020-0548-6.

Inputs are sample-by-KO tables.  The profile comparison follows the source
notebook: restrict to the pairwise shared sample and KO surface, L1-normalize
each row, exclude a sample if either row total is zero, then correlate KO ranks
within each retained sample. The exact surface and zero-fill rules are this
repository's documented operationalization, not a claim of verbatim paper code.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _checked(matrix: pd.DataFrame, name: str) -> pd.DataFrame:
    if matrix.index.duplicated().any():
        raise ValueError(f"{name} has duplicate sample identifiers")
    if matrix.columns.duplicated().any():
        raise ValueError(f"{name} has duplicate KO identifiers")
    numeric = matrix.astype(float)
    if not np.isfinite(numeric.to_numpy()).all() or (numeric.to_numpy() < 0).any():
        raise ValueError(f"{name} must contain finite, non-negative values")
    numeric.index = numeric.index.astype(str)
    numeric.columns = numeric.columns.astype(str)
    return numeric


def normalize_rows(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return L1 relative abundances and pre-normalization row totals.

    Zero-total rows are represented as NaN; callers decide whether to exclude
    or otherwise handle them.
    """
    numeric = _checked(matrix, "matrix")
    totals = numeric.sum(axis=1)
    return numeric.div(totals.replace(0.0, np.nan), axis=0), totals


def align_to_ko_universe(
    matrix: pd.DataFrame,
    ko_ids: Sequence[str],
    *,
    require_all: bool = False,
) -> pd.DataFrame:
    """Return ``matrix`` on an ordered KO surface, zero-filling absent KOs.

    This is the expanded PICRUSt2/adaptation evaluation rule: a WGS-derived
    KO universe is fixed first, then every prediction is reindexed to it.
    It deliberately differs from the legacy pairwise-intersection profile
    metric, which remains available through :func:`samplewise_spearman`.
    """
    numeric = _checked(matrix, "matrix")
    ordered = [str(ko) for ko in ko_ids]
    if len(ordered) != len(set(ordered)):
        raise ValueError("KO universe contains duplicate identifiers")
    missing = sorted(set(ordered) - set(numeric.columns))
    if require_all and missing:
        raise ValueError(f"matrix is missing {len(missing)} required KO identifiers")
    return numeric.reindex(columns=ordered, fill_value=0.0)


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, str]:
    """Spearman correlation with explicit undefined-result statuses."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if int(finite.sum()) < 3:
        return np.nan, np.nan, "too_few_finite_values"
    x_values, y_values = x_values[finite], y_values[finite]
    if np.allclose(x_values, x_values[0]):
        return np.nan, np.nan, "truth_constant"
    if np.allclose(y_values, y_values[0]):
        return np.nan, np.nan, "prediction_constant"
    rho, pvalue = spearmanr(x_values, y_values)
    return float(rho), float(pvalue), "defined"


def samplewise_spearman_on_universe(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    ko_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Score profiles after WGS-surface alignment and zero filling.

    If ``ko_ids`` is omitted, the ordered universe is the sorted set of KOs
    positive in the supplied WGS rows.  Samples must have identical order;
    this fail-closed condition is important for cached held-out predictions.
    """
    truth_checked, prediction_checked = _checked(truth, "truth"), _checked(prediction, "prediction")
    if list(truth_checked.index) != list(prediction_checked.index):
        raise ValueError("truth and prediction samples must be identically aligned")
    if ko_ids is None:
        ko_ids = sorted(truth_checked.columns[truth_checked.sum(axis=0).gt(0)])
    truth_aligned = align_to_ko_universe(truth_checked, ko_ids, require_all=True)
    prediction_aligned = align_to_ko_universe(prediction_checked, ko_ids)
    truth_relative, truth_totals = normalize_rows(truth_aligned)
    prediction_relative, prediction_totals = normalize_rows(prediction_aligned)
    if (truth_totals <= 0).any() or (prediction_totals <= 0).any():
        raise ValueError("zero-total sample after KO-universe alignment")
    rows = []
    for sample_id in truth_aligned.index:
        rho, pvalue, status = safe_spearman(
            truth_relative.loc[sample_id].to_numpy(dtype=float),
            prediction_relative.loc[sample_id].to_numpy(dtype=float),
        )
        rows.append(
            {
                "sample_id": sample_id,
                "n_evaluation_kos": len(ko_ids),
                "wgs_total_before_norm": float(truth_totals.loc[sample_id]),
                "method_total_before_norm": float(prediction_totals.loc[sample_id]),
                "spearman_r": rho,
                "spearman_pvalue": pvalue,
                "spearman_status": status,
                "ko_surface_policy": "ordered_wgs_positive_universe_zero_fill_then_l1",
            }
        )
    return pd.DataFrame(rows)


def binary_ranking_metrics(
    y_true: Sequence[bool | int], y_score: Sequence[float]
) -> dict:
    """Return detection AUROC and average-precision AUPRC with class audits."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    truth_values = np.asarray(y_true, dtype=bool)
    score_values = np.asarray(y_score, dtype=float)
    if truth_values.shape != score_values.shape:
        raise ValueError("truth and score must have identical shapes")
    finite = np.isfinite(score_values)
    truth_values, score_values = truth_values[finite], score_values[finite]
    n_positive = int(truth_values.sum())
    n_negative = int((~truth_values).sum())
    status = "defined"
    if n_positive == 0:
        status = "no_positive_class"
    elif n_negative == 0:
        status = "no_negative_class"
    return {
        "auroc": float(roc_auc_score(truth_values, score_values)) if status == "defined" else np.nan,
        "aupr": float(average_precision_score(truth_values, score_values)) if status == "defined" else np.nan,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_fraction": n_positive / len(truth_values) if len(truth_values) else np.nan,
        "status": status,
    }


def within_study_k_shot_mean_benchmark(
    truth: pd.DataFrame,
    ko_ids: Sequence[str],
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
    n_repeats: int = 20,
    random_seed: int = 20260822,
) -> tuple[pd.DataFrame, dict]:
    """Target-excluded, same-study WGS mean-profile reference.

    This is a model-free few-shot reference, not a training or LOSO routine.
    Each target/repeat has a deterministic independent RNG seed so results do
    not depend on dataset iteration order.
    """
    aligned = align_to_ko_universe(truth, ko_ids, require_all=True)
    normalized, totals = normalize_rows(aligned)
    if (totals <= 0).any():
        raise ValueError("truth has a zero-total sample")
    normalized = normalized.fillna(0.0)
    sample_ids = list(normalized.index)
    supported_k = sorted({int(k) for k in k_values if 1 <= int(k) < len(sample_ids)})
    if not supported_k:
        raise ValueError("at least one k must be positive and smaller than sample count")
    if int(n_repeats) < 1:
        raise ValueError("n_repeats must be positive")
    values, rows = normalized.to_numpy(dtype=float), []
    for k in supported_k:
        for position, sample_id in enumerate(sample_ids):
            candidates = np.delete(np.arange(len(sample_ids)), position)
            seed = int.from_bytes(
                hashlib.sha256(f"{int(random_seed)}|{k}|{sample_id}".encode()).digest()[:8],
                byteorder="little",
                signed=False,
            )
            rng = np.random.default_rng(seed)
            for repeat in range(1, int(n_repeats) + 1):
                picked = np.sort(rng.choice(candidates, size=k, replace=False))
                rho, pvalue, status = safe_spearman(values[position], values[picked].mean(axis=0))
                rows.append({
                    "sample_id": sample_id, "k": k, "repeat": repeat,
                    "spearman_r": rho, "spearman_pvalue": pvalue,
                    "spearman_status": status,
                    "reference_sample_ids": ";".join(sample_ids[value] for value in picked),
                    "target_excluded_from_reference": True,
                })
    digest = hashlib.sha256("\n".join(map(str, ko_ids)).encode()).hexdigest()
    return pd.DataFrame(rows), {
        "n_samples": len(sample_ids), "n_kos": len(ko_ids),
        "supported_k_values": supported_k, "n_repeats": int(n_repeats),
        "random_seed": int(random_seed), "ko_ids_sha256": digest,
        "target_sample_excluded": True, "labels_used": False,
        "construction": "mean_relative_KO_profile_of_k_random_same_study_WGS_peers",
    }


def profile_overlap(truth: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    """Report the pairwise sample/KO intersection used by profile scoring."""
    truth, prediction = _checked(truth, "truth"), _checked(prediction, "prediction")
    samples = sorted(set(truth.index) & set(prediction.index))
    kos = sorted(set(truth.columns) & set(prediction.columns))
    return {
        "shared_samples": samples,
        "shared_kos": kos,
        "n_shared_samples": len(samples),
        "n_shared_kos": len(kos),
        "n_truth_kos": truth.shape[1],
        "n_prediction_kos": prediction.shape[1],
        "n_truth_kos_missing_from_prediction": len(set(truth.columns) - set(prediction.columns)),
        "n_prediction_kos_absent_from_truth": len(set(prediction.columns) - set(truth.columns)),
    }


def samplewise_spearman(truth: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    """Compute the legacy per-sample PICRUSt-vs-WGS Spearman table.

    Constant vectors are explicitly marked undefined, matching the source
    notebook's NaN result.  This function does not impute absent KO columns.
    """
    truth, prediction = _checked(truth, "truth"), _checked(prediction, "prediction")
    overlap = profile_overlap(truth, prediction)
    samples, kos = overlap["shared_samples"], overlap["shared_kos"]
    if not samples or not kos:
        raise ValueError("at least one shared sample and KO are required")
    truth_n, truth_totals = normalize_rows(truth.loc[samples, kos])
    pred_n, pred_totals = normalize_rows(prediction.loc[samples, kos])
    keep = (truth_totals > 0) & (pred_totals > 0)
    rows: list[dict] = []
    for sample_id in truth_n.index[keep]:
        x = truth_n.loc[sample_id].to_numpy(float)
        y = pred_n.loc[sample_id].to_numpy(float)
        if len(x) < 2:
            rho, pvalue, status = np.nan, np.nan, "too_few_kos"
        elif np.all(x == x[0]):
            rho, pvalue, status = np.nan, np.nan, "truth_constant"
        elif np.all(y == y[0]):
            rho, pvalue, status = np.nan, np.nan, "prediction_constant"
        else:
            rho, pvalue = spearmanr(x, y)
            status = "defined"
        rows.append(
            {
                "sample_id": sample_id,
                "n_common_kos": len(kos),
                "wgs_total_before_norm": truth_totals.loc[sample_id],
                "method_total_before_norm": pred_totals.loc[sample_id],
                "spearman_r": rho,
                "spearman_pvalue": pvalue,
                "spearman_status": status,
            }
        )
    return pd.DataFrame(rows)


def per_sample_richness_overlap(truth: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's nonzero-KO overlap diagnostic per sample."""
    truth, prediction = _checked(truth, "truth"), _checked(prediction, "prediction")
    samples = sorted(set(truth.index) & set(prediction.index))
    shared_kos = sorted(set(truth.columns) & set(prediction.columns))
    rows = []
    for sample_id in samples:
        truth_present = set(truth.columns[truth.loc[sample_id] > 0])
        predicted_present = set(prediction.columns[prediction.loc[sample_id] > 0])
        overlap = truth_present & predicted_present
        rows.append({"sample_id": sample_id, "n_nonzero_kos_wgs": len(truth_present),
                     "n_nonzero_kos_method": len(predicted_present), "n_nonzero_kos_overlap": len(overlap),
                     "fraction_wgs_nonzero_kos_recovered_by_method": len(overlap) / len(truth_present) if truth_present else np.nan,
                     "n_shared_kos_available_for_method": len(shared_kos)})
    return pd.DataFrame(rows)


def samplewise_profile_metrics(truth: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    """Return the exploratory notebook's Spearman, Pearson, and cosine profile metrics.

    As with :func:`samplewise_spearman`, scoring is pairwise on a common KO
    surface after per-sample L1 normalization and zero-total exclusion.
    """
    truth, prediction = _checked(truth, "truth"), _checked(prediction, "prediction")
    overlap = profile_overlap(truth, prediction)
    samples, kos = overlap["shared_samples"], overlap["shared_kos"]
    if not samples or not kos:
        raise ValueError("at least one shared sample and KO are required")
    truth_n, truth_totals = normalize_rows(truth.loc[samples, kos])
    pred_n, pred_totals = normalize_rows(prediction.loc[samples, kos])
    rows = []
    for sample_id in truth_n.index[(truth_totals > 0) & (pred_totals > 0)]:
        x, y = truth_n.loc[sample_id].to_numpy(float), pred_n.loc[sample_id].to_numpy(float)
        constant_x, constant_y = len(x) < 2 or np.all(x == x[0]), len(y) < 2 or np.all(y == y[0])
        status = "too_few_kos" if len(x) < 2 else ("truth_constant" if constant_x else ("prediction_constant" if constant_y else "defined"))
        if status == "defined":
            spearman, spearman_p = spearmanr(x, y)
            pearson, pearson_p = pearsonr(x, y)
            cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        else:
            spearman = spearman_p = pearson = pearson_p = cosine = np.nan
        rows.append({"sample_id": sample_id, "n_common_kos": len(kos), "wgs_total_before_norm": truth_totals.loc[sample_id],
                     "method_total_before_norm": pred_totals.loc[sample_id], "spearman_r": spearman,
                     "spearman_pvalue": spearman_p, "pearson_r": pearson, "pearson_pvalue": pearson_p,
                     "cosine_similarity": cosine, "status": status})
    return pd.DataFrame(rows)


def per_ko_spearman(truth: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    """Exploratory per-KO Spearman across matched samples on the shared surface."""
    truth, prediction = _checked(truth, "truth"), _checked(prediction, "prediction")
    overlap = profile_overlap(truth, prediction)
    samples, kos = overlap["shared_samples"], overlap["shared_kos"]
    if len(samples) < 2 or not kos:
        raise ValueError("at least two shared samples and one KO are required")
    truth_n, truth_totals = normalize_rows(truth.loc[samples, kos])
    pred_n, pred_totals = normalize_rows(prediction.loc[samples, kos])
    keep = (truth_totals > 0) & (pred_totals > 0)
    rows = []
    for ko in kos:
        x, y = truth_n.loc[keep, ko].to_numpy(float), pred_n.loc[keep, ko].to_numpy(float)
        if len(x) < 3:
            rho, pvalue, status = np.nan, np.nan, "too_few_samples"
        elif np.all(x == x[0]):
            rho, pvalue, status = np.nan, np.nan, "truth_constant"
        elif np.all(y == y[0]):
            rho, pvalue, status = np.nan, np.nan, "prediction_constant"
        else:
            rho, pvalue, status = *spearmanr(x, y), "defined"
        rows.append({"KO": ko, "n_samples": int(keep.sum()), "wgs_mean": float(np.mean(x)),
                     "wgs_nonzero_fraction": float(np.mean(x > 0)), "spearman_across_samples": rho,
                     "spearman_pvalue": pvalue, "spearman_status": status})
    return pd.DataFrame(rows)


def summarize_samplewise_spearman(scores: pd.DataFrame) -> dict:
    """Summarize a table returned by :func:`samplewise_spearman`."""
    values = scores["spearman_r"].dropna()
    return {
        "n_samples": len(scores),
        "n_common_kos": int(scores["n_common_kos"].iloc[0]) if len(scores) else 0,
        "mean_spearman": values.mean(),
        "median_spearman": values.median(),
        "std_spearman": values.std(),
        "min_spearman": values.min(),
        "max_spearman": values.max(),
    }
