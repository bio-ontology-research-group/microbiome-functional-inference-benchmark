"""ROC/PR and decile outcome-composition diagnostics for effect estimates."""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.axes
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


Q_THRESHOLD = 0.05
OUTCOME_ORDER = ("sig_negative", "nonsignificant", "sig_positive")
OUTCOME_COLORS = {"sig_negative": "#B2182B", "nonsignificant": "#CFCFCF", "sig_positive": "#D9A520"}


def wgs_outcome_class(qvalue: Sequence[float], effect: Sequence[float], *, threshold: float = Q_THRESHOLD) -> np.ndarray:
    """Classify WGS outcomes from BH q-values and signed WGS effects."""
    q, e = np.asarray(qvalue, float), np.asarray(effect, float)
    if q.shape != e.shape or not np.isfinite(q).all() or not np.isfinite(e).all():
        raise ValueError("qvalue and effect must be finite vectors of equal length")
    if ((q < 0) | (q > 1)).any():
        raise ValueError("qvalue must lie in [0, 1]")
    significant = q <= threshold
    if (significant & (e == 0)).any():
        raise ValueError("a significant WGS KO cannot have exactly zero effect")
    return np.where(significant & (e > 0), "sig_positive", np.where(significant & (e < 0), "sig_negative", "nonsignificant"))


def _oriented(effect: np.ndarray, direction: str) -> np.ndarray:
    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be 'positive' or 'negative'")
    value = np.asarray(effect, float)
    if not np.isfinite(value).all():
        raise ValueError("prediction effects must be finite")
    return (value if direction == "positive" else -value) + 0.0


def _native_threshold(oriented: float, direction: str) -> float:
    return float((oriented if direction == "positive" else -oriented) + 0.0)


def _curve_rows(score: np.ndarray, target: np.ndarray, direction: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    n, positives = len(target), int(target.sum())
    baseline = positives / n if n else np.nan
    base = {"direction": direction, "K": n, "target_total": positives, "baseline_target_prevalence": baseline}
    if positives == 0 or positives == n:
        status = "undefined_no_targets" if positives == 0 else "undefined_all_targets"
        return pd.DataFrame(), pd.DataFrame(), {**base, "ap": np.nan, "roc_auc": np.nan, "status": status}
    precision, recall, pr_thresholds = precision_recall_curve(target.astype(int), score)
    fpr, tpr, roc_thresholds = roc_curve(target.astype(int), score, drop_intermediate=False)
    pr = pd.DataFrame({"direction": direction, "point_type": ["threshold"] * len(pr_thresholds) + ["sklearn_plot_origin"],
                       "threshold_oriented": list(pr_thresholds) + [np.nan],
                       "threshold_native": [_native_threshold(v, direction) for v in pr_thresholds] + [np.nan],
                       "precision": precision, "recall": recall})
    roc = pd.DataFrame({"direction": direction, "threshold_oriented": roc_thresholds,
                        "threshold_native": [_native_threshold(v, direction) if np.isfinite(v) else np.nan for v in roc_thresholds],
                        "fpr": fpr, "tpr": tpr})
    return pr, roc, {**base, "ap": float(average_precision_score(target.astype(int), score)), "roc_auc": float(roc_auc_score(target.astype(int), score)), "status": "defined"}


def directional_ranking_curves(wgs_qvalue: Sequence[float], wgs_effect: Sequence[float], prediction_effect: Sequence[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute exact sklearn ROC/PR curves for positive and negative WGS targets.

    Positive targets are WGS ``q <= .05 & effect > 0``; negative targets use
    the analogous negative sign and orient the score by ``-prediction_effect``.
    AP is sklearn average precision, not trapezoid PR-AUC.
    """
    classes = wgs_outcome_class(wgs_qvalue, wgs_effect)
    predicted = np.asarray(prediction_effect, float)
    if len(classes) != len(predicted):
        raise ValueError("effect vectors differ in length")
    prs, rocs, summary = [], [], []
    for direction, target_name in (("positive", "sig_positive"), ("negative", "sig_negative")):
        pr, roc, row = _curve_rows(_oriented(predicted, direction), classes == target_name, direction)
        prs.append(pr); rocs.append(roc); summary.append(row)
    summary_table = pd.DataFrame(summary)
    summary_table["mean_directional_ap"] = np.nan
    summary_table["mean_directional_roc_auc"] = np.nan
    if (summary_table.status == "defined").all():
        summary_table.loc[:, "mean_directional_ap"] = summary_table.ap.mean()
        summary_table.loc[:, "mean_directional_roc_auc"] = summary_table.roc_auc.mean()
    return pd.concat(prs, ignore_index=True), pd.concat(rocs, ignore_index=True), summary_table


def decile_outcome_composition(
    ko_ids: Sequence[str], prediction_effect: Sequence[float], wgs_qvalue: Sequence[float], wgs_effect: Sequence[float], *, max_bins: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ascending predicted effects into equal-count rank bins and summarize WGS outcomes.

    Ties are broken lexically by KO identifier, exactly as in the source
    diagnostic, and the assignments record the resulting deterministic order.
    """
    ids = np.asarray(ko_ids).astype(str)
    prediction = np.asarray(prediction_effect, float)
    if len(ids) != len(set(ids)) or not np.isfinite(prediction).all():
        raise ValueError("KO identifiers must be unique and prediction effects finite")
    if not len(ids):
        return pd.DataFrame(columns=["KO", "prediction_effect", "rank0", "bin", "n_bins", "wgs_class"]), pd.DataFrame()
    if max_bins < 1:
        raise ValueError("max_bins must be positive")
    outcome = wgs_outcome_class(wgs_qvalue, wgs_effect)
    if len(outcome) != len(ids):
        raise ValueError("all per-KO vectors must have equal length")
    order = np.lexsort((ids, prediction + 0.0))
    rank0 = np.empty(len(ids), dtype=int); rank0[order] = np.arange(len(ids))
    n_bins = min(len(ids), max_bins)
    bins = (n_bins * rank0) // len(ids) + 1
    assignment = pd.DataFrame({"KO": ids, "prediction_effect": prediction, "rank0": rank0, "bin": bins, "n_bins": n_bins, "wgs_class": outcome})
    rows = []
    for bin_number in range(1, n_bins + 1):
        block = assignment.loc[assignment.bin == bin_number]
        values = block.prediction_effect.to_numpy(float)
        row = {"bin": bin_number, "bin_label": f"D{bin_number}", "n_bins": n_bins, "n_kos": len(block),
               "pred_min": np.quantile(values, 0.0, method="linear"), "pred_q1": np.quantile(values, .25, method="linear"),
               "pred_median": np.quantile(values, .5, method="linear"), "pred_q3": np.quantile(values, .75, method="linear"),
               "pred_max": np.quantile(values, 1.0, method="linear"), "n_unique_prediction": int(np.unique(values + 0.0).size)}
        for outcome_name in OUTCOME_ORDER:
            row[f"n_{outcome_name}"] = int((block.wgs_class == outcome_name).sum())
            row[f"p_{outcome_name}"] = row[f"n_{outcome_name}"] / len(block)
        row["p_any_significant"] = row["p_sig_negative"] + row["p_sig_positive"]
        row["balance_pos_minus_neg"] = row["p_sig_positive"] - row["p_sig_negative"]
        rows.append(row)
    return assignment, pd.DataFrame(rows)


def plot_decile_outcome_composition(summary: pd.DataFrame, ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
    """Draw the established stacked proportion-per-decile plot from a summary table."""
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    required = {"bin_label", *(f"p_{name}" for name in OUTCOME_ORDER)}
    if not required <= set(summary.columns):
        raise ValueError(f"summary needs {sorted(required)}")
    bottom = np.zeros(len(summary))
    x = np.arange(len(summary))
    for name in OUTCOME_ORDER:
        values = summary[f"p_{name}"].to_numpy(float)
        ax.bar(x, values, bottom=bottom, label=name, color=OUTCOME_COLORS[name], width=.82)
        bottom += values
    ax.set(xticks=x, xticklabels=summary.bin_label, ylim=(0, 1), ylabel="Proportion of WGS outcome class", xlabel="Predicted-effect rank bin")
    return ax
