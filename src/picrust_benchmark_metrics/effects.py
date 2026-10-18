"""Project extensions for biological-group effect-size comparison."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from .matchado import _labels
from .profile import normalize_rows


def rank_biserial_effect_by_ko(matrix: pd.DataFrame, labels: pd.Series, group_a: str, group_b: str) -> pd.DataFrame:
    """Per-KO signed rank-biserial effect: ``2U/(nA*nB)-1`` for group A."""
    numeric = matrix.astype(float)
    numeric.index, numeric.columns = numeric.index.astype(str), numeric.columns.astype(str)
    if numeric.index.duplicated().any() or numeric.columns.duplicated().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("matrix must have unique identifiers and finite values")
    fixed = _labels(labels, numeric.index, group_a, group_b)
    a, b = numeric.loc[fixed.eq(group_a)].to_numpy(), numeric.loc[fixed.eq(group_b)].to_numpy()
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each group needs at least two samples")
    u = np.asarray(mannwhitneyu(a, b, axis=0, alternative="two-sided", method="auto").statistic, dtype=float)
    return pd.DataFrame({"KO": numeric.columns, "mannwhitney_u": u,
                         "rank_biserial_effect": 2.0 * u / (len(a) * len(b)) - 1.0,
                         "n_group_a": len(a), "n_group_b": len(b)})


def _correlation(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, str]:
    if len(x) < 3:
        return np.nan, "too_few_kos"
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan, "constant_vector"
    value = spearmanr(x, y).statistic if kind == "spearman" else pearsonr(x, y).statistic
    return float(value), "defined"


def biological_group_effect_concordance(
    truth: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    labels: pd.Series,
    *,
    group_a: str,
    group_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare WGS and predicted rank-biserial effects on one shared surface.

    This is the project's unfiltered effect relationship: all shared KOs stay
    in the surface, including structural zeros. It is not the Matchado metric.
    """
    if not predictions:
        raise ValueError("at least one prediction is required")
    truth = truth.copy().astype(float); truth.index, truth.columns = truth.index.astype(str), truth.columns.astype(str)
    fixed = _labels(labels, truth.index, group_a, group_b)
    matrices = {"WGS": truth, **{name: frame.copy().astype(float) for name, frame in predictions.items()}}
    kos = set(truth.columns)
    for name, frame in matrices.items():
        frame.index, frame.columns = frame.index.astype(str), frame.columns.astype(str)
        if list(frame.index) != list(truth.index) or frame.columns.duplicated().any():
            raise ValueError(f"{name} is not aligned to WGS")
        kos &= set(frame.columns)
        matrices[name] = frame
    kos = sorted(kos)
    if len(kos) < 3:
        raise ValueError("at least three shared KOs are required")
    relative = {}
    for name, frame in matrices.items():
        relative[name], totals = normalize_rows(frame.loc[:, kos])
        if (totals <= 0).any():
            raise ValueError(f"{name} has a zero-total sample on the shared surface")
        relative[name] = relative[name].fillna(0.0)
    wgs = rank_biserial_effect_by_ko(relative["WGS"], fixed, group_a, group_b).set_index("KO")
    summary, details = [], []
    for method in predictions:
        pred = rank_biserial_effect_by_ko(relative[method], fixed, group_a, group_b).set_index("KO")
        x, y = wgs.loc[kos, "rank_biserial_effect"].to_numpy(), pred.loc[kos, "rank_biserial_effect"].to_numpy()
        spearman, spearman_status = _correlation(x, y, "spearman")
        pearson, pearson_status = _correlation(x, y, "pearson")
        summary.append({"prediction_method": method, "spearman_effect_correlation": spearman,
                        "pearson_effect_correlation": pearson, "spearman_status": spearman_status,
                        "pearson_status": pearson_status, "n_shared_kos": len(kos), "n_group_a": int((fixed == group_a).sum()),
                        "n_group_b": int((fixed == group_b).sum()), "effect_definition": "2U/(n_group_a*n_group_b)-1",
                        "profile_normalization": "per_sample_l1", "ko_surface_policy": "shared_intersection_no_prevalence_filter"})
        table = pd.DataFrame({"KO": kos, "prediction_method": method, "rank_biserial_effect_wgs": x,
                              "rank_biserial_effect_prediction": y})
        table["effect_difference_prediction_minus_wgs"] = y - x
        table["effect_direction_agrees"] = np.sign(x) == np.sign(y)
        details.append(table)
    return pd.DataFrame(summary), pd.concat(details, ignore_index=True)
