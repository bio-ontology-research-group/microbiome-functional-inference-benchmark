"""Sun-style signed-inference correlation and WGS abundance control."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .matchado import wilcoxon_rank_sum_by_ko
from .profile import normalize_rows, safe_spearman


def permute_ko_abundances_across_samples(
    matrix: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Independently shuffle each WGS KO vector while keeping labels fixed."""
    values = matrix.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("matrix must be finite and contain at least two samples")
    original = values.copy()
    for column in range(values.shape[1]):
        values[:, column] = original[rng.permutation(values.shape[0]), column]
    return pd.DataFrame(values, index=matrix.index.copy(), columns=matrix.columns.copy())


def sun_signed_wilcoxon_by_ko(
    matrix: pd.DataFrame, labels: pd.Series, group_a: str, group_b: str
) -> pd.DataFrame:
    """Sun's log10(p) times mean-difference-sign KO statistic."""
    result = wilcoxon_rank_sum_by_ko(matrix, labels, group_a, group_b)
    fixed = labels.reindex(matrix.index).astype(str)
    mean_difference = (
        matrix.loc[fixed.eq(group_a)].to_numpy(float).mean(axis=0)
        - matrix.loc[fixed.eq(group_b)].to_numpy(float).mean(axis=0)
    )
    result["mean_group_a_minus_b"] = mean_difference
    result["signed_log10_pvalue"] = np.log10(
        np.clip(result.pvalue.to_numpy(float), 1e-300, 1.0)
    ) * np.sign(mean_difference)
    return result


def _labels(labels: pd.Series, sample_ids: Sequence[str], group_a: str, group_b: str) -> pd.Series:
    fixed = labels.copy()
    fixed.index = fixed.index.astype(str)
    fixed = fixed.reindex(pd.Index(sample_ids, dtype=str))
    if fixed.isna().any() or set(fixed.astype(str)) != {group_a, group_b}:
        raise ValueError("labels must be complete and contain exactly the declared groups")
    return fixed.astype(str)


def sun_inference_correlation_abundance_randomization(
    truth: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    labels: pd.Series,
    *,
    group_a: str,
    group_b: str,
    n_randomizations: int = 100,
    random_seed: int = 20260901,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce Sun's signed-inference correlation and WGS-shuffle control.

    This follows the published descriptive control: each WGS KO is shuffled
    over samples, labels stay fixed, and the prediction is never shuffled.
    It is not a label-permutation test and therefore remains distinct from
    Matchado's 20 prediction-label randomizations.
    """
    if not predictions or int(n_randomizations) < 1:
        raise ValueError("at least one prediction and one randomization are required")
    truth = truth.copy().astype(float)
    truth.index, truth.columns = truth.index.astype(str), truth.columns.astype(str)
    fixed = _labels(labels, truth.index, group_a, group_b)
    truth_relative, truth_total = normalize_rows(truth)
    if (truth_total <= 0).any():
        raise ValueError("truth has a zero-total sample")
    truth_relative = truth_relative.fillna(0.0)
    truth_relative = truth_relative.loc[:, truth_relative.sum(axis=0).gt(0)]
    if truth_relative.shape[1] < 3:
        raise ValueError("truth needs three nonzero KOs")
    fixed_tests, common = {}, {}
    for method, prediction in predictions.items():
        candidate = prediction.copy().astype(float)
        candidate.index, candidate.columns = candidate.index.astype(str), candidate.columns.astype(str)
        if list(candidate.index) != list(truth.index):
            raise ValueError(f"{method} samples are not identically aligned to truth")
        relative, totals = normalize_rows(candidate)
        if (totals <= 0).any():
            raise ValueError(f"{method} has a zero-total sample")
        relative = relative.fillna(0.0).loc[:, lambda value: value.sum(axis=0).gt(0)]
        kos = sorted(set(truth_relative.columns) & set(relative.columns))
        if len(kos) < 3:
            raise ValueError(f"{method} has fewer than three shared nonzero KOs")
        common[method] = kos
        fixed_tests[method] = sun_signed_wilcoxon_by_ko(relative.loc[:, kos], fixed, group_a, group_b).set_index("KO")
    truth_tests = sun_signed_wilcoxon_by_ko(truth_relative, fixed, group_a, group_b).set_index("KO")
    observed, detail = [], []
    for method, kos in common.items():
        rho, pvalue, status = safe_spearman(
            truth_tests.loc[kos, "signed_log10_pvalue"], fixed_tests[method].loc[kos, "signed_log10_pvalue"]
        )
        observed.append({"prediction_method": method, "observed_inference_correlation": rho,
                         "observed_spearman_pvalue": pvalue, "observed_spearman_status": status,
                         "n_shared_nonzero_kos": len(kos)})
        joined = truth_tests.loc[kos].reset_index().merge(fixed_tests[method].loc[kos].reset_index(), on="KO", suffixes=("_wgs", "_prediction"))
        joined.insert(0, "prediction_method", method)
        detail.append(joined)
    random_rows = []
    for number in range(1, int(n_randomizations) + 1):
        seed = int.from_bytes(hashlib.sha256(f"{int(random_seed)}|sun_wgs_ko_abundance|{number}".encode()).digest()[:8], "little")
        shuffled = sun_signed_wilcoxon_by_ko(
            permute_ko_abundances_across_samples(truth_relative, np.random.default_rng(seed)), fixed, group_a, group_b
        ).set_index("KO")
        for method, kos in common.items():
            rho, pvalue, status = safe_spearman(shuffled.loc[kos, "signed_log10_pvalue"], fixed_tests[method].loc[kos, "signed_log10_pvalue"])
            random_rows.append({"prediction_method": method, "randomization": number, "randomization_seed": seed,
                                "inference_correlation": rho, "spearman_pvalue": pvalue, "spearman_status": status,
                                "n_shared_nonzero_kos": len(kos), "randomized_profile": "WGS",
                                "fixed_profile": "prediction", "labels_permuted": False,
                                "abundance_permuted_independently_by_ko": True})
    randomizations = pd.DataFrame(random_rows)
    summaries = []
    observed_frame = pd.DataFrame(observed).set_index("prediction_method")
    for method in common:
        null = randomizations.loc[randomizations.prediction_method.eq(method), "inference_correlation"].dropna()
        summaries.append({**observed_frame.loc[method].to_dict(), "prediction_method": method,
                          "randomization_mean": null.mean(), "randomization_sd": null.std(ddof=1),
                          "randomization_min": null.min(), "randomization_max": null.max(),
                          "n_randomizations": int(n_randomizations), "randomization_seed": int(random_seed),
                          "randomization_method": "sun_2020_independent_wgs_ko_abundance_permutation",
                          "published_source": "Sun et al. 2020", "randomized_profile": "WGS",
                          "fixed_profile": "prediction", "labels_permuted": False})
    return pd.DataFrame(summaries), randomizations, pd.concat(detail, ignore_index=True)
