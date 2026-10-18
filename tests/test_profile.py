import numpy as np
import pandas as pd

from picrust_benchmark_metrics.profile import (
    binary_ranking_metrics,
    per_ko_spearman,
    per_sample_richness_overlap,
    samplewise_profile_metrics,
    samplewise_spearman,
    samplewise_spearman_on_universe,
    within_study_k_shot_mean_benchmark,
)


def test_samplewise_spearman_excludes_zero_totals_and_marks_constant_prediction():
    truth = pd.DataFrame([[1, 2, 3], [0, 0, 0], [1, 2, 3]], index=["a", "b", "c"], columns=["K1", "K2", "K3"])
    prediction = pd.DataFrame([[3, 2, 1], [1, 1, 1], [2, 2, 2]], index=["a", "b", "c"], columns=["K1", "K2", "K3"])
    result = samplewise_spearman(truth, prediction).set_index("sample_id")
    assert list(result.index) == ["a", "c"]
    assert result.loc["a", "spearman_r"] == -1.0
    assert result.loc["c", "spearman_status"] == "prediction_constant"
    assert np.isnan(result.loc["c", "spearman_r"])


def test_exploratory_profile_extensions_are_available_without_model_code():
    truth = pd.DataFrame([[1, 2, 3], [2, 3, 4], [4, 1, 2]], index=["a", "b", "c"], columns=["K1", "K2", "K3"])
    prediction = truth.copy()
    assert samplewise_profile_metrics(truth, prediction).spearman_r.eq(1).all()
    assert per_ko_spearman(truth, prediction).spearman_across_samples.eq(1).all()
    assert per_sample_richness_overlap(truth, prediction).fraction_wgs_nonzero_kos_recovered_by_method.eq(1).all()


def test_wgs_universe_profile_metric_zero_fills_unavailable_prediction_kos():
    truth = pd.DataFrame([[2, 1, 4], [3, 2, 1]], index=["a", "b"], columns=["K1", "K2", "K3"])
    prediction = pd.DataFrame([[2, 1], [3, 2]], index=["a", "b"], columns=["K1", "K2"])
    scores = samplewise_spearman_on_universe(truth, prediction)
    assert scores.n_evaluation_kos.eq(3).all()
    assert scores.spearman_r.notna().all()
    assert scores.ko_surface_policy.eq("ordered_wgs_positive_universe_zero_fill_then_l1").all()


def test_detection_and_target_excluded_k_shot_reference_are_deterministic():
    metrics = binary_ranking_metrics([True, False, True, False], [0.9, 0.1, 0.8, 0.2])
    assert metrics["auroc"] == 1.0 and metrics["aupr"] == 1.0
    truth = pd.DataFrame([[1, 2, 3], [2, 4, 1], [3, 1, 2]], index=["a", "b", "c"], columns=["K1", "K2", "K3"])
    first, metadata = within_study_k_shot_mean_benchmark(truth, truth.columns, k_values=[1], n_repeats=2, random_seed=7)
    second, _ = within_study_k_shot_mean_benchmark(truth, truth.columns, k_values=[1], n_repeats=2, random_seed=7)
    assert first.equals(second)
    assert first.target_excluded_from_reference.all()
    assert metadata["target_sample_excluded"] is True
