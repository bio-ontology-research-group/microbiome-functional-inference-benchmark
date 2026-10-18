import numpy as np
import pandas as pd

from picrust_benchmark_metrics.matchado import (
    benjamini_hochberg,
    classification_set_metrics,
    matchado_bh_shared_wgs_reference_metrics,
    matchado_differential_abundance_metrics,
    r_default_wilcoxon_by_ko,
)


def test_empty_prediction_uses_legacy_zero_division_policy():
    assert classification_set_metrics({"K1"}, set()) == {"tp": 0, "fp": 0, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_matchado_metrics_return_published_metric_family():
    truth = pd.DataFrame({"K1": [8, 9, 7, 1, 2, 1], "K2": [1, 1, 2, 7, 8, 9], "K3": [3, 3, 3, 3, 3, 3]}, index=list("abcdef"))
    pred = truth.copy(); pred["K1"] = [6, 7, 6, 1, 1, 1]
    labels = pd.Series(["A", "A", "A", "B", "B", "B"], index=truth.index)
    scores, tests = matchado_differential_abundance_metrics(truth, {"PICRUSt2": pred}, labels, group_a="A", group_b="B", n_permutations=3)
    assert set(scores.metric) == {"precision", "recall", "f1", "auroc", "aupr"}
    assert set(scores.published_source) == {"Matchado et al. 2024"}
    assert set(scores.published_source_doi) == {"10.1099/mgen.0.001203"}
    assert set(scores.loc[scores.metric.isin(["precision", "recall", "f1"]), "metric_provenance"]) == {"published_matchado"}
    assert set(scores.loc[scores.metric.isin(["auroc", "aupr"]), "metric_provenance"]) == {"project_ranking_extension"}
    assert len(tests) == 3 and set(tests.prediction_method) == {"PICRUSt2"}


def test_bh_and_degenerate_r_default_wilcoxon():
    matrix = pd.DataFrame({"K0": [0, 0, 0, 0], "K1": [0, 1, 3, 4]}, index=["a", "b", "c", "d"])
    labels = pd.Series(["A", "A", "B", "B"], index=matrix.index)
    table = r_default_wilcoxon_by_ko(matrix, labels, "A", "B").set_index("KO")
    assert table.loc["K0", "pvalue"] == 1
    assert table.loc["K0", "bh_qvalue"] == 1
    assert table.loc["K0", "test_method"] == "degenerate_all_identical_p1_convention"
    assert np.allclose(benjamini_hochberg([0.01, 0.04, 0.5]), [0.03, 0.06, 0.5])


def test_shared_wgs_family_zero_fills_missing_prediction_ko():
    ids = [f"s{i}" for i in range(8)]
    labels = pd.Series(["A"] * 4 + ["B"] * 4, index=ids)
    truth = pd.DataFrame({"K1": [8, 7, 9, 8, 1, 0, 1, 0], "K2": [1, 1, 1, 1, 7, 8, 7, 8]}, index=ids)
    prediction = pd.DataFrame({"K1": truth.K1, "background": 1.0}, index=ids)
    summary, detail = matchado_bh_shared_wgs_reference_metrics(
        truth, {"method": prediction}, labels, group_a="A", group_b="B"
    )
    assert summary.loc[0, "n_tested_kos"] == 2
    missing = detail.set_index("KO").loc["K2"]
    assert missing["pvalue_prediction"] == 1
    assert missing["test_method_prediction"] == "degenerate_all_identical_p1_convention"
