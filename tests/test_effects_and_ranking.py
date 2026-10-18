import numpy as np
import pandas as pd

from picrust_benchmark_metrics.effects import biological_group_effect_concordance
from picrust_benchmark_metrics.ranking import decile_outcome_composition, directional_ranking_curves


def test_effect_concordance_uses_rank_biserial_relationship():
    truth = pd.DataFrame({"K1": [8, 7, 1, 2], "K2": [1, 2, 8, 7], "K3": [3, 4, 3, 4]}, index=list("abcd"))
    labels = pd.Series(["A", "A", "B", "B"], index=truth.index)
    summary, detail = biological_group_effect_concordance(truth, {"pred": truth.copy()}, labels, group_a="A", group_b="B")
    assert summary.loc[0, "spearman_effect_correlation"] == 1.0
    assert detail.effect_direction_agrees.all()


def test_ranking_curves_and_deciles_keep_tied_effects_deterministic():
    q = [.01, .9, .02, .9, .01, .9]
    wgs = [-.8, -.1, .7, .1, .3, -.2]
    pred = [-.5, -.5, .8, .1, .1, -.2]
    pr, roc, summary = directional_ranking_curves(q, wgs, pred)
    assert set(summary.status) == {"defined"}
    assert not pr.empty and not roc.empty
    assignments, bins = decile_outcome_composition([f"K{i}" for i in range(6)], pred, q, wgs, max_bins=3)
    assert assignments.loc[assignments.prediction_effect.eq(-.5), "rank0"].tolist() == [0, 1]
    assert np.allclose(bins[["p_sig_negative", "p_nonsignificant", "p_sig_positive"]].sum(axis=1), 1)
