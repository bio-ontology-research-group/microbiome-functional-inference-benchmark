import numpy as np
import pandas as pd

from picrust_benchmark_metrics.sun import (
    permute_ko_abundances_across_samples,
    sun_inference_correlation_abundance_randomization,
)


def test_sun_control_keeps_each_ko_marginal_and_labels_fixed():
    truth = pd.DataFrame(
        {"K1": [9, 8, 7, 1, 2, 1], "K2": [1, 2, 1, 8, 7, 9], "K3": [2, 3, 4, 4, 3, 2]},
        index=list("abcdef"),
    )
    permuted = permute_ko_abundances_across_samples(truth, np.random.default_rng(3))
    for ko in truth:
        assert np.array_equal(np.sort(truth[ko]), np.sort(permuted[ko]))
    labels = pd.Series(["case", "case", "case", "control", "control", "control"], index=truth.index)
    summary, randomizations, details = sun_inference_correlation_abundance_randomization(
        truth, {"PICRUSt2": truth.copy()}, labels, group_a="case", group_b="control", n_randomizations=3, random_seed=3
    )
    assert summary.loc[0, "published_source"] == "Sun et al. 2020"
    assert summary.loc[0, "published_source_doi"] == "10.1186/s40168-020-00815-y"
    assert not randomizations.labels_permuted.any()
    assert len(details) == 3
