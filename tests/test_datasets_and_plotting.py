import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from picrust_benchmark_metrics import (
    load_processed_study,
    plot_directional_metric_summary,
    plot_metric_row_grid,
)


def test_processed_study_loader(tmp_path):
    path = tmp_path / "study.npz"
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["s1", "s2"]),
        ko_ids=np.asarray(["K1", "K2"]),
        labels=np.asarray(["A", "B"]),
        wgs=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        picrust2=np.asarray([[1.0, 0.0], [2.0, 1.0]]),
        picrust2_adaptation=np.asarray([[1.0, 1.0], [2.0, 2.0]]),
        tax4fun2=np.asarray([[0.0, 1.0], [1.0, 2.0]]),
        metadata_json=np.asarray(json.dumps({"dataset": "study", "group_a": "A", "group_b": "B"})),
    )
    truth, predictions, labels, metadata = load_processed_study(path)
    assert truth.shape == (2, 2)
    assert set(predictions) == {"PICRUSt2", "PICRUSt2 + adaptation", "Tax4Fun2"}
    assert labels.to_dict() == {"s1": "A", "s2": "B"}
    assert metadata["dataset"] == "study"


def test_reusable_plot_functions_smoke():
    recovery = pd.DataFrame([
        {"dataset": "d", "prediction_method": method, "metric": metric, "estimate": .5}
        for method in ("PICRUSt2", "Tax4Fun2")
        for metric in ("f1", "recall", "precision")
    ])
    fig, _ = plot_metric_row_grid(recovery, ["d"], ["PICRUSt2", "Tax4Fun2"])
    plt.close(fig)

    curves = pd.DataFrame([
        {"dataset": "d", "prediction_method": method, "direction": direction,
         "ap": .4, "roc_auc": .6, "baseline_target_prevalence": .2}
        for method in ("PICRUSt2", "Tax4Fun2")
        for direction in ("negative", "positive")
    ])
    fig, _ = plot_directional_metric_summary(curves, "roc_auc", ["PICRUSt2", "Tax4Fun2"])
    plt.close(fig)
