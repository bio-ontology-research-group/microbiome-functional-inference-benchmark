#!/usr/bin/env python3
"""Replay notebook metrics from the compact study bundles in ``data/studies``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from picrust_benchmark_metrics import (
    load_processed_studies,
    matchado_differential_abundance_metrics,
    samplewise_spearman_on_universe,
    sun_inference_correlation_abundance_randomization,
)


METHODS = ("PICRUSt2", "PICRUSt2 + adaptation", "Tax4Fun2")


def replay_bundled_group_metrics(
    studies_directory: str | Path,
    *,
    matchado_permutations: int = 20,
    sun_randomizations: int = 100,
):
    """Recompute profile, Matchado-style, and Sun-style results for all bundles."""
    studies = load_processed_studies(studies_directory)
    profile_rows, matchado_rows, sun_rows, control_rows = [], [], [], []
    for dataset, (truth, predictions, labels, metadata) in studies.items():
        universe = truth.columns.astype(str).tolist()
        for method in METHODS:
            prediction = predictions[method]
            valid = prediction.sum(axis=1).gt(0)
            scored = samplewise_spearman_on_universe(truth.loc[valid], prediction.loc[valid], universe)
            scored.insert(0, "prediction_method", method)
            scored.insert(0, "dataset", dataset)
            profile_rows.append(scored)

        matchado, _ = matchado_differential_abundance_metrics(
            truth,
            {method: predictions[method] for method in METHODS},
            labels,
            group_a=metadata["group_a"],
            group_b=metadata["group_b"],
            n_permutations=matchado_permutations,
        )
        matchado.insert(0, "dataset", dataset)
        matchado["group_a"], matchado["group_b"] = metadata["group_a"], metadata["group_b"]
        matchado_rows.append(matchado)

        sun, controls, _ = sun_inference_correlation_abundance_randomization(
            truth,
            {method: predictions[method] for method in METHODS},
            labels,
            group_a=metadata["group_a"],
            group_b=metadata["group_b"],
            n_randomizations=sun_randomizations,
        )
        sun.insert(0, "dataset", dataset)
        sun["group_a"], sun["group_b"] = metadata["group_a"], metadata["group_b"]
        sun_rows.append(sun)
        controls.insert(0, "dataset", dataset)
        control_rows.append(controls)

    sources = sorted(Path(studies_directory).glob("*.npz"))
    return (
        pd.concat(profile_rows, ignore_index=True),
        pd.concat(matchado_rows, ignore_index=True),
        pd.concat(sun_rows, ignore_index=True),
        pd.concat(control_rows, ignore_index=True),
        sources,
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    profile, matchado, sun, controls, _ = replay_bundled_group_metrics(root / "data/studies")
    print(f"profiles={profile.shape}; matchado={matchado.shape}; sun={sun.shape}; controls={controls.shape}")
