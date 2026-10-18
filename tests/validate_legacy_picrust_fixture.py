#!/usr/bin/env python3
"""Reproduce one completed legacy PICRUSt2 sample-profile result, read-only.

This deliberately reads only a single source sample column and its WGS row.
It neither writes to the source project nor processes a dataset. The expected
value is the completed output of notebook 03 in the source project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from picrust_benchmark_metrics import samplewise_spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sample", default="10Dr2")
    args = parser.parse_args()
    root, sample = args.source_root, args.sample
    mapping = pd.read_csv(root / "eq_benchmark_data/functional/picrust2/sample_mapping.tsv", sep="\t", dtype=str)
    source_id = mapping.loc[mapping.sample_id.eq(sample), "picrust2_id"].iloc[0]
    expected = pd.read_csv(root / "outputs/exploratory_spearman/picrust2_vs_wgs_spearman_per_sample.tsv", sep="\t")
    expected_value = float(expected.loc[expected.sample_id.eq(sample), "spearman_r"].iloc[0])
    # Notebook 03 loads source tables as strings before coercing values; retain
    # that parse path so this is a literal reproduction rather than a close one.
    wgs = pd.read_csv(root / "outputs/sample_KO_counts.tsv", sep="\t", dtype=str)
    wgs = wgs.loc[wgs.sample_id.eq(sample)].drop(columns=["sample_key", "trip", "site", "sample_id"]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    wgs.index = [sample]
    prediction = pd.read_csv(root / "eq_benchmark_data/functional/picrust2/ko_pred_metagenome_unstrat.tsv", sep="\t", usecols=["function", source_id], dtype=str)
    prediction["function"] = prediction["function"].str.removeprefix("ko:")
    prediction = prediction.loc[prediction.function.str.fullmatch(r"K\d{5}", na=False)].set_index("function").T.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    prediction.index = [sample]
    shared = sorted(set(wgs.columns) & set(prediction.columns))
    actual = float(samplewise_spearman(wgs.loc[:, shared], prediction.loc[:, shared]).loc[0, "spearman_r"])
    # Notebook output was created in a different local software environment.
    # Spearman agrees to 11+ decimal places; use a strict numerical tolerance,
    # not a brittle bitwise comparison across SciPy/pandas versions.
    if not np.isclose(actual, expected_value, rtol=0.0, atol=1e-10):
        raise SystemExit(f"FAIL {sample}: expected={expected_value:.17g}, actual={actual:.17g}")
    print(f"PASS {sample}: expected={expected_value:.17g}; actual={actual:.17g}; absolute_difference={abs(actual - expected_value):.3g}; shared KOs={len(shared)}")


if __name__ == "__main__":
    main()
