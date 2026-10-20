#!/usr/bin/env python3
"""Validate combined_benchmark structure, provenance, and numerical replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"
TOLERANCE = 1e-12


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def max_abs(left: pd.Series, right: pd.Series) -> float:
    a, b = left.to_numpy(float), right.to_numpy(float)
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return np.inf
    return float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0


def main() -> None:
    receipt = json.loads((HERE / "RUN_COMPLETE.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["training_performed"] is False
    assert receipt["paired_samples"] == 1717
    assert receipt["benchmark_source_samples"] == 621
    assert receipt["expansion_samples"] == 1096
    assert receipt["unique_source_projects"] == 24
    assert receipt["source_project_environment_scopes"] == 29
    assert receipt["accepted_biological_contrasts"] == 17

    profile = pd.read_csv(RESULTS / "combined_profile_sample_long.tsv.gz", sep="\t")
    registry = pd.read_csv(RESULTS / "combined_sample_environment_registry.tsv.gz", sep="\t")
    scope = pd.read_csv(RESULTS / "scope_profile_summary.tsv", sep="\t")
    contrasts = pd.read_csv(RESULTS / "combined_contrast_registry.tsv", sep="\t")
    assert len(profile) == 3434 and len(registry) == 1717
    assert profile.sample_key.nunique() == 1717
    assert profile.groupby("sample_key").method_label.nunique().eq(2).all()
    assert registry.source_project_id.nunique() == 24
    assert registry.environment_scope_id.nunique() == 29
    assert len(scope) == 58 and len(contrasts) == 17
    assert set(profile.method_label) == {"PICRUSt2", "Tax4Fun2"}
    assert profile.sample_spearman.isna().sum() == 4
    assert set(profile.loc[profile.sample_spearman.isna(), "status"]) == {"zero_method_total_on_wgs_universe"}
    assert set(registry.loc[registry.dataset.eq("blueberry"), "empo_3"]) == {"Unknown (contradictory)"}
    assert set(registry.loc[registry.dataset.eq("topsoil"), "empo_2"]) == {"Non-saline"}
    assert set(registry.loc[registry.dataset.eq("han_river"), "empo_2"]) == {"Non-saline"}

    # Completed profile tables must be copied numerically, not approximately reconstructed.
    base_ids = set(registry.loc[registry.provenance.eq("benchmark_source"), "dataset"])
    base_pic = pd.read_csv(REPO / "data/reference/sample_spearman_long.tsv.gz", sep="\t")
    base_pic = base_pic.loc[base_pic.dataset.isin(base_ids) & base_pic.method_label.eq("PICRUSt2")]
    got_pic = profile.loc[profile.provenance.eq("benchmark_source") & profile.method_label.eq("PICRUSt2")]
    check = got_pic.merge(base_pic, on=["dataset", "sample_id"], suffixes=("_got", "_ref"), validate="one_to_one")
    profile_pic_diff = max_abs(check.sample_spearman_got, check.sample_spearman_ref)

    base_tax = pd.read_csv(REPO / "data/reference/tax4fun2_sample_spearman.tsv.gz", sep="\t")
    base_tax = base_tax.loc[base_tax.dataset.isin(base_ids)]
    got_tax = profile.loc[profile.provenance.eq("benchmark_source") & profile.method_label.eq("Tax4Fun2")]
    check = got_tax.merge(base_tax, on=["dataset", "sample_id"], suffixes=("_got", "_ref"), validate="one_to_one")
    profile_tax_diff = max_abs(check.sample_spearman_got, check.sample_spearman_ref)

    exp_ref = pd.read_csv(REPO / "expansion_benchmark/results/profile_sample_spearman_wgs_universe.tsv.gz", sep="\t")
    got_exp = profile.loc[profile.provenance.eq("expansion")]
    check = got_exp.merge(exp_ref, on=["dataset", "sample_id", "method_label"], suffixes=("_got", "_ref"), validate="one_to_one")
    profile_exp_diff = max_abs(check.sample_spearman_got, check.sample_spearman_ref)
    assert max(profile_pic_diff, profile_tax_diff, profile_exp_diff) <= TOLERANCE

    # The recomputed original PICRUSt2 project extensions must reproduce their
    # completed references before accepting the newly added Tax4Fun2 rows.
    effect = pd.read_csv(RESULTS / "combined_effect.tsv", sep="\t")
    effect_got = effect.loc[effect.provenance.eq("benchmark_source") & effect.prediction_method.eq("PICRUSt2")]
    effect_ref = pd.read_csv(REPO / "data/reference/effect_long.tsv", sep="\t")
    effect_ref = effect_ref.loc[
        effect_ref.dataset.isin(base_ids) & effect_ref.method_label.eq("PICRUSt2") & effect_ref.metric.eq("spearman")
    ]
    check = effect_got.merge(effect_ref, on="dataset", validate="one_to_one")
    effect_diff = max_abs(check.spearman_effect_wgs_bh_selected, check.effect_correlation)

    directional = pd.read_csv(RESULTS / "combined_directional.tsv", sep="\t")
    dir_got = directional.loc[
        directional.provenance.eq("benchmark_source") & directional.prediction_method.eq("PICRUSt2")
    ]
    dir_ref = pd.read_csv(REPO / "data/reference/pr_roc_metric_summary.tsv", sep="\t")
    dir_ref = dir_ref.loc[dir_ref.dataset.isin(base_ids) & dir_ref.prediction_method.eq("PICRUSt2")]
    check = dir_got.merge(dir_ref, on=["dataset", "direction"], suffixes=("_got", "_ref"), validate="one_to_one")
    roc_diff = max_abs(check.roc_auc, check.roc_auc_replay)
    ap_diff = max_abs(check.ap, check.ap_replay)
    assert max(effect_diff, roc_diff, ap_diff) <= TOLERANCE

    # Exact row coverage. Tax4Fun2 is unavailable only for olive season 1 on
    # group-level surfaces, and the absence remains explicit in source tables.
    expected = {
        "combined_effect.tsv": {"PICRUSt2": 17, "Tax4Fun2": 16},
        "combined_directional.tsv": {"PICRUSt2": 34, "Tax4Fun2": 32},
        "combined_bh.tsv": {"PICRUSt2": 17, "Tax4Fun2": 16},
        "combined_sun.tsv": {"PICRUSt2": 14, "Tax4Fun2": 13},
    }
    for filename, counts in expected.items():
        table = pd.read_csv(RESULTS / filename, sep="\t")
        assert table.groupby("prediction_method").size().to_dict() == counts

    for name in [
        "01_environment_coverage_census.png", "02_scope_profile_dumbbell.png",
        "03_scope_method_difference.png", "04_environment_macro_profile.png",
        "05_matchado_raw_p_f1.png", "06_sun_inference_correlation.png",
        "07_effect_relationship.png", "08_directional_roc_auc.png",
    ]:
        path = HERE / "figures" / name
        assert path.exists() and path.stat().st_size > 20_000

    inputs = pd.read_csv(RESULTS / "input_manifest.tsv", sep="\t")
    for row in inputs.itertuples(index=False):
        path = REPO / row.path
        assert path.exists() and digest(path) == row.sha256

    outputs = pd.read_csv(HERE / "OUTPUT_MANIFEST.tsv", sep="\t")
    for row in outputs.itertuples(index=False):
        path = HERE / row.path
        assert path.exists() and digest(path) == row.sha256

    report = {
        "status": "PASS",
        "tolerance": TOLERANCE,
        "max_abs_diff_profile_benchmark_picrust2": profile_pic_diff,
        "max_abs_diff_profile_benchmark_tax4fun2": profile_tax_diff,
        "max_abs_diff_profile_expansion": profile_exp_diff,
        "max_abs_diff_effect_picrust2_reference": effect_diff,
        "max_abs_diff_directional_roc_picrust2_reference": roc_diff,
        "max_abs_diff_directional_ap_picrust2_reference": ap_diff,
        "input_hashes_verified": len(inputs),
        "output_hashes_verified": len(outputs),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
