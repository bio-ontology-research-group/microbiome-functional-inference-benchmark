#!/usr/bin/env python3
"""Build the compact processed-data bundle committed with this repository.

This maintainer script reads the frozen source workspace and writes only the
processed inputs and completed reference tables used by the public notebooks.
It never modifies the source workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from picrust_benchmark_metrics import samplewise_spearman_on_universe
from replay_published_group_metrics import load_accepted_profiles_and_labels


REFERENCE_FILES = {
    "sample_spearman_long.tsv.gz": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/sample_spearman_long.tsv.gz",
    "stratum_summary.tsv": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/stratum_summary.tsv",
    "group_summary.tsv": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/group_summary.tsv",
    "coverage_by_stratum_method.tsv": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/coverage_by_stratum_method.tsv",
    "surface_summary.tsv": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/surface_summary.tsv",
    "effect_cohort.tsv": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/effect_cohort.tsv",
    "effect_selection_fixed.tsv": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/effect_selection_fixed.tsv",
    "quantile_per_ko_assignments.tsv.gz": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/quantile_per_ko_assignments.tsv.gz",
    "effect_long.tsv": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/effect_long.tsv",
    "classification_summary.tsv": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/classification_summary.tsv",
    "quantile_composition_by_bin.tsv": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/quantile_composition_by_bin.tsv",
    "quantile_study_baselines.tsv": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/analysis_r1/quantile_study_baselines.tsv",
    "pr_roc_metric_summary.tsv": "functional_inference_generalization/results/unified5_loso23_e3_six_method_comparison_r1/figures_r1/pr_roc_metric_summary.tsv",
    "study_strata.tsv": "functional_inference_generalization/results/full_loso23_contract_r1/contract/study_strata.tsv",
    "matchado_label_metric_results_long.tsv": "functional_inference_generalization/results/matchado_label_metric_results_long.tsv",
    "tax4fun2_completed_reference.tsv.gz": "functional_inference_generalization/results/foundation_models_completed_all_ko_dual_loss_grouped_loso_r1/analysis_r2/sample_spearman_per_sample.tsv.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_study_bundle(path: Path, dataset: str, matrices, labels: pd.Series, config: dict) -> tuple[int, int]:
    full_truth = matrices["WGS"]
    universe = np.asarray(full_truth.columns[full_truth.sum(axis=0).gt(0)].astype(str), dtype=str)
    truth = full_truth.loc[labels.index]
    np.savez_compressed(
        path,
        sample_ids=np.asarray(truth.index.astype(str), dtype=str),
        ko_ids=universe,
        labels=np.asarray(labels.astype(str), dtype=str),
        wgs=truth.reindex(columns=universe, fill_value=0.0).to_numpy(),
        picrust2=matrices["PICRUSt2"].loc[labels.index].reindex(columns=universe, fill_value=0.0).to_numpy(),
        picrust2_adaptation=matrices["PICRUSt2 + adaptation"].loc[labels.index].reindex(columns=universe, fill_value=0.0).to_numpy(),
        tax4fun2=matrices["Tax4Fun2"].loc[labels.index].reindex(columns=universe, fill_value=0.0).to_numpy(),
        metadata_json=np.asarray(json.dumps({"dataset": dataset, **config}, sort_keys=True)),
    )
    return len(labels), len(universe)


def tax4fun2_profile_tables(cache, strata: pd.DataFrame, profile_meta: pd.DataFrame):
    rows, coverage = [], []
    lookup = strata.set_index("dataset")
    metadata = profile_meta.drop_duplicates("dataset").set_index("dataset")
    for dataset, matrices in cache.items():
        truth, prediction = matrices["WGS"], matrices["Tax4Fun2"]
        universe = sorted(truth.columns[truth.sum(axis=0).gt(0)].astype(str))
        aligned = prediction.reindex(columns=universe, fill_value=0.0)
        valid = aligned.sum(axis=1).gt(0)
        scored = samplewise_spearman_on_universe(truth.loc[valid], prediction.loc[valid], universe)
        meta = metadata.loc[dataset]
        scored.insert(0, "dataset", dataset)
        scored["heldout_group"] = meta.heldout_group
        scored["fold_name"] = meta.fold_name
        scored["display_name"] = meta.display_name
        scored["method_order"] = 3
        scored["method_label"] = "Tax4Fun2"
        scored["arm"] = "native_tax4fun2"
        scored["seed"] = -1
        scored["sample_spearman"] = scored.spearman_r
        scored["n_wgs_universe_kos"] = len(universe)
        scored["n_model_head_kos"] = prediction.shape[1]
        scored["surface_id"] = "public_tax4fun2_wgs_universe_replay"
        rows.append(scored)
        overlap = len(set(universe) & set(prediction.columns.astype(str)))
        coverage.append({
            "dataset": dataset,
            "heldout_group": meta.heldout_group,
            "method": "Tax4Fun2",
            "n_samples": len(scored),
            "n_zero_total_samples": int((~valid).sum()),
            "n_wgs_universe_kos": len(universe),
            "n_model_head_kos": prediction.shape[1],
            "n_universe_kos_in_head": overlap,
            "n_zero_filled_universe_kos": len(universe) - overlap,
            "fraction_universe_in_head": overlap / len(universe),
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coverage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    studies_dir, reference_dir = output / "studies", output / "reference"
    studies_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    cache, configs, roles, strata, _, source_files = load_accepted_profiles_and_labels(source)
    manifest_rows = []
    for dataset, config in configs.items():
        label_rows = roles.loc[roles.dataset.eq(dataset), ["sample_id", "group_label"]].drop_duplicates()
        labels = label_rows.set_index("sample_id").group_label.astype(str)
        destination = studies_dir / f"{dataset}.npz"
        n_samples, n_kos = write_study_bundle(destination, dataset, cache[dataset], labels, config)
        manifest_rows.append({
            "file": destination.relative_to(output).as_posix(), "kind": "processed_study_bundle",
            "dataset": dataset, "n_samples": n_samples, "n_kos": n_kos,
            "group_a": config["group_a"], "group_b": config["group_b"],
            "bytes": destination.stat().st_size, "sha256": sha256(destination),
            "source": "accepted frozen WGS/native predictions/basic-MLP predictions/Tax4Fun2 profiles and audited labels",
        })

    for destination_name, source_name in REFERENCE_FILES.items():
        source_path = source / source_name
        destination = reference_dir / destination_name
        if destination_name == "matchado_label_metric_results_long.tsv":
            legacy = pd.read_csv(source_path, sep="\t")
            legacy = legacy[
                legacy.prediction_method.isin(["PICRUSt2", "Tax4Fun2"])
                & legacy.metric.isin(["precision", "recall", "f1"])
            ]
            legacy.to_csv(destination, sep="\t", index=False)
        else:
            shutil.copy2(source_path, destination)
        manifest_rows.append({
            "file": destination.relative_to(output).as_posix(), "kind": "completed_reference_table",
            "dataset": "", "n_samples": "", "n_kos": "", "group_a": "", "group_b": "",
            "bytes": destination.stat().st_size, "sha256": sha256(destination), "source": source_name,
        })

    profile_meta = pd.read_csv(reference_dir / "sample_spearman_long.tsv.gz", sep="\t",
                               usecols=["dataset", "heldout_group", "fold_name", "display_name"])
    tax4fun_profile, tax4fun_coverage = tax4fun2_profile_tables(cache, strata, profile_meta)
    for frame, name in ((tax4fun_profile, "tax4fun2_sample_spearman.tsv.gz"),
                        (tax4fun_coverage, "tax4fun2_coverage.tsv")):
        destination = reference_dir / name
        frame.to_csv(destination, sep="\t", index=False, compression="gzip" if name.endswith(".gz") else None)
        manifest_rows.append({
            "file": destination.relative_to(output).as_posix(), "kind": "recomputed_reference_table",
            "dataset": "all_26_strata", "n_samples": "", "n_kos": "", "group_a": "", "group_b": "",
            "bytes": destination.stat().st_size, "sha256": sha256(destination),
            "source": "recomputed from the frozen source matrices with picrust_benchmark_metrics",
        })

    manifest = pd.DataFrame(manifest_rows).sort_values(["kind", "file"])
    manifest.to_csv(output / "manifest.tsv", sep="\t", index=False)
    source_manifest = pd.DataFrame({
        "source_path": [path.relative_to(source).as_posix() for path in dict.fromkeys(source_files)],
        "bytes": [path.stat().st_size for path in dict.fromkeys(source_files)],
        "sha256": [sha256(path) for path in dict.fromkeys(source_files)],
    })
    source_manifest.to_csv(output / "source_artifact_manifest.tsv", sep="\t", index=False)
    print(f"Wrote {len(configs)} study bundles and {len(manifest) - len(configs)} reference tables to {output}")
    print(f"Total public data bytes: {sum(path.stat().st_size for path in output.rglob('*') if path.is_file()):,}")


if __name__ == "__main__":
    main()
