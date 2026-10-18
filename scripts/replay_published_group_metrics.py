"""Read-only replay of published group metrics on the accepted 10-study cohort.

This module contains source-artifact loading only. Metric logic comes from the
clean ``picrust_benchmark_metrics`` package. It never writes to the source
workspace and never trains or runs a model.
"""

from __future__ import annotations

import json
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("PICRUSt2", "PICRUSt2 + adaptation", "Tax4Fun2")
ADAPTATION_ARM = "picrust2_teacher_only_residual"


def _source_modules(source_root: Path):
    scripts = source_root / "functional_inference_generalization/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    with redirect_stdout(io.StringIO()):
        import evaluate_expanded_all_ko_report as access
    return access


def _new_wgs_paths(source_root: Path) -> dict[str, Path]:
    package = source_root / "outputs/approved_paired_studies"
    return {
        "emp500_saline_sediment": package / "prjeb42019-exact64-approved-paired-study-20260929-r2/native_inputs/wgs_tpm.sample_by_ko.tsv.gz",
        "plastic_biofilm": package / "prjna1195827-exact73-approved-paired-study-20260929-r2/native_inputs/wgs_tpm.sample_by_ko.tsv.gz",
        "olive_xylem_season1": package / "prjna629675-exact24-approved-paired-study-20260929-r1/native_inputs/wgs_exact24/wgs_tpm.sample_by_ko.tsv.gz",
        "olive_xylem_season2": package / "prjna629675-exact24-approved-paired-study-20260929-r1/native_inputs/wgs_exact24/wgs_tpm.sample_by_ko.tsv.gz",
        "skin_B": package / "prjna471898-skinb-exact44-stage5-20261009-r2/native_inputs/wgs_tpm.sample_by_ko.tsv.gz",
        "skin_A": package / "prjna1162089-skina-exact202-stage5-20261011-r1/native_inputs/wgs_tpm.sample_by_ko.tsv.gz",
        "bonneville_salt_flats": package / "prjna522308-exact16-approved-paired-study-20261009-r1/native_inputs/wgs_tpm.sample_by_ko.tsv.gz",
    }


def _new_tax4fun_paths(source_root: Path) -> dict[str, tuple[Path, Path, str | None]]:
    package = source_root / "outputs/approved_paired_studies"
    return {
        "emp500_saline_sediment": (package / "prjeb42019-exact64-approved-paired-study-20260929-r2/native_inputs/tax4fun2.sample_by_ko.tsv.gz", package / "prjeb42019-exact64-approved-paired-study-20260929-r2/crosswalk/sample_crosswalk.tsv", None),
        "plastic_biofilm": (package / "prjna1195827-exact73-approved-paired-study-20260929-r2/native_inputs/tax4fun2.sample_by_ko.tsv.gz", package / "prjna1195827-exact73-approved-paired-study-20260929-r2/crosswalk/sample_crosswalk.tsv", None),
        "olive_xylem_season1": (package / "prjna629675-exact24-approved-paired-study-20260929-r1/native_inputs/season1/tax4fun2.sample_by_ko.tsv.gz", package / "prjna629675-exact24-approved-paired-study-20260929-r1/crosswalk/sample_crosswalk.tsv", "season1"),
        "olive_xylem_season2": (package / "prjna629675-exact24-approved-paired-study-20260929-r1/native_inputs/season2/tax4fun2.sample_by_ko.tsv.gz", package / "prjna629675-exact24-approved-paired-study-20260929-r1/crosswalk/sample_crosswalk.tsv", "season2"),
        "skin_B": (package / "prjna471898-skinb-exact44-stage5-20261009-r2/native_inputs/tax4fun2.sample_by_ko.tsv.gz", package / "prjna471898-skinb-exact44-stage5-20261009-r2/crosswalk/sample_crosswalk.tsv", None),
        "skin_A": (package / "prjna1162089-skina-exact202-stage5-20261011-r1/native_inputs/tax4fun2.sample_by_ko.tsv.gz", package / "prjna1162089-skina-exact202-stage5-20261011-r1/crosswalk/sample_crosswalk.tsv", None),
        "bonneville_salt_flats": (package / "prjna522308-exact16-approved-paired-study-20261009-r1/native_inputs/tax4fun2.sample_by_ko.tsv.gz", package / "prjna522308-exact16-approved-paired-study-20261009-r1/crosswalk/sample_crosswalk.tsv", None),
    }


def _load_tax4fun(path: Path, crosswalk_path: Path, stratum: str | None) -> pd.DataFrame:
    matrix = _load_wgs(path)
    crosswalk = pd.read_csv(crosswalk_path, sep="\t", dtype=str)
    crosswalk = crosswalk.loc[crosswalk.inclusion_status.eq("included")].copy()
    if stratum is not None:
        crosswalk = crosswalk.loc[crosswalk.stratum.eq(stratum)].copy()
    target_column = "biological_sample_id" if "biological_sample_id" in crosswalk else "stage3b_wgs_sample_id"
    mapping = crosswalk.set_index("stage3a_amplicon_sample_id")[target_column].astype(str)
    if not set(mapping.index).issubset(matrix.index):
        raise RuntimeError(f"Tax4Fun2/crosswalk mismatch: {path}")
    matrix = matrix.loc[mapping.index].copy()
    matrix.index = mapping.loc[matrix.index].tolist()
    return matrix


def _load_wgs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", compression="gzip")
    frame = frame.set_index(frame.iloc[:, 0].astype(str)).drop(columns=frame.columns[0])
    frame.columns = frame.columns.astype(str)
    return frame.astype(float)


def _load_prediction(path: Path, datasets: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        if not np.array_equal(bundle["dataset"].astype(str), datasets) or not np.array_equal(bundle["sample_id"].astype(str), samples):
            raise RuntimeError(f"Cached prediction sample order differs: {path}")
        return bundle["profiles"].astype(float), bundle["ko_ids"].astype(str)


def _task_run_dir(tasks: pd.DataFrame, retrieval: Path, fold_name: str) -> Path:
    row = tasks[(tasks.model == ADAPTATION_ARM) & (tasks.fold_name == fold_name)]
    if len(row) != 1:
        raise RuntimeError(f"Expected one cached adaptation task for {fold_name}")
    source_out = json.loads(row.iloc[0].steps)[0]["out_dir"]
    prefix = "results/unified5_package_r7/"
    if not source_out.startswith(prefix):
        raise RuntimeError(f"Unexpected cached-task path: {source_out}")
    return retrieval / source_out[len(prefix):]


def load_accepted_profiles_and_labels(source_root: Path):
    """Load WGS/PICRUSt2/adaptation/Tax4Fun2 profiles and frozen labels."""
    from picrust_benchmark_metrics import align_to_ko_universe, normalize_rows

    source_root = Path(source_root).resolve()
    access = _source_modules(source_root)
    functional = source_root / "functional_inference_generalization"
    results = functional / "results"
    contract = results / "full_loso23_contract_r1/contract"
    retrieval = results / "unified5_package_r7_final_retrieval_r1"
    teacher_dir = results / "legacy4_package_23fold_1639_r1/inputs/teacher_native_1639_r1"
    tasks = pd.read_csv(retrieval / "TASKS_115.tsv", sep="\t")
    outer = pd.read_csv(contract / "outer_group_registry.tsv", sep="\t").sort_values("fold_index")
    strata = pd.read_csv(contract / "study_strata.tsv", sep="\t")

    with redirect_stdout(io.StringIO()):
        profiles, profile_sources = access.load_profile_registry()
    wgs = {dataset: block["WGS"] for dataset, block in profiles.items()}
    tax4fun = {dataset: block["Tax4Fun2"] for dataset, block in profiles.items()}
    new_wgs = _new_wgs_paths(source_root)
    new_tax4fun = _new_tax4fun_paths(source_root)
    for dataset, path in new_wgs.items():
        wgs[dataset] = _load_wgs(path)
    for dataset, (path, crosswalk_path, stratum) in new_tax4fun.items():
        tax4fun[dataset] = _load_tax4fun(path, crosswalk_path, stratum)
    if set(wgs) != set(strata.dataset) or set(tax4fun) != set(strata.dataset):
        raise RuntimeError("WGS/Tax4Fun2 registry is not the accepted 26-stratum contract")

    teacher_path = teacher_dir / "teacher_native.npz"
    with np.load(teacher_path, allow_pickle=False) as bundle:
        native = {
            dataset: pd.DataFrame(
                bundle[f"values__{dataset}"],
                index=bundle[f"samples__{dataset}"].astype(str),
                columns=bundle[f"kos__{dataset}"].astype(str),
            )
            for dataset in strata.dataset
        }

    cache: dict[str, dict[str, pd.DataFrame]] = {}
    raw_cache: dict[str, dict[str, pd.DataFrame]] = {}
    prediction_paths = []
    for fold_name in outer.fold_name:
        with np.load(contract / f"fold_targets/{fold_name}.npz", allow_pickle=False) as target:
            test_dataset = target["test_dataset"].astype(str)
            test_sample = target["test_sample_id"].astype(str)
        prediction_path = _task_run_dir(tasks, retrieval, fold_name) / "predictions.npz"
        values, ko_ids = _load_prediction(prediction_path, test_dataset, test_sample)
        prediction_paths.append(prediction_path)
        for dataset in dict.fromkeys(test_dataset):
            positions = np.flatnonzero(test_dataset == dataset)
            sample_ids = test_sample[positions].tolist()
            truth = wgs[dataset].loc[sample_ids]
            universe = sorted(truth.columns[truth.sum(axis=0).gt(0)].astype(str))
            raw = {
                "WGS": truth,
                "PICRUSt2": native[dataset].loc[sample_ids],
                "PICRUSt2 + adaptation": pd.DataFrame(values[positions], index=sample_ids, columns=ko_ids),
                "Tax4Fun2": tax4fun[dataset].loc[sample_ids],
            }
            raw_cache[dataset] = raw
            aligned = {}
            for method, frame in raw.items():
                matrix = align_to_ko_universe(frame, universe, require_all=(method == "WGS"))
                relative, totals = normalize_rows(matrix)
                if (totals <= 0).any() and method != "Tax4Fun2":
                    raise RuntimeError(f"Zero-total row for {dataset}/{method}")
                aligned[method] = relative.fillna(0.0)
            cache[dataset] = aligned

    configs, roles, label_sources = access.load_effect_configs(cache)
    provenance = [
        contract / "outer_group_registry.tsv", contract / "study_strata.tsv",
        retrieval / "TASKS_115.tsv", teacher_path,
        *profile_sources, *new_wgs.values(),
        *(item for values in new_tax4fun.values() for item in values[:2]),
        *prediction_paths, *label_sources,
    ]
    return raw_cache, configs, roles, strata, outer, list(dict.fromkeys(map(Path, provenance)))


def replay_published_group_metrics(
    source_root: Path,
    *,
    matchado_permutations: int = 20,
    sun_randomizations: int = 100,
):
    """Run profile plus published-definition Matchado and Sun metrics."""
    from picrust_benchmark_metrics import (
        matchado_differential_abundance_metrics,
        samplewise_spearman_on_universe,
        sun_inference_correlation_abundance_randomization,
    )

    cache, configs, roles, strata, outer, provenance = load_accepted_profiles_and_labels(source_root)
    stratum_lookup = strata.set_index("dataset")
    profile_meta_path = Path(source_root) / (
        "functional_inference_generalization/results/"
        "unified5_loso23_e3_six_method_comparison_r1/analysis_r1/sample_spearman_long.tsv.gz"
    )
    profile_meta = pd.read_csv(
        profile_meta_path, sep="\t", usecols=["dataset", "heldout_group", "fold_name", "display_name"]
    ).drop_duplicates("dataset").set_index("dataset")
    provenance.append(profile_meta_path)
    profile_rows, coverage_rows = [], []
    for dataset, matrices in cache.items():
        truth, prediction = matrices["WGS"], matrices["Tax4Fun2"]
        universe = sorted(truth.columns[truth.sum(axis=0).gt(0)].astype(str))
        tax_on_universe = prediction.reindex(columns=universe, fill_value=0.0)
        valid = tax_on_universe.sum(axis=1).gt(0)
        scored = samplewise_spearman_on_universe(truth.loc[valid], prediction.loc[valid], universe)
        info, meta = stratum_lookup.loc[dataset], profile_meta.loc[dataset]
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
        scored["surface_id"] = "clean_tax4fun2_wgs_universe_replay"
        profile_rows.append(scored)
        overlap = len(set(universe) & set(prediction.columns.astype(str)))
        coverage_rows.append({"dataset": dataset, "heldout_group": meta.heldout_group,
                              "method": "Tax4Fun2", "n_samples": len(scored),
                              "n_zero_total_samples": int((~valid).sum()),
                              "n_wgs_universe_kos": len(universe), "n_model_head_kos": prediction.shape[1],
                              "n_universe_kos_in_head": overlap,
                              "n_zero_filled_universe_kos": len(universe) - overlap,
                              "fraction_universe_in_head": overlap / len(universe)})

    matchado_rows, sun_rows, sun_randomization_rows = [], [], []
    for dataset, config in configs.items():
        role = roles.loc[roles.dataset.eq(dataset), ["sample_id", "group_label"]].drop_duplicates()
        labels = role.set_index("sample_id").group_label.astype(str)
        matrices = {name: cache[dataset][name].loc[labels.index] for name in ("WGS", *METHODS)}
        matchado, _ = matchado_differential_abundance_metrics(
            matrices["WGS"], {name: matrices[name] for name in METHODS}, labels,
            group_a=config["group_a"], group_b=config["group_b"],
            n_permutations=matchado_permutations,
        )
        matchado.insert(0, "dataset", dataset)
        matchado["group_a"], matchado["group_b"] = config["group_a"], config["group_b"]
        matchado_rows.append(matchado)

        sun, randomizations, _ = sun_inference_correlation_abundance_randomization(
            matrices["WGS"], {name: matrices[name] for name in METHODS}, labels,
            group_a=config["group_a"], group_b=config["group_b"],
            n_randomizations=sun_randomizations,
        )
        sun.insert(0, "dataset", dataset)
        sun["group_a"], sun["group_b"] = config["group_a"], config["group_b"]
        sun_rows.append(sun)
        randomizations.insert(0, "dataset", dataset)
        sun_randomization_rows.append(randomizations)
    return (
        pd.concat(profile_rows, ignore_index=True), pd.DataFrame(coverage_rows),
        pd.concat(matchado_rows, ignore_index=True), pd.concat(sun_rows, ignore_index=True),
        pd.concat(sun_randomization_rows, ignore_index=True), provenance,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    profile, coverage, matchado, sun, randomizations, _ = replay_published_group_metrics(args.source_root)
    print(f"Tax4Fun2 profiles: {profile.shape}; coverage: {coverage.shape}; "
          f"Matchado: {matchado.shape}; Sun: {sun.shape}; controls: {randomizations.shape}")
