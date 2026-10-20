#!/usr/bin/env python3
"""Run the Table 2.2 completed-expansion benchmark without fitting models.

The established repository is imported read-only.  Source matrices are read
from the research workspace and every generated artifact is written below
``expansion_benchmark/``.  Only native PICRUSt2 and Tax4Fun2 predictions are
evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCE = Path(
    os.environ.get(
        "PICRUST_BENCHMARK_SOURCE_ROOT",
        "/home/hernan_melmoth/Documents/phd_work/functions_inference_benchmark_EQ",
    )
).resolve()
sys.path.insert(0, str(REPO / "src"))

from picrust_benchmark_metrics import (  # noqa: E402
    benjamini_hochberg,
    biological_group_effect_concordance,
    classification_set_metrics,
    decile_outcome_composition,
    directional_ranking_curves,
    matchado_bh_shared_wgs_reference_metrics,
    matchado_differential_abundance_metrics,
    normalize_rows,
    per_ko_spearman,
    per_sample_richness_overlap,
    plot_directional_metric_summary,
    plot_directional_roc_grid,
    plot_ko_coverage,
    plot_matchado_randomization_dumbbells,
    plot_metric_row_grid,
    plot_profile_study_panels,
    plot_signed_decile_grid,
    plot_sun_observed_controls,
    rank_biserial_effect_by_ko,
    samplewise_profile_metrics,
    samplewise_spearman,
    samplewise_spearman_on_universe,
    sun_inference_correlation_abundance_randomization,
    wilcoxon_rank_sum_by_ko,
)

METHODS = ("PICRUSt2", "Tax4Fun2")
KO_RE = re.compile(r"(?:ko:)?(K\d{5})$")
RT = {"sep": "\t", "index": False, "float_format": "%.17g"}


@dataclass(frozen=True)
class StudySpec:
    dataset: str
    display_name: str
    accession: str
    empo: str
    expected_samples: int
    package: str
    wgs: tuple[str, ...]
    picrust2: tuple[str, ...]
    tax4fun2: tuple[str, ...]
    crosswalk: str
    canonical_column: str
    wgs_column: str
    prediction_column: str
    status: str = "completed_available_black_table_2_2"
    loader: str = "standard"


def _t(*values: str) -> tuple[str, ...]:
    return tuple(values)


STUDIES = (
    StudySpec(
        "han_river", "Han River", "PRJNA530373", "Water (non-saline)", 45,
        "outputs/approved_paired_studies/han-river45-approved-paired-study-20260913-r5",
        _t("original_matrices/wgs_tpm.sample_by_ko.tsv.gz"),
        _t("original_matrices/picrust2.sample_by_ko.tsv.gz"),
        _t("original_matrices/tax4fun2.sample_by_ko.tsv.gz"),
        "crosswalk/sample_crosswalk.tsv", "biological_sample_id", "biological_sample_id", "analysis_id",
    ),
    StudySpec(
        "topsoil", "Topsoil", "PRJEB10725", "Soil (non-saline)", 18,
        "outputs/approved_paired_studies/prjeb10725-soil-exact18-approved-paired-study-20260914-r1",
        _t("native_matrices/exact18/wgs_tpm.sample_by_ko.tsv.gz"),
        _t("native_matrices/miseq_exact11/picrust2.sample_by_ko.tsv.gz", "native_matrices/hiseq_exact7/picrust2.sample_by_ko.tsv.gz"),
        _t("native_matrices/miseq_exact11/tax4fun2.sample_by_ko.tsv.gz", "native_matrices/hiseq_exact7/tax4fun2.sample_by_ko.tsv.gz"),
        "crosswalk/sample_crosswalk.tsv", "biological_sample_id", "wgs_sample_id", "amplicon_native_sample_id",
    ),
    StudySpec(
        "empty_quarter", "Empty Quarter", "project dataset", "Soil (non-saline)", 343,
        ".", _t("outputs/sample_KO_relative_abundance.tsv"),
        _t("eq_benchmark_data/functional/picrust2/ko_pred_metagenome_unstrat.tsv"),
        _t("eq_benchmark_data/functional/tax4fun2/ko_pred_metagenome_unstrat.tsv"),
        "eq_benchmark_data/functional/picrust2/sample_mapping.tsv", "sample_id", "sample_id", "picrust2_id",
        loader="empty_quarter",
    ),
    StudySpec(
        "osd_ocean", "Ocean Sampling Day", "PRJEB8682", "Water (saline)", 60,
        "outputs/approved_paired_studies/prjeb8682-exact60-approved-paired-study-20260916-r1",
        _t("native_inputs/wgs_tpm.sample_by_ko.tsv.gz"), _t("native_inputs/picrust2.sample_by_ko.tsv.gz"),
        _t("native_inputs/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "biological_sample_id", "stage3b_wgs_sample_id", "stage3a_amplicon_sample_id",
    ),
    StudySpec(
        "bonneville_salt_flats", "Bonneville Salt Flats", "PRJNA522308", "Hypersaline (saline)", 16,
        "outputs/approved_paired_studies/prjna522308-exact16-approved-paired-study-20261009-r1",
        _t("native_inputs/wgs_tpm.sample_by_ko.tsv.gz"), _t("native_inputs/picrust2.sample_by_ko.tsv.gz"),
        _t("native_inputs/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "biological_sample_id", "stage3b_wgs_sample_id", "stage3a_amplicon_sample_id",
    ),
    StudySpec(
        "plastic_biofilm", "Plastic biofilm", "PRJNA1195827", "Surface (saline)", 73,
        "outputs/approved_paired_studies/prjna1195827-exact73-approved-paired-study-20260929-r2",
        _t("native_inputs/wgs_tpm.sample_by_ko.tsv.gz"), _t("native_inputs/picrust2.sample_by_ko.tsv.gz"),
        _t("native_inputs/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "biological_sample_id", "stage3b_wgs_sample_id", "stage3a_amplicon_sample_id",
    ),
    StudySpec(
        "oral_plaque", "Oral plaque", "PRJEB98488", "Animal proximal gut", 28,
        "studies/PRJEB98488/approved_paired_studies/prjeb98488-exact28-approved-paired-study-20260913-r1",
        _t("native_matrices/wgs_tpm.sample_by_ko.tsv.gz"), _t("native_matrices/picrust2.sample_by_ko.tsv.gz"),
        _t("native_matrices/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "biological_sample_id", "biological_sample_id", "biological_sample_id",
    ),
    StudySpec(
        "skin_ctcl", "Human skin CTCL", "PRJEB36953", "Animal surface", 36,
        "outputs/approved_paired_studies/prjeb36953-exact36-approved-paired-study-20260919-r1",
        _t("native_inputs/wgs_tpm.sample_by_ko.tsv.gz"), _t("native_inputs/picrust2.sample_by_ko.tsv.gz"),
        _t("native_inputs/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "biological_sample_id", "stage3b_wgs_sample_id", "stage3a_amplicon_sample_id",
    ),
    StudySpec(
        "skin_A", "Human skin A", "PRJNA1162089", "Animal surface", 202,
        "outputs/approved_paired_studies/prjna1162089-skina-exact202-stage5-20261011-r1",
        _t("native_inputs/wgs_tpm.sample_by_ko.tsv.gz"), _t("native_inputs/picrust2.sample_by_ko.tsv.gz"),
        _t("native_inputs/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "pair_id", "stage3b_wgs_sample_id", "stage3a_amplicon_sample_id",
    ),
    StudySpec(
        "skin_B", "Human skin B", "PRJNA471898", "Animal surface", 44,
        "outputs/approved_paired_studies/prjna471898-skinb-exact44-stage5-20261009-r2",
        _t("native_inputs/wgs_tpm.sample_by_ko.tsv.gz"), _t("native_inputs/picrust2.sample_by_ko.tsv.gz"),
        _t("native_inputs/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "pair_id", "stage3b_wgs_sample_id", "stage3a_amplicon_sample_id",
    ),
    StudySpec(
        "olive_xylem", "Olive xylem", "PRJNA629675", "Plant corpus", 24,
        "outputs/approved_paired_studies/prjna629675-exact24-approved-paired-study-20260929-r1",
        _t("native_inputs/wgs_exact24/wgs_tpm.sample_by_ko.tsv.gz"),
        _t("native_inputs/season1/picrust2.sample_by_ko.tsv.gz", "native_inputs/season2/picrust2.sample_by_ko.tsv.gz"),
        _t("native_inputs/season1/tax4fun2.sample_by_ko.tsv.gz", "native_inputs/season2/tax4fun2.sample_by_ko.tsv.gz"),
        "crosswalk/sample_crosswalk.tsv", "biological_sample_id", "stage3b_wgs_sample_id", "stage3a_amplicon_sample_id",
    ),
    StudySpec(
        "activated_sludge", "Activated sludge", "PRJNA484416", "EMPO leaf pending", 60,
        "local_study_archive/PRJNA484416/payload/results/approved_paired_studies/activated-sludge60-r1-b341f-approved-paired-study-20260828-r2",
        _t("harmonized/native_space/wgs_tpm.sample_by_ko.tsv.gz"),
        _t("harmonized/native_space/picrust2.sample_by_ko.tsv.gz"),
        _t("harmonized/native_space/tax4fun2.sample_by_ko.tsv.gz"),
        "crosswalk/sample_crosswalk.tsv", "biological_sample_id", "biological_sample_id", "biological_sample_id",
    ),
    StudySpec(
        "bioreactor", "Engineered bioreactor", "PRJEB36484", "EMPO leaf pending", 9,
        "outputs/approved_paired_studies/prjeb36484-exact9-approved-paired-study-20260915-r1",
        _t("native_inputs/wgs_tpm.sample_by_ko.tsv.gz"), _t("native_inputs/picrust2.sample_by_ko.tsv.gz"),
        _t("native_inputs/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "biological_sample_id", "stage3b_sample_id", "stage3a_sample_id",
    ),
    StudySpec(
        "emp500_nonsaline_sediment", "EMP500 lake/stream sediment", "PRJEB42019", "Sediment (non-saline)", 37,
        "outputs/approved_paired_studies/prjeb42019-nonsalinesediment-exact37-approved-paired-study-20261020-r1",
        _t("native/wgs_tpm.sample_by_ko.tsv.gz"), _t("native/picrust2.sample_by_ko.tsv.gz"),
        _t("native/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "biosample", "biosample", "amplicon_sample_id",
    ),
    StudySpec(
        "emp500_saline_sediment", "EMP500 marine sediment", "PRJEB42019", "Sediment (saline)", 64,
        "outputs/approved_paired_studies/prjeb42019-exact64-approved-paired-study-20260929-r2",
        _t("native_inputs/wgs_tpm.sample_by_ko.tsv.gz"), _t("native_inputs/picrust2.sample_by_ko.tsv.gz"),
        _t("native_inputs/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "biological_sample_id", "stage3b_wgs_sample_id", "stage3a_amplicon_sample_id",
    ),
    StudySpec(
        "emp500_shipworm", "EMP500 shipworm tissue", "PRJEB42019", "Animal corpus", 14,
        "outputs/approved_paired_studies/prjeb42019-animalcorpus-exact14-approved-paired-study-20261020-r1",
        _t("native/wgs_tpm.sample_by_ko.tsv.gz"), _t("native/picrust2.sample_by_ko.tsv.gz"),
        _t("native/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "biosample", "biosample", "amplicon_sample_id",
    ),
    StudySpec(
        "emp500_pitcher_plant", "EMP500 pitcher plant fluid", "PRJEB42019", "Plant surface", 13,
        "outputs/approved_paired_studies/prjeb42019-plantsurface-exact13-approved-paired-study-20261020-r1",
        _t("native_inputs/wgs_tpm.sample_by_ko.tsv.gz"), _t("native_inputs/picrust2.sample_by_ko.tsv.gz"),
        _t("native_inputs/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "biological_sample_id", "biological_sample_id", "amplicon_sample_id",
    ),
    StudySpec(
        "emp500_lichen", "EMP500 lichen thallus", "PRJEB42019", "Fungus corpus", 10,
        "outputs/approved_paired_studies/prjeb42019-funguscorpus-exact10-approved-paired-study-20261020-r1",
        _t("native_inputs/wgs_tpm.sample_by_ko.tsv.gz"), _t("native_inputs/picrust2.sample_by_ko.tsv.gz"),
        _t("native_inputs/tax4fun2.sample_by_ko.tsv.gz"), "crosswalk/sample_crosswalk.tsv",
        "biological_sample_id", "biological_sample_id", "amplicon_sample_id",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_digest(values: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def standardize_kos(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    mapped = []
    for value in frame.columns.astype(str):
        match = KO_RE.fullmatch(value.strip())
        if match:
            mapped.append(match.group(1))
        else:
            mapped.append("")
    keep = np.asarray(mapped) != ""
    frame = frame.loc[:, keep].copy()
    frame.columns = np.asarray(mapped)[keep]
    if frame.columns.duplicated().any():
        frame = frame.T.groupby(level=0, sort=True).sum().T
    values = frame.to_numpy(float)
    if frame.empty or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"invalid KO matrix: {path}")
    return frame.sort_index(axis=1)


def read_sample_by_ko(paths: list[Path]) -> pd.DataFrame:
    blocks = []
    for path in paths:
        raw = pd.read_csv(path, sep="\t", index_col=0)
        raw.index = raw.index.astype(str)
        if raw.index.duplicated().any():
            raise ValueError(f"duplicate samples: {path}")
        blocks.append(standardize_kos(raw, path))
    merged = pd.concat(blocks, axis=0, join="outer").fillna(0.0)
    if merged.index.duplicated().any():
        raise ValueError(f"duplicate samples after merging {paths}")
    return merged


def map_matrix(matrix: pd.DataFrame, crosswalk: pd.DataFrame, source_column: str, canonical_column: str, order: list[str]) -> pd.DataFrame:
    if set(matrix.index) == set(order):
        renamed = matrix.copy()
    else:
        if source_column == canonical_column:
            raise ValueError(f"matrix does not match canonical {canonical_column} values")
        mapping = crosswalk.set_index(source_column)[canonical_column]
        if not mapping.index.is_unique or mapping.isna().any():
            raise ValueError(f"invalid mapping {source_column}->{canonical_column}")
        if set(mapping.index) <= set(matrix.index):
            renamed = matrix.loc[mapping.index].rename(index=mapping.to_dict())
        else:
            missing = sorted(set(matrix.index) - set(mapping.index))[:5]
            absent = sorted(set(mapping.index) - set(matrix.index))[:5]
            raise ValueError(f"matrix/crosswalk mismatch: matrix-only={missing}, approved-crosswalk-missing={absent}")
    if renamed.index.duplicated().any() or set(renamed.index) != set(order):
        raise ValueError("canonical sample mapping is incomplete or nonunique")
    return renamed.loc[order]


def load_empty_quarter(spec: StudySpec) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, list[dict]]:
    wgs_path = SOURCE / spec.wgs[0]
    pic_path = SOURCE / spec.picrust2[0]
    tax_path = SOURCE / spec.tax4fun2[0]
    pmap_path = SOURCE / spec.crosswalk
    tmap_path = SOURCE / "eq_benchmark_data/functional/tax4fun2/sample_mapping.tsv"
    pmap = pd.read_csv(pmap_path, sep="\t", dtype=str)
    tmap = pd.read_csv(tmap_path, sep="\t", dtype=str)
    mapping_columns = ["picrust2_id", "sample_id"]
    if not pmap[mapping_columns].sort_values("picrust2_id").reset_index(drop=True).equals(
        tmap[mapping_columns].sort_values("picrust2_id").reset_index(drop=True)
    ):
        raise ValueError("Empty Quarter PICRUSt2 and Tax4Fun2 mappings differ")
    wgs_ids = pd.read_csv(wgs_path, sep="\t", usecols=["sample_id"], dtype=str).sample_id
    counts = pmap.sample_id.value_counts()
    overlap = set(wgs_ids) & set(pmap.sample_id)
    ambiguous = set(counts[counts.gt(1)].index) & overlap
    expected_ambiguous = {"F1PRr2", "F41Dr3", "F58Sr3", "T1Dr1"}
    if ambiguous != expected_ambiguous:
        raise ValueError(f"Empty Quarter ambiguous mapping changed: {sorted(ambiguous)}")
    sample_ids = sorted(overlap - ambiguous)
    if len(sample_ids) != spec.expected_samples:
        raise ValueError(f"Empty Quarter strict paired count changed: {len(sample_ids)}")

    wgs_raw = pd.read_csv(wgs_path, sep="\t", dtype={"sample_id": str, "trip": str, "site": str})
    metadata = wgs_raw.loc[wgs_raw.sample_id.isin(sample_ids), ["sample_id", "trip", "site"]].copy()
    wgs = standardize_kos(wgs_raw.set_index("sample_id").loc[sample_ids], wgs_path)

    def prediction(path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, sep="\t", index_col=0)
        raw.index = raw.index.astype(str)
        selected = pmap[pmap.sample_id.isin(sample_ids)].copy()
        if selected.sample_id.duplicated().any():
            raise ValueError("Empty Quarter selected prediction mapping is not unique")
        cols = selected.picrust2_id.tolist()
        if not set(cols) <= set(raw.columns.astype(str)):
            raise ValueError(f"prediction columns missing from {path}")
        out = raw.loc[:, cols].T
        out.index = selected.set_index("picrust2_id").loc[cols, "sample_id"].tolist()
        return standardize_kos(out.loc[sample_ids], path)

    predictions = {"PICRUSt2": prediction(pic_path), "Tax4Fun2": prediction(tax_path)}
    audit = [
        {"dataset": spec.dataset, "role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for role, path in (("wgs", wgs_path), ("PICRUSt2", pic_path), ("Tax4Fun2", tax_path), ("prediction_mapping", pmap_path), ("prediction_mapping_check", tmap_path))
    ]
    return wgs, predictions, metadata.set_index("sample_id").loc[sample_ids].reset_index(), audit


def load_study(spec: StudySpec) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, list[dict]]:
    if spec.loader == "empty_quarter":
        return load_empty_quarter(spec)
    package = SOURCE / spec.package
    crosswalk_path = package / spec.crosswalk
    crosswalk = pd.read_csv(crosswalk_path, sep="\t", dtype=str)
    if "inclusion_status" in crosswalk:
        crosswalk = crosswalk.loc[crosswalk.inclusion_status.eq("included")].copy()
    order = crosswalk[spec.canonical_column].astype(str).tolist()
    if len(order) != spec.expected_samples or len(order) != len(set(order)):
        raise ValueError(f"{spec.dataset}: expected {spec.expected_samples} unique samples, found {len(order)}")
    paths = {
        "wgs": [package / value for value in spec.wgs],
        "PICRUSt2": [package / value for value in spec.picrust2],
        "Tax4Fun2": [package / value for value in spec.tax4fun2],
    }
    wgs = map_matrix(read_sample_by_ko(paths["wgs"]), crosswalk, spec.wgs_column, spec.canonical_column, order)
    predictions = {
        method: map_matrix(read_sample_by_ko(paths[method]), crosswalk, spec.prediction_column, spec.canonical_column, order)
        for method in METHODS
    }
    audit = []
    for role, items in paths.items():
        for path in items:
            audit.append({"dataset": spec.dataset, "role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    audit.append({"dataset": spec.dataset, "role": "crosswalk", "path": str(crosswalk_path), "bytes": crosswalk_path.stat().st_size, "sha256": sha256(crosswalk_path)})
    return wgs, predictions, crosswalk, audit


def profile_spearman_with_explicit_zero_overlap(
    truth: pd.DataFrame, prediction: pd.DataFrame, universe: list[str]
) -> pd.DataFrame:
    """Use the library scorer while retaining samples with no WGS-universe overlap."""
    truth_aligned = truth.reindex(columns=universe)
    prediction_aligned = prediction.reindex(columns=universe, fill_value=0.0)
    valid = truth_aligned.sum(axis=1).gt(0) & prediction_aligned.sum(axis=1).gt(0)
    blocks = []
    if valid.any():
        blocks.append(samplewise_spearman_on_universe(truth.loc[valid], prediction.loc[valid], universe))
    if (~valid).any():
        blocks.append(pd.DataFrame({
            "sample_id": truth.index[~valid],
            "n_evaluation_kos": len(universe),
            "wgs_total_before_norm": truth_aligned.loc[~valid].sum(axis=1).to_numpy(float),
            "method_total_before_norm": prediction_aligned.loc[~valid].sum(axis=1).to_numpy(float),
            "spearman_r": np.nan,
            "spearman_pvalue": np.nan,
            "spearman_status": "zero_method_total_on_wgs_universe",
            "ko_surface_policy": "ordered_wgs_positive_universe_zero_fill_then_l1",
        }))
    result = pd.concat(blocks, ignore_index=True).set_index("sample_id").loc[truth.index]
    result.index.name = "sample_id"
    return result.reset_index()


def profile_metrics(studies: dict, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_rows, legacy_rows, richness_rows, ko_rows, summary_rows, coverage_rows = [], [], [], [], [], []
    for spec in STUDIES:
        wgs, predictions = studies[spec.dataset][0], studies[spec.dataset][1]
        universe = sorted(wgs.columns[wgs.sum(axis=0).gt(0)].astype(str))
        for method, prediction in predictions.items():
            aligned = profile_spearman_with_explicit_zero_overlap(wgs, prediction, universe)
            aligned.insert(0, "method_label", method)
            aligned.insert(0, "dataset", spec.dataset)
            aligned["display_name"] = spec.display_name
            aligned = aligned.rename(columns={"spearman_r": "sample_spearman"})
            sample_rows.append(aligned)

            legacy = samplewise_profile_metrics(wgs, prediction)
            legacy.insert(0, "method_label", method)
            legacy.insert(0, "dataset", spec.dataset)
            legacy_rows.append(legacy)

            richness = per_sample_richness_overlap(wgs, prediction)
            richness.insert(0, "method_label", method)
            richness.insert(0, "dataset", spec.dataset)
            richness_rows.append(richness)

            per_ko = per_ko_spearman(wgs, prediction)
            per_ko.insert(0, "method_label", method)
            per_ko.insert(0, "dataset", spec.dataset)
            ko_rows.append(per_ko)

            shared = sorted(set(wgs.columns) & set(prediction.columns))
            present = len(set(universe) & set(prediction.columns))
            values = aligned.sample_spearman.dropna()
            summary_rows.append({
                "dataset": spec.dataset, "display_name": spec.display_name, "accession": spec.accession,
                "method_label": method, "n_samples": len(aligned), "n_wgs_positive_kos": len(universe),
                "n_method_native_kos": prediction.shape[1], "n_shared_native_kos": len(shared),
                "n_wgs_universe_kos_present_in_method": present,
                "fraction_wgs_universe_present_in_method": present / len(universe),
                "mean_sample_spearman": values.mean(), "median_sample_spearman": values.median(),
                "mean_pairwise_pearson": legacy.pearson_r.mean(), "mean_pairwise_cosine": legacy.cosine_similarity.mean(),
                "mean_wgs_richness_recovered": richness.fraction_wgs_nonzero_kos_recovered_by_method.mean(),
                "median_per_ko_spearman": per_ko.spearman_across_samples.median(),
                "n_defined_per_ko_spearman": int(per_ko.spearman_status.eq("defined").sum()),
            })
            coverage_rows.append({"dataset": spec.dataset, "method": method, "n_wgs_universe_kos": len(universe),
                                  "n_available_kos": present, "fraction_universe_in_head": present / len(universe)})
    sample = pd.concat(sample_rows, ignore_index=True)
    legacy = pd.concat(legacy_rows, ignore_index=True)
    richness = pd.concat(richness_rows, ignore_index=True)
    perko = pd.concat(ko_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    coverage = pd.DataFrame(coverage_rows)
    sample.to_csv(out / "profile_sample_spearman_wgs_universe.tsv.gz", **RT)
    legacy.to_csv(out / "profile_sample_pairwise_metrics.tsv.gz", **RT)
    richness.to_csv(out / "profile_richness_overlap.tsv.gz", **RT)
    perko.to_csv(out / "profile_per_ko_spearman.tsv.gz", **RT)
    summary.to_csv(out / "profile_summary.tsv", **RT)
    coverage.to_csv(out / "profile_coverage.tsv", **RT)
    return sample, summary, coverage


def labels_registry() -> pd.DataFrame:
    path = SOURCE / "functional_inference_generalization/data/metadata/expansion_validation_label_groups_20260921_r2.tsv"
    return pd.read_csv(path, sep="\t", dtype=str)


def unpaired_contrasts(studies: dict) -> list[dict]:
    registry = labels_registry()
    contrasts = []
    for dataset, cid, a, b, status in (
        ("han_river", "control_han_river_upstream_vs_downstream", "upstream", "downstream", "accepted_control"),
        ("activated_sludge", "control_activated_sludge_disturbed_vs_stable", "disturbed", "stable", "accepted_control"),
        ("topsoil", "control_topsoil_control_vs_warmed", "control", "warmed", "accepted_control_zero_bh"),
    ):
        block = registry.loc[registry.dataset.eq(dataset), ["sample_id", "label"]].copy()
        block = block.loc[block.label.isin([a, b])]
        contrasts.append({"contrast_id": cid, "dataset": dataset, "design": "unpaired", "group_a": a, "group_b": b,
                          "labels": block.set_index("sample_id").label, "status": status,
                          "label_source": "frozen expansion validation label registry r2"})

    top_cw = studies["topsoil"][2]
    contrasts.append({"contrast_id": "topsoil_biome_tundra_vs_temperate_grassland", "dataset": "topsoil", "design": "unpaired",
                      "group_a": "tundra", "group_b": "temperate grassland",
                      "labels": top_cw.set_index("biological_sample_id").environment_biome,
                      "status": "computed_explicit_metadata", "label_source": "Stage-5 crosswalk environment_biome"})

    olive_cw = studies["olive_xylem"][2]
    for season in ("season1", "season2"):
        block = olive_cw.loc[olive_cw.stratum.eq(season)].copy()
        label = pd.Series(np.where(block.biological_sample_id.str.startswith("FS"), "FS17", "Kalamata"),
                          index=block.biological_sample_id, name="label")
        contrasts.append({"contrast_id": f"olive_xylem_{season}_FS17_vs_Kalamata", "dataset": "olive_xylem", "design": "unpaired",
                          "group_a": "FS17", "group_b": "Kalamata", "labels": label,
                          "status": "computed_provisional_zero_bh", "label_source": "documented sample-alias prefix; seasons separate"})
    return contrasts


def pair_definitions(studies: dict) -> list[dict]:
    rows = []
    han_ids = studies["han_river"][0].index.astype(str)
    sites = sorted({value.split("_", 1)[1] for value in han_ids if value.startswith("201608_")})
    pairs = pd.DataFrame({"pair_id": sites, "sample_a": [[f"201608_{site}"] for site in sites],
                          "sample_b": [[f"201702_{site}"] for site in sites]})
    rows.append({"contrast_id": "han_river_summer_vs_winter_site_paired", "dataset": "han_river", "design": "paired",
                 "group_a": "summer_Aug2016", "group_b": "winter_Feb2017", "pairs": pairs,
                 "status": "computed_provisional", "label_source": "Stage-5 tranche; month/season interpretation remains provisional"})

    contract_path = SOURCE / "functional_inference_generalization/results/full_loso23_contract_r1/contract/sample_registry.tsv"
    contract = pd.read_csv(contract_path, sep="\t", dtype=str)
    contract_ids = set(contract.loc[contract.dataset.eq("empty_quarter"), "biological_sample_id"])
    meta = studies["empty_quarter"][2].set_index("sample_id")
    meta = meta.loc[meta.index.isin(contract_ids)].copy()
    meta["compartment"] = [re.search(r"(PRr|Dr|Sr)\d+$", value).group(1) for value in meta.index]
    meta["block"] = meta.trip + "/" + meta.site
    for cid, comp_a, comp_b, a, b in (
        ("empty_quarter_surface_vs_deep_site_paired", "Sr", "Dr", "surface_soil", "deep_soil"),
        ("empty_quarter_rhizosphere_vs_surface_site_paired", "PRr", "Sr", "rhizosphere", "surface_soil"),
    ):
        blocks = sorted(set(meta.loc[meta.compartment.eq(comp_a), "block"]) & set(meta.loc[meta.compartment.eq(comp_b), "block"]))
        pairs = pd.DataFrame({
            "pair_id": blocks,
            "sample_a": [meta.index[(meta.block.eq(block)) & meta.compartment.eq(comp_a)].tolist() for block in blocks],
            "sample_b": [meta.index[(meta.block.eq(block)) & meta.compartment.eq(comp_b)].tolist() for block in blocks],
        })
        rows.append({"contrast_id": cid, "dataset": "empty_quarter", "design": "paired", "group_a": a, "group_b": b,
                     "pairs": pairs, "status": "computed_provisional",
                     "label_source": "sample suffix plus trip/site blocks; primary label table unavailable"})
    return rows


def screen_family(matrix: pd.DataFrame, sample_ids: list[str]) -> tuple[pd.DataFrame, list[str], int]:
    relative, totals = normalize_rows(matrix.loc[sample_ids])
    if (totals <= 0).any():
        raise ValueError("zero-total sample in contrast")
    relative = relative.fillna(0.0)
    minimum = max(5, int(math.ceil(0.10 * len(sample_ids))))
    kos = sorted(relative.columns[(relative > 0).sum(axis=0).ge(minimum)])
    return relative, kos, minimum


def paired_test(a: pd.DataFrame, b: pd.DataFrame, kos: list[str]) -> pd.DataFrame:
    A, B = a[kos].to_numpy(float), b[kos].to_numpy(float)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for index, ko in enumerate(kos):
            difference = A[:, index] - B[:, index]
            nonzero = difference[difference != 0]
            if len(nonzero) == 0:
                rows.append((ko, 0.0, 1.0, 0.0, 0))
                continue
            ranks = rankdata(np.abs(nonzero))
            wplus = ranks[nonzero > 0].sum()
            wminus = ranks[nonzero < 0].sum()
            pvalue = wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto").pvalue
            rows.append((ko, wplus, float(pvalue), (wplus - wminus) / (wplus + wminus), len(nonzero)))
    result = pd.DataFrame(rows, columns=["KO", "statistic", "pvalue", "effect", "n_nonzero_pairs"])
    result["bh_qvalue"] = benjamini_hochberg(result.pvalue)
    return result


def avg_pair_sides(relative: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.DataFrame([relative.loc[row.sample_a].mean(axis=0) for row in pairs.itertuples()], index=pairs.pair_id)
    b = pd.DataFrame([relative.loc[row.sample_b].mean(axis=0) for row in pairs.itertuples()], index=pairs.pair_id)
    return a, b


def screen_replay_unpaired(wgs: pd.DataFrame, labels: pd.Series, a: str, b: str) -> pd.DataFrame:
    sample_ids = [value for value in wgs.index if value in labels.index and labels.loc[value] in {a, b}]
    relative, kos, _ = screen_family(wgs, sample_ids)
    tests = wilcoxon_rank_sum_by_ko(relative.loc[:, kos], labels.loc[sample_ids], a, b)
    effects = rank_biserial_effect_by_ko(relative.loc[:, kos], labels.loc[sample_ids], a, b).set_index("KO")
    tests["bh_qvalue"] = benjamini_hochberg(tests.pvalue)
    tests["effect"] = tests.KO.map(effects.rank_biserial_effect)
    return tests


def screen_replay_paired(wgs: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    ids = sorted({sample for row in pairs.itertuples() for sample in row.sample_a + row.sample_b})
    relative, kos, _ = screen_family(wgs, ids)
    a, b = avg_pair_sides(relative, pairs)
    return paired_test(a, b, kos)


def validate_screen(contrast_id: str, observed: pd.DataFrame) -> dict:
    path = SOURCE / f"functional_inference_generalization/results/expansion_candidate_wgs_effect_screen_r1/per_ko_tests/{contrast_id}.tsv.gz"
    reference = pd.read_csv(path, sep="\t").sort_values("KO").reset_index(drop=True)
    candidate = observed.sort_values("KO").reset_index(drop=True)
    same_kos = candidate.KO.tolist() == reference.KO.tolist()
    diffs = {}
    for column in ("pvalue", "bh_qvalue", "effect"):
        diffs[column] = float(np.max(np.abs(candidate[column].to_numpy(float) - reference[column].to_numpy(float)))) if same_kos else np.nan
    return {"contrast_id": contrast_id, "reference_path": str(path), "n_kos": len(candidate), "ko_identity_exact": same_kos,
            **{f"max_abs_diff_{key}": value for key, value in diffs.items()},
            "validation_pass": bool(same_kos and all(value <= 1e-12 for value in diffs.values()))}


def correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(spearmanr(x, y).statistic if kind == "spearman" else pearsonr(x, y).statistic)


def project_outputs_from_effects(contrast: dict, tests: pd.DataFrame, prediction_effects: dict[str, np.ndarray], out_rows: dict):
    kos = tests.KO.astype(str).to_numpy()
    q = tests.bh_qvalue.to_numpy(float)
    wgs_effect = tests.effect.to_numpy(float)
    selected = q < 0.05
    for method, effect in prediction_effects.items():
        out_rows["effect"].append({"contrast_id": contrast["contrast_id"], "dataset": contrast["dataset"], "design": contrast["design"],
                                   "prediction_method": method, "n_kos": len(kos), "n_wgs_bh_selected": int(selected.sum()),
                                   "spearman_effect_all": correlation(wgs_effect, effect, "spearman"),
                                   "pearson_effect_all": correlation(wgs_effect, effect, "pearson"),
                                   "spearman_effect_wgs_bh_selected": correlation(wgs_effect[selected], effect[selected], "spearman"),
                                   "pearson_effect_wgs_bh_selected": correlation(wgs_effect[selected], effect[selected], "pearson")})
        pr, roc, summary = directional_ranking_curves(q, wgs_effect, effect)
        for table, key in ((pr, "pr"), (roc, "roc"), (summary, "ranking")):
            table.insert(0, "prediction_method", method)
            table.insert(0, "dataset", contrast["contrast_id"])
            out_rows[key].append(table)
        assignment, deciles = decile_outcome_composition(kos, effect, q, wgs_effect)
        assignment.insert(0, "prediction_method", method); assignment.insert(0, "dataset", contrast["contrast_id"])
        deciles.insert(0, "prediction_method", method); deciles.insert(0, "dataset", contrast["contrast_id"])
        out_rows["assignment"].append(assignment); out_rows["decile"].append(deciles)
    classes = np.where((q <= .05) & (wgs_effect > 0), "sig_positive", np.where((q <= .05) & (wgs_effect < 0), "sig_negative", "nonsignificant"))
    out_rows["baseline"].append({"dataset": contrast["contrast_id"], "n_kos": len(kos),
                                 "p_sig_positive": float(np.mean(classes == "sig_positive")),
                                 "p_sig_negative": float(np.mean(classes == "sig_negative")),
                                 "p_any_significant": float(np.mean(classes != "nonsignificant"))})


def group_metrics(studies: dict, out: Path) -> dict[str, pd.DataFrame]:
    raw_rows, raw_details, bh_rows, bh_details, sun_rows, sun_null, sun_details = [], [], [], [], [], [], []
    screen_checks, contrast_rows, availability_rows, project_availability_rows = [], [], [], []
    project = {key: [] for key in ("effect", "pr", "roc", "ranking", "assignment", "decile", "baseline")}

    for contrast in unpaired_contrasts(studies):
        dataset = contrast["dataset"]
        wgs, predictions = studies[dataset][0], studies[dataset][1]
        labels = contrast["labels"]
        sample_ids = [value for value in wgs.index if value in labels.index and labels.loc[value] in {contrast["group_a"], contrast["group_b"]}]
        truth = wgs.loc[sample_ids]
        preds = {method: matrix.loc[sample_ids] for method, matrix in predictions.items()}
        fixed = labels.loc[sample_ids]
        contrast_rows.append({key: contrast[key] for key in ("contrast_id", "dataset", "design", "group_a", "group_b", "status", "label_source")} |
                             {"n_group_a": int(fixed.eq(contrast["group_a"]).sum()), "n_group_b": int(fixed.eq(contrast["group_b"]).sum()), "n_pairs": np.nan})

        contrast_bh_details = []
        for method, matrix in preds.items():
            common = sorted(set(truth.columns) & set(matrix.columns))
            zero_ids = matrix.loc[:, common].sum(axis=1).loc[lambda values: values <= 0].index.tolist()
            if zero_ids:
                availability_rows.append({
                    "contrast_id": contrast["contrast_id"], "prediction_method": method,
                    "paper_metric_status": "unavailable_zero_total_on_pairwise_shared_ko_surface",
                    "n_zero_total_samples": len(zero_ids), "zero_total_sample_ids": ";".join(zero_ids),
                    "ko_surface_policy": "WGS-and-one-method shared native KO surface",
                })
                continue
            one = {method: matrix}
            raw, detail = matchado_differential_abundance_metrics(
                truth, one, fixed, group_a=contrast["group_a"], group_b=contrast["group_b"], n_permutations=20
            )
            raw.insert(0, "dataset", contrast["contrast_id"]); detail.insert(0, "dataset", contrast["contrast_id"])
            raw_rows.append(raw); raw_details.append(detail)

            bh, detail_bh = matchado_bh_shared_wgs_reference_metrics(
                truth, one, fixed, group_a=contrast["group_a"], group_b=contrast["group_b"]
            )
            bh.insert(0, "dataset", contrast["contrast_id"]); detail_bh.insert(0, "dataset", contrast["contrast_id"])
            bh_rows.append(bh); bh_details.append(detail_bh); contrast_bh_details.append(detail_bh)

            sun, null, sun_detail = sun_inference_correlation_abundance_randomization(
                truth, one, fixed, group_a=contrast["group_a"], group_b=contrast["group_b"], n_randomizations=100
            )
            for table in (sun, null, sun_detail): table.insert(0, "dataset", contrast["contrast_id"])
            sun_rows.append(sun); sun_null.append(null); sun_details.append(sun_detail)
            availability_rows.append({
                "contrast_id": contrast["contrast_id"], "prediction_method": method,
                "paper_metric_status": "computed", "n_zero_total_samples": 0, "zero_total_sample_ids": "",
                "ko_surface_policy": "WGS-and-one-method shared native KO surface",
            })

        replay = screen_replay_unpaired(truth, fixed, contrast["group_a"], contrast["group_b"])
        screen_checks.append(validate_screen(contrast["contrast_id"], replay))

        kos = replay.KO.tolist()
        prediction_effects = {}
        for method, matrix in preds.items():
            rel, totals = normalize_rows(matrix)
            zero_ids = totals.loc[totals <= 0].index.tolist()
            if zero_ids:
                project_availability_rows.append({
                    "contrast_id": contrast["contrast_id"], "prediction_method": method,
                    "project_metric_status": "unavailable_zero_total_native_prediction",
                    "n_zero_total_samples": len(zero_ids), "zero_total_sample_ids": ";".join(zero_ids),
                })
                continue
            aligned = rel.reindex(columns=kos, fill_value=0.0).fillna(0.0)
            prediction_effects[method] = rank_biserial_effect_by_ko(aligned, fixed, contrast["group_a"], contrast["group_b"]).set_index("KO").loc[kos, "rank_biserial_effect"].to_numpy(float)
            project_availability_rows.append({
                "contrast_id": contrast["contrast_id"], "prediction_method": method,
                "project_metric_status": "computed", "n_zero_total_samples": 0, "zero_total_sample_ids": "",
            })
        project_outputs_from_effects(contrast, replay[["KO", "bh_qvalue", "effect"]], prediction_effects, project)

    for contrast in pair_definitions(studies):
        dataset = contrast["dataset"]
        wgs, predictions = studies[dataset][0], studies[dataset][1]
        replay = screen_replay_paired(wgs, contrast["pairs"])
        screen_checks.append(validate_screen(contrast["contrast_id"], replay))
        contrast_rows.append({key: contrast[key] for key in ("contrast_id", "dataset", "design", "group_a", "group_b", "status", "label_source")} |
                             {"n_group_a": len(contrast["pairs"]), "n_group_b": len(contrast["pairs"]), "n_pairs": len(contrast["pairs"])})
        ids = sorted({sample for row in contrast["pairs"].itertuples() for sample in row.sample_a + row.sample_b})
        kos = replay.KO.tolist()
        prediction_effects = {}
        pred_tests = {}
        for method, matrix in predictions.items():
            relative, totals = normalize_rows(matrix.loc[ids])
            zero_ids = totals.loc[totals <= 0].index.tolist()
            if zero_ids:
                project_availability_rows.append({
                    "contrast_id": contrast["contrast_id"], "prediction_method": method,
                    "project_metric_status": "unavailable_zero_total_native_prediction",
                    "n_zero_total_samples": len(zero_ids), "zero_total_sample_ids": ";".join(zero_ids),
                })
                continue
            aligned = relative.reindex(columns=kos, fill_value=0.0).fillna(0.0)
            a, b = avg_pair_sides(aligned, contrast["pairs"])
            tested = paired_test(a, b, kos)
            pred_tests[method] = tested
            prediction_effects[method] = tested.effect.to_numpy(float)
            score = classification_set_metrics(set(replay.loc[replay.bh_qvalue.lt(.05), "KO"]), set(tested.loc[tested.bh_qvalue.lt(.05), "KO"]))
            bh_rows.append(pd.DataFrame([{
                "dataset": contrast["contrast_id"], "prediction_method": method, **score, "n_tested_kos": len(kos),
                "n_wgs_significant": int(replay.bh_qvalue.lt(.05).sum()), "n_prediction_significant": int(tested.bh_qvalue.lt(.05).sum()),
                "prevalence_threshold": .10, "q_threshold": .05, "ko_family_policy": "paired_screen_wgs_detectable_10_percent_minimum_5",
                "multiple_testing_method": "benjamini_hochberg_separately_within_wgs_and_prediction",
                "significance_rule": "bh_qvalue < 0.05", "metric_provenance": "project_paired_extension_not_matchado_or_sun",
                "modification_reason": "paired biological design requires signed-rank tests on matched units"}]))
            merged = replay.merge(tested, on="KO", suffixes=("_wgs", "_prediction"))
            merged.insert(0, "prediction_method", method); merged.insert(0, "dataset", contrast["contrast_id"])
            bh_details.append(merged)
            project_availability_rows.append({
                "contrast_id": contrast["contrast_id"], "prediction_method": method,
                "project_metric_status": "computed", "n_zero_total_samples": 0, "zero_total_sample_ids": "",
            })
        project_outputs_from_effects(contrast, replay[["KO", "bh_qvalue", "effect"]], prediction_effects, project)

    outputs = {
        "matchado_raw_p_summary": pd.concat(raw_rows, ignore_index=True),
        "matchado_raw_p_per_ko": pd.concat(raw_details, ignore_index=True),
        "project_bh_summary": pd.concat(bh_rows, ignore_index=True),
        "project_bh_per_ko": pd.concat(bh_details, ignore_index=True),
        "sun_summary": pd.concat(sun_rows, ignore_index=True),
        "sun_randomizations": pd.concat(sun_null, ignore_index=True),
        "sun_per_ko": pd.concat(sun_details, ignore_index=True),
        "contrast_manifest": pd.DataFrame(contrast_rows),
        "paper_metric_availability": pd.DataFrame(availability_rows),
        "project_metric_availability": pd.DataFrame(project_availability_rows),
        "wgs_screen_validation": pd.DataFrame(screen_checks),
        "effect_summary": pd.DataFrame(project["effect"]),
        "directional_pr_points": pd.concat(project["pr"], ignore_index=True),
        "directional_roc_points": pd.concat(project["roc"], ignore_index=True),
        "directional_metric_summary": pd.concat(project["ranking"], ignore_index=True),
        "decile_assignments": pd.concat(project["assignment"], ignore_index=True),
        "decile_composition": pd.concat(project["decile"], ignore_index=True),
        "decile_baselines": pd.DataFrame(project["baseline"]),
    }
    for name, table in outputs.items():
        suffix = ".tsv.gz" if name.endswith("per_ko") or name.endswith("points") or name.endswith("assignments") else ".tsv"
        table.to_csv(out / f"{name}{suffix}", **RT)
    return outputs


def make_figures(sample: pd.DataFrame, summary: pd.DataFrame, coverage: pd.DataFrame, group: dict, figures: Path):
    registry = pd.DataFrame({"dataset": [spec.dataset for spec in STUDIES], "is_expansion_group": True})
    original_boxplot = Axes.boxplot
    def seaborn_012_matplotlib_38_boxplot(self, *args, **kwargs):
        kwargs.pop("legend", None)
        return original_boxplot(self, *args, **kwargs)
    Axes.boxplot = seaborn_012_matplotlib_38_boxplot
    try:
        fig, _ = plot_profile_study_panels(sample, registry, METHODS, ncols=6)
    finally:
        Axes.boxplot = original_boxplot
    fig._suptitle.set_text("Sample-level KO profile agreement across completed Table 2.2 expansion scopes")
    fig.savefig(figures / "01_profile_spearman_all_completed_expansions.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, _ = plot_ko_coverage(coverage, METHODS)
    fig.savefig(figures / "02_ko_coverage_all_completed_expansions.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    raw = group["matchado_raw_p_summary"]
    availability = group["paper_metric_availability"]
    computed_counts = availability.loc[availability.paper_metric_status.eq("computed")].groupby("contrast_id").prediction_method.nunique()
    order = [value for value in group["contrast_manifest"].query("design == 'unpaired'").contrast_id
             if computed_counts.get(value, 0) == len(METHODS)]
    contrast_labels = {
        "control_han_river_upstream_vs_downstream": "Han upstream/downstream",
        "control_activated_sludge_disturbed_vs_stable": "Sludge disturbed/stable",
        "control_topsoil_control_vs_warmed": "Topsoil warming",
        "topsoil_biome_tundra_vs_temperate_grassland": "Topsoil biome",
        "olive_xylem_season1_FS17_vs_Kalamata": "Olive season 1",
        "olive_xylem_season2_FS17_vs_Kalamata": "Olive season 2",
        "han_river_summer_vs_winter_site_paired": "Han summer/winter",
        "empty_quarter_surface_vs_deep_site_paired": "Empty Quarter surface/deep",
        "empty_quarter_rhizosphere_vs_surface_site_paired": "Empty Quarter rhizosphere/surface",
    }
    plot_order = [contrast_labels[value] for value in order]
    plot_raw = raw.assign(dataset=raw.dataset.map(contrast_labels).fillna(raw.dataset))
    fig, _ = plot_metric_row_grid(plot_raw, plot_order, METHODS, title="Matchado-derived raw-p recovery: completed expansion contrasts")
    fig.savefig(figures / "03_matchado_raw_p_recovery.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, _ = plot_matchado_randomization_dumbbells(plot_raw, plot_order, METHODS)
    fig.savefig(figures / "04_matchado_label_randomization.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    plot_sun = group["sun_summary"].assign(dataset=lambda value: value.dataset.map(contrast_labels).fillna(value.dataset))
    plot_null = group["sun_randomizations"].assign(dataset=lambda value: value.dataset.map(contrast_labels).fillna(value.dataset))
    fig, _ = plot_sun_observed_controls(plot_sun, plot_null, plot_order, METHODS)
    fig.savefig(figures / "05_sun_observed_and_wgs_abundance_controls.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    ranking = group["directional_metric_summary"]
    plot_ranking = ranking.assign(dataset=ranking.dataset.map(contrast_labels).fillna(ranking.dataset))
    fig, _ = plot_directional_metric_summary(plot_ranking, "ap", METHODS)
    fig.savefig(figures / "06_directional_average_precision.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, _ = plot_directional_metric_summary(plot_ranking, "roc_auc", METHODS)
    fig.savefig(figures / "07_directional_roc_auc.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    defined = ranking.groupby(["dataset", "prediction_method"]).status.apply(lambda s: s.eq("defined").all()).unstack(fill_value=False)
    roc_order = [dataset for dataset in group["contrast_manifest"].contrast_id if dataset in defined.index and defined.loc[dataset, list(METHODS)].all()]
    if roc_order:
        fig, _ = plot_directional_roc_grid(group["directional_roc_points"], ranking, roc_order, METHODS,
                                           display_names=contrast_labels, benchmark_count=0)
        fig.savefig(figures / "08_directional_roc_curves.png", dpi=180, bbox_inches="tight"); plt.close(fig)

        cohort = group["contrast_manifest"][["contrast_id", "group_a", "group_b"]].rename(
            columns={"contrast_id": "dataset"}
        )
        fig, _ = plot_signed_decile_grid(group["decile_composition"], group["decile_baselines"], cohort,
                                         roc_order, METHODS, display_names=contrast_labels, benchmark_count=0)
        fig.savefig(figures / "09_directional_decile_composition.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, max(5, .36 * len(STUDIES))))
    wide = summary.pivot(index="display_name", columns="method_label", values="mean_sample_spearman").loc[[s.display_name for s in STUDIES]]
    y = np.arange(len(wide))
    for offset, method in zip((-.14, .14), METHODS):
        ax.scatter(wide[method], y + offset, label=method, s=42)
    ax.set(yticks=y, yticklabels=wide.index, xlabel="Mean sample-wise Spearman", title="Completed Table 2.2 expansions")
    ax.invert_yaxis(); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(figures / "10_profile_mean_summary.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="replace only generated expansion_benchmark results/figures")
    args = parser.parse_args()
    results, figures = HERE / "results", HERE / "figures"
    if results.exists() or figures.exists():
        if not args.overwrite:
            raise FileExistsError("results/ or figures/ exists; pass --overwrite to replace generated expansion artifacts")
        for path in (results, figures):
            if path.exists(): shutil.rmtree(path)
    results.mkdir(); figures.mkdir()

    studies, audit_rows, status_rows = {}, [], []
    for spec in STUDIES:
        wgs, predictions, metadata, audit = load_study(spec)
        if len(wgs) != spec.expected_samples or any(len(value) != spec.expected_samples for value in predictions.values()):
            raise ValueError(f"{spec.dataset}: sample count differs after loading")
        if any(list(value.index) != list(wgs.index) for value in predictions.values()):
            raise ValueError(f"{spec.dataset}: methods are not aligned to WGS")
        studies[spec.dataset] = (wgs, predictions, metadata)
        audit_rows.extend(audit)
        status_rows.append({"dataset": spec.dataset, "display_name": spec.display_name, "accession": spec.accession,
                            "empo_level_3": spec.empo, "table_2_2_status": spec.status, "n_samples": len(wgs),
                            "wgs_kos": wgs.shape[1], "picrust2_kos": predictions["PICRUSt2"].shape[1],
                            "tax4fun2_kos": predictions["Tax4Fun2"].shape[1], "training_performed": False})
        print(f"loaded {spec.dataset}: n={len(wgs)}", flush=True)

    pd.DataFrame(status_rows).to_csv(results / "study_status.tsv", **RT)
    sample, summary, coverage = profile_metrics(studies, results)
    group = group_metrics(studies, results)
    extra_inputs = [
        (SOURCE.parent / "chapter_2_draft/main.tex", "table_2_2_scope_source"),
        (SOURCE / "functional_inference_generalization/data/metadata/expansion_validation_label_groups_20260921_r2.tsv", "accepted_label_registry"),
        (SOURCE / "functional_inference_generalization/results/full_loso23_contract_r1/contract/sample_registry.tsv", "empty_quarter_screen_contract"),
    ]
    extra_inputs.extend(
        (Path(path), "completed_wgs_screen_reference")
        for path in group["wgs_screen_validation"].reference_path
    )
    for path, role in extra_inputs:
        audit_rows.append({"dataset": "analysis_scope", "role": role, "path": str(path),
                           "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(audit_rows).drop_duplicates(["role", "path"]).to_csv(results / "input_file_manifest.tsv", **RT)
    if not group["wgs_screen_validation"].validation_pass.all():
        raise RuntimeError("one or more WGS-screen reference replays failed")
    make_figures(sample, summary, coverage, group, figures)

    manifest_rows = []
    for path in sorted([*results.rglob("*"), *figures.rglob("*")]):
        if path.is_file():
            manifest_rows.append({"path": str(path.relative_to(HERE)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(HERE / "OUTPUT_MANIFEST.tsv", **RT)
    receipt = {
        "status": "PASS", "table_2_2_interpretation": "black completed available expansion entries only",
        "n_expansion_studies": len(STUDIES), "n_samples_total": int(sum(row["n_samples"] for row in status_rows)),
        "methods": list(METHODS), "training_performed": False, "source_root": str(SOURCE),
        "all_screen_replays_pass": True, "generated_files": len(manifest_rows),
    }
    (HERE / "RUN_COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
