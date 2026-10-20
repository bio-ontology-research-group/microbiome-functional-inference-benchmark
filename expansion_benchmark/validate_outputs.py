#!/usr/bin/env python3
"""Validate committed expansion outputs without reading the source workspace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


receipt = json.loads((HERE / "RUN_COMPLETE.json").read_text())
assert receipt["status"] == "PASS"
assert receipt["table_2_2_interpretation"] == "black completed available expansion entries only"
assert receipt["n_expansion_studies"] == 18 and receipt["n_samples_total"] == 1096
assert receipt["methods"] == ["PICRUSt2", "Tax4Fun2"]
assert receipt["training_performed"] is False and receipt["all_screen_replays_pass"] is True
assert isinstance(receipt["source_root"], str) and receipt["source_root"]
assert receipt["generated_files"] == 36

study = pd.read_csv(RESULTS / "study_status.tsv", sep="\t")
assert len(study) == 18
assert study.dataset.is_unique
assert int(study.n_samples.sum()) == 1096
assert not study.training_performed.any()

profile = pd.read_csv(
    RESULTS / "profile_sample_spearman_wgs_universe.tsv.gz",
    sep="\t",
    dtype={"sample_id": str},
)
assert len(profile) == 2 * 1096
assert set(profile.method_label) == {"PICRUSt2", "Tax4Fun2"}
expected = study.set_index("dataset").n_samples.to_dict()
counts = profile.groupby(["dataset", "method_label"]).size()
assert all(int(counts.loc[(dataset, method)]) == n for dataset, n in expected.items() for method in receipt["methods"])
undefined = profile.loc[profile.spearman_status.eq("zero_method_total_on_wgs_universe")]
assert set(undefined.method_label) <= {"Tax4Fun2"}
assert len(undefined) == 4

validation = pd.read_csv(RESULTS / "wgs_screen_validation.tsv", sep="\t")
assert len(validation) == 9
assert validation.validation_pass.all() and validation.ko_identity_exact.all()
assert validation.filter(regex="^max_abs_diff_").max().max() <= 1e-12

for name, status_column in (
    ("paper_metric_availability.tsv", "paper_metric_status"),
    ("project_metric_availability.tsv", "project_metric_status"),
):
    table = pd.read_csv(RESULTS / name, sep="\t")
    assert set(table.prediction_method) == set(receipt["methods"])
    assert table[status_column].notna().all()

manifest = pd.read_csv(HERE / "OUTPUT_MANIFEST.tsv", sep="\t")
assert len(manifest) == receipt["generated_files"]
for row in manifest.itertuples(index=False):
    path = HERE / row.path
    assert path.is_file()
    assert path.stat().st_size == row.bytes
    assert sha256(path) == row.sha256

print("PASS: expansion outputs, hashes, scope, method set, and reference tolerances are valid")
