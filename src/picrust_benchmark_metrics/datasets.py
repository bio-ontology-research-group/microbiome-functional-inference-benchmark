"""Load the compact processed study bundles distributed with the repository."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ARRAY_TO_METHOD = {
    "picrust2": "PICRUSt2",
    "picrust2_adaptation": "PICRUSt2 + adaptation",
    "tax4fun2": "Tax4Fun2",
}


def load_processed_study(path: str | Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.Series, dict]:
    """Load one bundled study as WGS, prediction matrices, labels, and metadata.

    Every returned matrix is sample-by-KO and shares the same ordered sample and
    KO identifiers. Prediction KOs unavailable for a method are stored as zeros.
    The files contain processed functional profiles only: no reads, assemblies,
    amplicon tables, model weights, or training inputs.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as bundle:
        required = {"sample_ids", "ko_ids", "labels", "wgs", "metadata_json", *ARRAY_TO_METHOD}
        missing = required - set(bundle.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        sample_ids = bundle["sample_ids"].astype(str)
        ko_ids = bundle["ko_ids"].astype(str)
        labels = pd.Series(bundle["labels"].astype(str), index=sample_ids, name="group_label")
        truth = pd.DataFrame(bundle["wgs"], index=sample_ids, columns=ko_ids)
        predictions = {
            method: pd.DataFrame(bundle[array], index=sample_ids, columns=ko_ids)
            for array, method in ARRAY_TO_METHOD.items()
        }
        metadata = json.loads(str(bundle["metadata_json"].item()))
    if truth.index.duplicated().any() or truth.columns.duplicated().any():
        raise ValueError(f"{path} contains duplicate sample or KO identifiers")
    if labels.isna().any() or not np.isfinite(truth.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains missing labels or non-finite WGS values")
    return truth, predictions, labels, metadata


def load_processed_studies(directory: str | Path) -> dict[str, tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.Series, dict]]:
    """Load all ``*.npz`` study bundles in lexical filename order."""
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No study bundles found in {directory}")
    studies = {}
    for path in paths:
        truth, predictions, labels, metadata = load_processed_study(path)
        dataset = str(metadata.get("dataset", path.stem))
        if dataset in studies:
            raise ValueError(f"Duplicate dataset identifier: {dataset}")
        studies[dataset] = (truth, predictions, labels, metadata)
    return studies
