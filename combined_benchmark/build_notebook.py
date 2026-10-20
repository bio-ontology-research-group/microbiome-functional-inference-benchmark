#!/usr/bin/env python3
"""Create the executable combined environment-summary notebook."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks" / "01_combined_environment_summary.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "bio_ontology_env", "language": "python", "name": "bio_ontology_env"
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
nb["cells"] = [
    markdown(
        """
# Combined benchmark and environment summary

This notebook consolidates the **benchmark-source studies plus the completed
expansion studies**. It does not treat “15 expansion studies” as the total
benchmark. The comparison is restricted to native **PICRUSt2** and
**Tax4Fun2** predictions; no fitting or training occurs.

The profile section uses source-project × biological-environment scopes so
technical strata do not inflate the study count. The group section uses only
accepted contrasts with usable labels and keeps paper-derived metric families
separate from repository extensions.
"""
    ),
    code(
        """
from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
from IPython.display import Image, display

cwd = Path.cwd().resolve()
if (cwd / "combined_benchmark").exists():
    REPO = cwd
elif cwd.name == "combined_benchmark":
    REPO = cwd.parent
else:
    candidates = [cwd, *cwd.parents]
    REPO = next(path for path in candidates if (path / "combined_benchmark").exists())
HERE = REPO / "combined_benchmark"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
METHODS = ["PICRUSt2", "Tax4Fun2"]

receipt = json.loads((HERE / "RUN_COMPLETE.json").read_text())
receipt
"""
    ),
    markdown(
        """
## 1. Coverage census

There are three deliberately different counts: unique source projects,
source-project × environment scopes, and paired samples. Profile metrics use
all paired samples, but the environment macro summary gives each scope equal
weight.
"""
    ),
    code(
        """
registry = pd.read_csv(RESULTS / "combined_sample_environment_registry.tsv.gz", sep="\t")
scope = pd.read_csv(RESULTS / "scope_profile_summary.tsv", sep="\t")
environment = pd.read_csv(RESULTS / "environment_profile_summary.tsv", sep="\t")

census = pd.DataFrame({
    "quantity": ["paired samples", "benchmark-source samples", "expansion samples",
                 "unique source projects", "source-project × environment scopes"],
    "value": [registry.sample_key.nunique(),
              registry.provenance.eq("benchmark_source").sum(),
              registry.provenance.eq("expansion").sum(),
              registry.source_project_id.nunique(),
              registry.environment_scope_id.nunique()],
})
display(census)
display(Image(filename=str(FIGURES / "01_environment_coverage_census.png"), width=1000))
"""
    ),
    markdown(
        """
## 2. Profile agreement by environment

The dumbbell plot reports the mean sample-wise Spearman correlation within
each source-project × environment scope. The paired-difference plot subtracts
PICRUSt2 from Tax4Fun2 within the same samples and uses deterministic 2,000-run
sample-bootstrap intervals. The macro panel then summarizes scope means—not
pooled samples—within the broad environment hierarchy.
"""
    ),
    code(
        """
summary = scope.pivot_table(
    index=["empo_1", "empo_2", "empo_3", "display_name", "provenance"],
    columns="method_label", values="mean_sample_spearman"
).reset_index()
summary["Tax4Fun2_minus_PICRUSt2"] = summary["Tax4Fun2"] - summary["PICRUSt2"]
display(summary.sort_values(["empo_1", "empo_2", "empo_3", "display_name"]).reset_index(drop=True))
display(Image(filename=str(FIGURES / "02_scope_profile_dumbbell.png"), width=1050))
display(Image(filename=str(FIGURES / "03_scope_method_difference.png"), width=1050))
display(Image(filename=str(FIGURES / "04_environment_macro_profile.png"), width=950))
"""
    ),
    markdown(
        """
The figures are descriptive. A higher value for one method in one environment
scope is not evidence of universal method superiority. Scope means also have
different sampling precision, shown in the result table, while the macro
summary intentionally weights scopes equally.
"""
    ),
    markdown(
        """
## 3. Biological-group contrasts and metric availability

Eight original benchmark contrasts and nine completed expansion contrasts
form the 17-row registry. Matchado-style and Sun-style metrics require
unpaired two-group designs, so the three paired expansion contrasts appear in
the project effect/directional outputs but not those paper-derived families.
Tax4Fun2 olive-season-1 group results are unavailable because two prediction
samples have zero native totals; no value is imputed.
"""
    ),
    code(
        """
contrasts = pd.read_csv(RESULTS / "combined_contrast_registry.tsv", sep="\t")
effect = pd.read_csv(RESULTS / "combined_effect.tsv", sep="\t")
directional = pd.read_csv(RESULTS / "combined_directional.tsv", sep="\t")
matchado = pd.read_csv(RESULTS / "combined_matchado.tsv", sep="\t")
bh = pd.read_csv(RESULTS / "combined_bh.tsv", sep="\t")
sun = pd.read_csv(RESULTS / "combined_sun.tsv", sep="\t")

rows = []
for contrast in contrasts.itertuples(index=False):
    for method in METHODS:
        rows.append({
            "contrast_id": contrast.contrast_id,
            "provenance": contrast.provenance,
            "design": contrast.design,
            "method": method,
            "effect_directional": ((effect.contrast_id == contrast.contrast_id) & (effect.prediction_method == method)).any(),
            "matchado_raw_p": ((matchado.contrast_id == contrast.contrast_id) & (matchado.prediction_method == method)).any(),
            "project_bh": ((bh.contrast_id == contrast.contrast_id) & (bh.prediction_method == method)).any(),
            "sun": ((sun.contrast_id == contrast.contrast_id) & (sun.prediction_method == method)).any(),
        })
availability = pd.DataFrame(rows)
display(availability)
"""
    ),
    markdown(
        """
## 4. Paper-derived metric families

The Matchado-style rows use unadjusted two-sided *p* < 0.05, WGS-significant
KOs as the reference, and 20 prediction-label randomizations. The Sun-style
rows correlate signed `log10(p)` group statistics and compare the observation
with 100 independent WGS KO-abundance shuffles. These definitions and plot
grammars are credited to the cited papers; applying them to this study set and
to Tax4Fun2 is the repository's implementation.
"""
    ),
    code(
        """
display(Image(filename=str(FIGURES / "05_matchado_raw_p_f1.png"), width=1050))
display(Image(filename=str(FIGURES / "06_sun_inference_correlation.png"), width=1050))
"""
    ),
    markdown(
        """
## 5. Repository extensions

The effect-size relation, directional ROC/precision-recall metrics, and BH
recovery table are repository extensions. They are not attributed to
PICRUSt/PICRUSt2, Sun, Matchado, or Tax4Fun2. BH is applied within each complete
KO family and uses *q* < 0.05 to limit the expected false-discovery proportion
across simultaneous KO tests.
"""
    ),
    code(
        """
display(Image(filename=str(FIGURES / "07_effect_relationship.png"), width=1050))
display(Image(filename=str(FIGURES / "08_directional_roc_auc.png"), width=1200))
"""
    ),
    markdown(
        """
## 6. Numerical validation

The validator checks all input/output hashes, census invariants, explicit
unavailable values, and completed-result agreement at absolute tolerance
`1e-12`. It also verifies the newly recomputed original PICRUSt2 effect and
directional rows against the frozen completed references before accepting the
parallel Tax4Fun2 calculations.
"""
    ),
    code(
        """
validation = subprocess.run(
    [sys.executable, str(HERE / "validate_outputs.py")],
    cwd=REPO, check=True, capture_output=True, text=True,
)
print(validation.stdout)
"""
    ),
    markdown(
        """
## 7. Choices to resolve before a manuscript-wide primary summary

- Retain the frozen EMPO-style hierarchy or conduct a separate reviewed
  environment-ontology pass.
- Keep equal scope weights (current), use sample weights, or fit a hierarchical
  model. Each estimates a different target.
- Place provisional/control expansion contrasts in the main figure or a
  supplement.
- Keep Brooks reverse-read results supplemental (current main-plot choice) or
  show forward and reverse technical strata together.

These are scientific presentation choices, not missing computations.
"""
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
