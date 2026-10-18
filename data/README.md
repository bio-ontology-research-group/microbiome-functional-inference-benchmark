# Included processed benchmark data

This directory contains only the processed inputs and completed result tables
needed by the two notebooks. It does not contain amplicon reads, WGS reads,
assemblies, model weights, training inputs, or cluster outputs.

## `studies/`

Ten accepted biological contrasts are distributed as compressed NumPy files.
Each file contains:

- `sample_ids` and `ko_ids`;
- the processed sample-by-KO WGS matrix (`wgs`);
- processed sample-by-KO predictions from PICRUSt2, PICRUSt2 plus the basic-MLP
  adaptation, and Tax4Fun2;
- the audited binary group label for each sample; and
- the declared `group_a` and `group_b` direction.

Prediction matrices are stored on the WGS-positive KO universe. KOs absent
from a method are represented by zero, matching the notebook evaluation rule.
Use `picrust_benchmark_metrics.load_processed_study` to load one file.

## `reference/`

These are compact completed output tables used for numerical regression checks
and for the 26-stratum overview. The full profile overview contains **11
benchmark strata plus 15 expansion strata**. The ten matrix bundles are the
subset with accepted biological-group labels.

The reference directory contains results from two intentionally different
differential-KO definitions. `matchado_label_metric_results_long.tsv` uses the
Matchado-derived unadjusted *p* < 0.05 rule. `classification_summary.tsv` and
the effect/ranking/decile tables use the project's BH-adjusted WGS reference
(*q* < 0.05 for the frozen selection; the ranking/decile classification helper
uses *q* ≤ 0.05). These result families are not interchangeable. The BH version
was introduced to address simultaneous testing across thousands of KOs and to
hold one WGS-defined reference family fixed across methods.

`manifest.tsv` records the byte size and SHA-256 digest of every distributed
file. `source_artifact_manifest.tsv` records the frozen source artifacts used
by the maintainer build script. Paths are relative to the separate source
workspace.

The repository license covers the code. These derived matrices retain the
provenance and reuse conditions of their originating public studies; consult
the study accessions and paper documentation before redistributing them in a
separate product.
