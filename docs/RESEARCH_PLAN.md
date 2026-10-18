# Living research plan

**Status:** active working document  
**Last updated:** 2026-10-18  
**Purpose:** retain corrections, decisions, and priorities outside chat memory.

## Scope correction

The project does **not** consist of 15 studies total. Its evidence base has two
distinct components:

1. the established benchmark-source studies; and
2. **15 expansion studies** considered alongside them.

Every registry, table, figure, and analysis must preserve that distinction.
The definitive count, eligibility state, and role of every study will be read
from the project registry rather than inferred from this statement.

## Current priority: metric review and reusable metric library

The immediate deliverable is a small, documented Python library that can be
called for each future benchmark dataset. It will consolidate the metrics that
already exist in validated or exploratory source notebooks, without changing
their scientific interpretation.

The inventory and library must cover three families:

1. **PICRUSt-style profile metrics.** Identify every currently used
   PICRUSt/PICRUSt2 comparison metric and its input, normalization, direction,
   aggregation level, and edge-case behavior from the source notebooks.
2. **Matchado et al.-derived biological-group metrics.** Reproduce the reported
   per-KO testing/classification quantities exactly, including the WGS-defined
   reference labels and F1, precision, and recall semantics.
3. **Project extensions.** Keep these separate from published-method metrics:
   - WGS effect size versus predicted effect size relationship;
   - classification ROC and AUC analyses;
   - precision-recall analyses where already implemented; and
   - decile plots showing the composition/proportion of observed outcomes
     across ranked predicted effects.

The consolidation now uses the accepted unified5 surface rather than one
example study: 23 biological outer groups, 26 strata, 1,639 samples, and the
10 pre-audited label/phenotype contrasts. The established-metrics notebook uses
native PICRUSt2, the cached basic-MLP `picrust2_teacher_only_residual` adaptation,
and Tax4Fun2. The exploratory-effects notebook deliberately retains its validated
two-method PICRUSt2/adaptation surface.

## Explicitly out of scope for this phase

- No new project-level null analysis yet. The published Matchado 20-label-permutation
  and Sun 100-WGS-abundance-permutation controls are retained as established metrics.
- No pretrained/foundation-model work, LOSO machinery, or model-training code.
  The permitted exception is read-only scoring of the already-cached
  `teacher_only_residual` basic-MLP adaptation outputs; this notebook does not
  retrain it or import its training workflow.
- No new study processing, cluster jobs, raw-data transfers, or tool expansion.
- No claim that a metric is final merely because it exists in a notebook.

## Scientific context retained as open decisions

- The EMPO-guided broad paired-16S/WGS collection and group-effect benchmarking
  critique/extension remain the organizing themes.
- Matchado et al. (2024) compared PICRUSt2, Tax4Fun2, PanFP, and MetGEM. Its
  reported group-effect classification uses per-KO Wilcoxon testing and
  WGS-significant KOs as reference labels. The effect-size relationship is a
  project extension, not attributed to Matchado.
- The Matchado-derived replay retains unadjusted *p* < 0.05. The separate
  project recovery analysis uses BH *q* < 0.05 over one WGS-defined family to
  address simultaneous KO testing and keep a common reference across methods.
  These outputs must remain separately named and must not be treated as the
  same metric.
- Sun et al. (2020) compared PICRUSt, PICRUSt2, and the original Tax4Fun. The
  repository applies Sun's metric definition to Tax4Fun2 and labels this as a
  new application rather than an exact method-for-method reproduction.
- Directional ROC/PR, AUROC/average precision, and prediction-ranked decile
  composition are project extensions. They are not attributed to Sun,
  Matchado, PICRUSt, PICRUSt2, or Tax4Fun2.
- Metric formalization later must distinguish explicit nulls, effect size,
  sign versus magnitude, shotgun split-half ceiling, and differential-abundance
  method sensitivity.
- EMP500 assessment, the populated EMPO level-3 count, and later tool inclusion
  remain unresolved.

## Working sequence

1. Inventory the published PICRUSt2, Matchado, and Sun metric definitions.
2. Replay the complete accepted profile benchmark for PICRUSt2, the basic-MLP adaptation, and Tax4Fun2.
3. Replay the frozen 10-study corrected effect/ranking/decile surface.
4. Keep unavailable reference surfaces explicit, especially the accepted-surface Sun randomization table.
5. Review reporting-unit, threshold, and adaptation-seed choices before declaring a primary estimand.

## Change log

| Date | Correction or decision | Status |
|---|---|---|
| 2026-10-17 | Study scope corrected: benchmark-source studies plus 15 expansion studies, not 15 total studies. | Accepted user correction |
| 2026-10-17 | Metric review/library is the current priority; null analysis is deferred. | Accepted user direction |
| 2026-10-17 | Pretrained-model and LOSO implementation are excluded from this paper workspace for now. | Accepted user direction |
| 2026-10-17 | Add the existing PICRUSt2 + `teacher_only_residual` basic-MLP adaptation as a read-only cached-prediction comparator; retain its WGS-positive surface separately from the fixed 1,606-KO legacy benchmark. | Accepted user correction |
| 2026-10-17 | Replace the one-study replay with the accepted full benchmark: 23 groups, 26 strata, 1,639 samples, and all 10 audited biological contrasts. | Accepted user correction |
| 2026-10-17 | Treat Matchado and Sun as metric families, not dataset names; use the already-selected/corrected phenotype subset for project extensions. | Accepted user correction |
| 2026-10-18 | Add Tax4Fun2 to Notebook 1 and mirror the PICRUSt2, Matchado, and Sun paper figure grammars; retain Notebook 2's validated two-method scope. | Accepted user correction |
