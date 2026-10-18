"""Small, model-free metrics for paired functional-inference benchmarks.

This package preserves established project metric semantics.  It deliberately
does not choose an estimand, construct a null analysis, or include training or
LOSO code.
"""

from .profile import (
    align_to_ko_universe,
    binary_ranking_metrics,
    normalize_rows,
    per_ko_spearman,
    per_sample_richness_overlap,
    profile_overlap,
    samplewise_profile_metrics,
    samplewise_spearman,
    samplewise_spearman_on_universe,
    safe_spearman,
    within_study_k_shot_mean_benchmark,
)
from .sun import (
    permute_ko_abundances_across_samples,
    sun_inference_correlation_abundance_randomization,
    sun_signed_wilcoxon_by_ko,
)
from .matchado import (
    benjamini_hochberg,
    classification_set_metrics,
    matchado_bh_shared_wgs_reference_metrics,
    matchado_differential_abundance_metrics,
    r_default_wilcoxon_by_ko,
    wilcoxon_rank_sum_by_ko,
)
from .effects import biological_group_effect_concordance, rank_biserial_effect_by_ko
from .ranking import (
    decile_outcome_composition,
    directional_ranking_curves,
    plot_decile_outcome_composition,
)
from .datasets import load_processed_studies, load_processed_study
from .plotting import (
    plot_directional_metric_summary,
    plot_directional_roc_grid,
    plot_ko_coverage,
    plot_matchado_randomization_dumbbells,
    plot_method_differences,
    plot_metric_row_grid,
    plot_profile_study_panels,
    plot_signed_decile_grid,
    plot_stratum_mean_ranges,
    plot_sun_observed_controls,
)

__all__ = [
    "normalize_rows",
    "align_to_ko_universe",
    "binary_ranking_metrics",
    "per_ko_spearman",
    "per_sample_richness_overlap",
    "profile_overlap",
    "samplewise_profile_metrics",
    "samplewise_spearman",
    "samplewise_spearman_on_universe",
    "safe_spearman",
    "within_study_k_shot_mean_benchmark",
    "classification_set_metrics",
    "benjamini_hochberg",
    "r_default_wilcoxon_by_ko",
    "matchado_bh_shared_wgs_reference_metrics",
    "matchado_differential_abundance_metrics",
    "wilcoxon_rank_sum_by_ko",
    "permute_ko_abundances_across_samples",
    "sun_inference_correlation_abundance_randomization",
    "sun_signed_wilcoxon_by_ko",
    "biological_group_effect_concordance",
    "rank_biserial_effect_by_ko",
    "directional_ranking_curves",
    "decile_outcome_composition",
    "plot_decile_outcome_composition",
    "load_processed_study",
    "load_processed_studies",
    "plot_profile_study_panels",
    "plot_stratum_mean_ranges",
    "plot_method_differences",
    "plot_ko_coverage",
    "plot_metric_row_grid",
    "plot_matchado_randomization_dumbbells",
    "plot_sun_observed_controls",
    "plot_directional_metric_summary",
    "plot_directional_roc_grid",
    "plot_signed_decile_grid",
]
