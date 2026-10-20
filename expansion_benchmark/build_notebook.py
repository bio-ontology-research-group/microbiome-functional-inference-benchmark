#!/usr/bin/env python3
"""Build the lightweight executed-report notebook for the expansion benchmark."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks" / "01_completed_table_2_2_expansion_benchmark.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nb["cells"] = [
    md("""
    # Completed Table 2.2 expansion benchmark

    This report is separate from the repository's two frozen starting notebooks.
    It summarizes a no-training replay over all 18 completed/available expansion
    scopes in Table 2.2, using only PICRUSt2 and Tax4Fun2 predictions. The runner
    reads processed source matrices and writes only below `expansion_benchmark/`.

    Set `RERUN_BENCHMARK = True` in the next cell to regenerate every result and
    figure. The committed execution reads and validates the generated artifacts.
    """),
    code("""
    from pathlib import Path
    import json
    import subprocess
    import sys
    import pandas as pd
    from IPython.display import Image, Markdown, display

    HERE = Path.cwd()
    if HERE.name == "notebooks":
        ROOT = HERE.parent
    elif (HERE / "expansion_benchmark").exists():
        ROOT = HERE / "expansion_benchmark"
    else:
        raise RuntimeError("Run from the repository root or expansion_benchmark/notebooks")

    RERUN_BENCHMARK = False
    if RERUN_BENCHMARK:
        subprocess.run([sys.executable, str(ROOT / "run_expansion_benchmark.py"), "--overwrite"], check=True)

    receipt = json.loads((ROOT / "RUN_COMPLETE.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["methods"] == ["PICRUSt2", "Tax4Fun2"]
    assert receipt["training_performed"] is False
    receipt
    """),
    md("""
    ## Study surface

    All black completed/available Table 2.2 scopes are included for profile
    metrics. These are expansion scopes, not the total number of benchmark
    studies in the repository. Orange/not-ready entries are excluded.
    """),
    code("""
    study = pd.read_csv(ROOT / "results/study_status.tsv", sep="\\t")
    assert len(study) == 18 and study.n_samples.sum() == 1096
    assert not study.training_performed.any()
    display(study)
    """),
    md("""
    ## Profile metrics across all completed scopes

    The tables use the reusable functions under `src/picrust_benchmark_metrics/`:
    sample-wise Spearman on the WGS-positive universe, native pairwise profile
    metrics, richness overlap, per-KO Spearman, and KO coverage. A zero-total
    method row on the WGS universe is retained as an explicit undefined result.
    """),
    code("""
    profile = pd.read_csv(ROOT / "results/profile_summary.tsv", sep="\\t")
    display(profile.pivot(index=["display_name", "n_samples"], columns="method_label",
                          values=["mean_sample_spearman", "mean_wgs_richness_recovered"]))
    display(Image(filename=str(ROOT / "figures/01_profile_spearman_all_completed_expansions.png")))
    display(Image(filename=str(ROOT / "figures/02_ko_coverage_all_completed_expansions.png")))
    display(Image(filename=str(ROOT / "figures/10_profile_mean_summary.png")))
    """),
    md("""
    ## Biological contrasts and availability

    Group metrics are restricted to completed label designs. Accepted controls,
    explicit metadata contrasts, provisional designs, and zero-BH contrasts are
    kept visibly distinct. No group is invented for profile-only studies.
    """),
    code("""
    contrasts = pd.read_csv(ROOT / "results/contrast_manifest.tsv", sep="\\t")
    paper_availability = pd.read_csv(ROOT / "results/paper_metric_availability.tsv", sep="\\t")
    project_availability = pd.read_csv(ROOT / "results/project_metric_availability.tsv", sep="\\t")
    display(contrasts)
    display(paper_availability)
    display(project_availability)
    """),
    md("""
    ## Matchado-derived recovery and randomization

    This section uses two-sided rank-sum tests and **unadjusted** *p* < 0.05 for
    significant-KO set recovery, plus 20 prediction-label randomizations. These
    definitions are derived from Matchado et al. (2024). Pairwise WGS/model KO
    intersections are this repository's stated surface choice. AUROC/AUPR rows
    returned by the function are project additions, not Matchado metrics.
    """),
    code("""
    matchado = pd.read_csv(ROOT / "results/matchado_raw_p_summary.tsv", sep="\\t")
    display(matchado[matchado.metric.isin(["precision", "recall", "f1"])])
    display(Image(filename=str(ROOT / "figures/03_matchado_raw_p_recovery.png")))
    display(Image(filename=str(ROOT / "figures/04_matchado_label_randomization.png")))
    """),
    md("""
    ## Sun-derived inference correlation and WGS-abundance controls

    Signed `log10(p)` WGS-versus-prediction correlation and 100 independent WGS
    KO-abundance shuffles are derived from Sun et al. (2020). The shuffled
    control changes WGS abundances while holding group labels and predictions
    fixed.
    """),
    code("""
    sun = pd.read_csv(ROOT / "results/sun_summary.tsv", sep="\\t")
    display(sun)
    display(Image(filename=str(ROOT / "figures/05_sun_observed_and_wgs_abundance_controls.png")))
    """),
    md("""
    ## Project extensions

    These are not attributed to the benchmark papers. The WGS target family
    requires detection in at least `max(5, ceil(10% × n))` labeled samples and
    uses BH-adjusted *q* values within that family. Unpaired effects are
    rank-biserial; paired effects use signed ranks after averaging replicates
    within each matched side. Directional PR/ROC and deciles use WGS *q* ≤ 0.05
    plus WGS effect sign as the observed target and continuous prediction effect
    as the ranking score.
    """),
    code("""
    bh = pd.read_csv(ROOT / "results/project_bh_summary.tsv", sep="\\t")
    effects = pd.read_csv(ROOT / "results/effect_summary.tsv", sep="\\t")
    direction = pd.read_csv(ROOT / "results/directional_metric_summary.tsv", sep="\\t")
    display(bh)
    display(effects)
    display(direction)
    display(Image(filename=str(ROOT / "figures/06_directional_average_precision.png")))
    display(Image(filename=str(ROOT / "figures/07_directional_roc_auc.png")))
    display(Image(filename=str(ROOT / "figures/08_directional_roc_curves.png")))
    display(Image(filename=str(ROOT / "figures/09_directional_decile_composition.png")))
    """),
    md("""
    ## Numerical validation against completed source results

    The nine completed WGS contrast screens are independently replayed. Passing
    requires exact ordered KO identity and maximum absolute differences no
    larger than `1e-12` for p-values, BH q-values, and effect sizes. The Empty
    Quarter profile surface has 343 strict paired samples; its historical screen
    comparison deliberately uses the exact frozen 341-sample contrast contract.
    """),
    code("""
    validation = pd.read_csv(ROOT / "results/wgs_screen_validation.tsv", sep="\\t")
    assert validation.validation_pass.all()
    assert validation.ko_identity_exact.all()
    numeric = validation.filter(regex="^max_abs_diff_")
    assert numeric.max().max() <= 1e-12
    display(validation)
    print("Largest absolute numerical difference:", numeric.max().max())
    """),
    md("""
    ## Interpretation boundaries still requiring scientific decisions

    - Provisional Empty Quarter compartment and Han River seasonal contrasts
      remain descriptive and must not be treated as causal validation.
    - Topsoil biome is confounded with site; olive seasons are kept separate.
    - Paired signed-rank analyses are project analogues, not original Matchado or
      Sun definitions.
    - Tax4Fun2 zero-total cells are explicitly unavailable; matched units are not
      deleted to force a score.
    - Group-based conclusions do not extend to the completed profile-only
      studies lacking an accepted or documented contrast.
    """),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
