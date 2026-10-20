# Completed Table 2.2 expansion benchmark

This directory is an additive extension of the repository's frozen starting
point. Nothing outside `expansion_benchmark/` is required to change when this
analysis is rebuilt. The runner reads completed processed matrices from the
research workspace, never writes to that workspace, and writes every generated
artifact here.

The analysis asks how two already-computed amplicon functional-inference
methods—PICRUSt2 and Tax4Fun2—behave across every completed, ready-to-use study
scope shown in black in Table 2.2 of the chapter draft. It performs no model
training and does not use raw reads, assemblies, cluster jobs, LOSO machinery,
or fitted adaptation models.

## Scope

The profile benchmark contains 18 completed expansion study scopes and 1,096
paired samples:

| Study scope | Accession | Samples |
| --- | --- | ---: |
| Han River | PRJNA530373 | 45 |
| Topsoil | PRJEB10725 | 18 |
| Empty Quarter | project dataset | 343 |
| Ocean Sampling Day | PRJEB8682 | 60 |
| Bonneville Salt Flats | PRJNA522308 | 16 |
| Plastic biofilm | PRJNA1195827 | 73 |
| Oral plaque | PRJEB98488 | 28 |
| Human skin CTCL | PRJEB36953 | 36 |
| Human skin A | PRJNA1162089 | 202 |
| Human skin B | PRJNA471898 | 44 |
| Olive xylem | PRJNA629675 | 24 |
| Activated sludge | PRJNA484416 | 60 |
| Engineered bioreactor | PRJEB36484 | 9 |
| EMP500 lake/stream sediment | PRJEB42019 | 37 |
| EMP500 marine sediment | PRJEB42019 | 64 |
| EMP500 shipworm tissue | PRJEB42019 | 14 |
| EMP500 pitcher plant fluid | PRJEB42019 | 13 |
| EMP500 lichen thallus | PRJEB42019 | 10 |

These are expansion scopes only; they are not presented as the repository's
total number of benchmark studies. Orange/not-ready Table 2.2 entries are
excluded rather than represented as completed.

Profile metrics are calculated for all 18 scopes. Biological-group metrics are
calculated only for the completed label designs that can define a contrast:
three accepted controls, topsoil biome, two separate olive seasons, and three
paired project contrasts. The exact design and status of each contrast are in
`results/contrast_manifest.tsv`. Other completed profile studies remain in the
profile benchmark but are not assigned invented biological groups.

## Metric provenance

Published precedents are credited rather than presented as new metrics:

- Profile correlation and functional-profile comparison follow the benchmark
  precedent of Langille et al. (2013) and Douglas et al. (2020).
- Raw-*p* differential-KO precision, recall, F1, and 20 prediction-label
  randomizations are derived from Matchado et al. (2024). Here each model is
  evaluated on its pairwise WGS/model native KO intersection; this KO-surface
  rule is a repository implementation choice.
- Signed group-inference correlation and 100 WGS-abundance shuffles are derived
  from Sun et al. (2020).

The BH-adjusted recovery table, WGS-versus-predicted effect relationships,
directional precision-recall/ROC summaries, AUCs, and prediction-ranked decile
composition are project extensions. In particular, BH *q* < 0.05 is not
described as Matchado's original threshold: Matchado-style recovery uses
unadjusted *p* < 0.05. The project extensions use the frozen WGS family detected
in at least `max(5, ceil(10% × n))` labeled samples, then apply BH correction
within that family. Paired designs use signed-rank tests and matched-pairs
rank-biserial effects; that paired analogue is also a project extension.

References:

- Langille MGI et al. (2013), *Nature Biotechnology*.
  <https://doi.org/10.1038/nbt.2676>
- Douglas GM et al. (2020), *Nature Biotechnology*.
  <https://doi.org/10.1038/s41587-020-0548-6>
- Matchado MS et al. (2024), *Microbial Genomics*.
  <https://doi.org/10.1099/mgen.0.001203>
- Sun S et al. (2020), *Microbiome*.
  <https://doi.org/10.1186/s40168-020-00815-y>

See the repository-level `docs/REFERENCES.md` for the full citation and
ownership map.

## Rebuild

From the repository root, with the package and notebook dependencies installed:

```bash
python expansion_benchmark/run_expansion_benchmark.py --overwrite
jupyter nbconvert --to notebook --execute --inplace \
  expansion_benchmark/notebooks/01_completed_table_2_2_expansion_benchmark.ipynb
```

The source root defaults to
`/home/hernan_melmoth/Documents/phd_work/functions_inference_benchmark_EQ` and
can be overridden with `PICRUST_BENCHMARK_SOURCE_ROOT`. It is treated as
read-only. `--overwrite` removes and recreates only the generated
`expansion_benchmark/results/` and `expansion_benchmark/figures/` directories.

`run_expansion_benchmark.py` records source paths, byte sizes, and SHA-256
digests in `results/input_file_manifest.tsv`; output hashes are stored in
`OUTPUT_MANIFEST.tsv`. A successful run writes `RUN_COMPLETE.json` only after
all nine completed WGS screen references pass at absolute tolerance `1e-12`.

## Outputs and known unavailable cells

- `results/profile_*`: sample, KO, richness, coverage, and summary metrics.
- `results/matchado_*` and `results/sun_*`: paper-derived unpaired metrics.
- `results/project_bh_*`, `effect_summary.tsv`, `directional_*`, and
  `decile_*`: project extensions.
- `results/paper_metric_availability.tsv` and
  `results/project_metric_availability.tsv`: explicit method/contrast
  availability, including zero-total prediction cases.
- `figures/`: paper-inspired profile, Matchado, and Sun figures plus project
  directional and decile figures.
- `notebooks/01_completed_table_2_2_expansion_benchmark.ipynb`: executed report
  and usage guide.

Tax4Fun2 has a small number of zero-total prediction rows on some evaluation
surfaces. Those cells are recorded as unavailable; samples or matched units are
not silently deleted, and numerical comparisons are not invented.

