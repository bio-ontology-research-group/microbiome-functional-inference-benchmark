# Metric attribution and references

This repository reimplements and adapts published benchmarking ideas. Naming a
metric family after a paper is attribution, not a claim that the paper's
authors contributed to this software or endorsed these implementations.

## Provenance of each metric family

| Repository analysis | Credit and status |
| --- | --- |
| Sample-wise Spearman agreement between inferred and WGS functional profiles | Published benchmarking precedent from Langille et al. (2013) and Douglas et al. (2020). The repository's explicit WGS-positive KO universe, zero-fill rule, and aggregation are documented implementation choices. |
| Raw-*p* differential-KO precision, recall, and F1 | Reimplementation of the evaluation used by Matchado et al. (2024): low-prevalence filtering, two-sided Wilcoxon rank-sum tests, unadjusted *p* < 0.05, WGS-significant KOs as the reference set, and prediction-label randomization. For direct multi-method comparison, this library uses one KO intersection and 5% prevalence surface shared by WGS and every supplied method; that all-method shared surface is a documented repository choice. |
| Observed-versus-randomized recovery dumbbell plot | Plotting grammar and 20-round prediction-label randomization follow Matchado et al. (2024), especially their Fig. 7. |
| Signed group-inference correlation | Reimplementation of Sun et al. (2020): signed `log10(p)` values from two-group Wilcoxon tests followed by a Spearman correlation between WGS and inferred profiles. |
| WGS-abundance shuffle control | Reimplementation of Sun et al. (2020): each WGS gene/KO abundance vector is shuffled independently across samples 100 times while group labels and predictions remain fixed. |
| BH-corrected shared-WGS-family recovery | Project implementation combining the FDR-based differential-KO set-recovery precedent in Douglas et al. (2020) with the set-recovery framing used by Matchado et al. (2024). Its WGS-only family and prediction zero-fill rules differ from both papers, so it is not presented as an exact metric from either paper. |
| WGS-versus-predicted effect-size relationship | Project-developed extension. |
| Directional ROC, precision-recall, ROC-AUC, and average precision | Project-developed extensions. AUROC/AUPR values returned alongside the raw-*p* Matchado replay are also library additions, not published Matchado metrics. |
| Prediction-ranked decile outcome composition | Project-developed extension. |

The figure functions reproduce useful visual grammars from the papers where
stated, but apply them to this repository's studies and methods. They are not
copies of the published figures.

## Method citations

The benchmark examples evaluate these inference methods or variants:

- **PICRUSt:** Langille, M. G. I., Zaneveld, J., Caporaso, J. G., et al.
  (2013). Predictive functional profiling of microbial communities using 16S
  rRNA marker gene sequences. *Nature Biotechnology*, 31, 814-821.
  <https://doi.org/10.1038/nbt.2676>
- **PICRUSt2:** Douglas, G. M., Maffei, V. J., Zaneveld, J. R., et al. (2020).
  PICRUSt2 for prediction of metagenome functions. *Nature Biotechnology*, 38,
  685-688. <https://doi.org/10.1038/s41587-020-0548-6>
- **Tax4Fun2:** Wemheuer, F., Taylor, J. A., Daniel, R., Johnston, E., Meinicke,
  P., Thomas, T., & Wemheuer, B. (2020). Tax4Fun2: prediction of
  habitat-specific functional profiles and functional redundancy based on 16S
  rRNA gene sequences. *Environmental Microbiome*, 15, 11.
  <https://doi.org/10.1186/s40793-020-00358-7>
- **Tax4Fun (evaluated by Sun et al., but not used as a current comparator):**
  Aßhauer, K. P., Wemheuer, B., Daniel, R., & Meinicke, P. (2015). Tax4Fun:
  predicting functional profiles from metagenomic 16S rRNA data.
  *Bioinformatics*, 31(17), 2882-2884.
  <https://doi.org/10.1093/bioinformatics/btv287>

The **PICRUSt2 + adaptation** comparator is this project's basic-MLP adaptation
of cached PICRUSt2 predictions. It is not a method introduced in any of the
papers above.

## Benchmark-metric references

- **Sun metric family:** Sun, S., Jones, R. B., & Fodor, A. A. (2020).
  Inference-based accuracy of metagenome prediction tools varies across sample
  types and functional categories. *Microbiome*, 8, 46.
  <https://doi.org/10.1186/s40168-020-00815-y>
- **Matchado metric family:** Matchado, M. S., Rühlemann, M., Reitmeier, S.,
  Kacprowski, T., Frost, F., Haller, D., Baumbach, J., & List, M. (2024). On
  the limits of 16S rRNA gene-based metagenome prediction and functional
  profiling. *Microbial Genomics*, 10(2), 001203.
  <https://doi.org/10.1099/mgen.0.001203>

Sun et al. evaluated PICRUSt, PICRUSt2, and the original Tax4Fun. The current
notebook applies their metric definitions to PICRUSt2, this project's basic-MLP
adaptation, and Tax4Fun2. Matchado et al. evaluated PICRUSt2, Tax4Fun2, PanFP,
and MetGEM; this repository reuses their metric definitions on the methods and
accepted study contrasts distributed here.
