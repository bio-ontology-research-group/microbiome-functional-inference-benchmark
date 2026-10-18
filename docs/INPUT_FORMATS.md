# Input formats

Metric functions accept ordinary sample-by-KO `pandas.DataFrame` objects.

## Functional profile matrices

- Rows are samples and the index contains unique sample identifiers.
- Columns are KEGG Orthology identifiers and are unique.
- Values are finite, numeric, and nonnegative.
- WGS and prediction matrices must contain the same samples in the same order
  unless the function explicitly performs an intersection.
- Functions that evaluate a declared WGS KO universe zero-fill KOs missing from
  the prediction before per-sample L1 normalization.

## Biological labels

Labels are a `pandas.Series` indexed by sample identifier. `group_a` and
`group_b` must be passed explicitly. Their order defines the reported effect
direction.

```python
summary, detail = matchado_differential_abundance_metrics(
    wgs,
    {"PICRUSt2": picrust2, "Tax4Fun2": tax4fun2},
    labels,
    group_a="case",
    group_b="control",
)
```

## Bundled studies

The files under `data/studies/` are compressed NumPy archives. Use:

```python
from picrust_benchmark_metrics import load_processed_study

wgs, predictions, labels, metadata = load_processed_study(
    "data/studies/matchado.npz"
)
```

The returned prediction dictionary has three keys: `PICRUSt2`, `PICRUSt2 +
adaptation`, and `Tax4Fun2`. All matrices share the ordered WGS-positive KO
universe. A missing prediction KO is represented by zero.

## Outputs

Metric functions return tidy `pandas.DataFrame` objects. Summary tables include
method names, metric values, sample or KO counts, and explicit status fields
for undefined results. Plot functions accept those tables and return the
Matplotlib figure and axes so labels or styles can be adjusted and saved by the
caller.
