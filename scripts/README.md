# Scripts

- `replay_bundled_group_metrics.py` reruns the profile, Matchado et al.-derived,
  and Sun et al.-derived calculations from the processed files committed under
  `data/studies/`. This is the script used by Notebook 1.
- `build_benchmark_notebooks.py` generates the two notebooks from their
  version-controlled source cells.
- `build_public_data_bundle.py` and `replay_published_group_metrics.py` are
  maintainer scripts for rebuilding the distributed data from the separate
  frozen evidence workspace. They are not required to run the published
  notebooks.

Repository users need only install the package and execute the notebooks.
