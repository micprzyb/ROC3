# Notebooks

Four executed notebooks, outputs embedded. Read them in order.

| notebook | question it answers |
|---|---|
| [`01_data_and_the_identification_problem`](01_data_and_the_identification_problem.ipynb) | Is the question answerable on this data, and why does the obvious regression fail? |
| [`02_model_zoo`](02_model_zoo.ipynb) | Seven estimators — what each assumes, what each recovers. |
| [`03_hyperparameter_optimization`](03_hyperparameter_optimization.ipynb) | **How do you tune a model whose target you cannot observe?** |
| [`04_comparison_and_selection`](04_comparison_and_selection.ipynb) | Is the heterogeneity real, and what is it worth? |

Each `build_NN_*.py` regenerates its notebook from source, which is how they are maintained
— edit the builder, not the `.ipynb`:

```bash
python build_03_hpo.py
PYTHONPATH=.. jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 03_hyperparameter_optimization.ipynb
```

**Scale.** Notebooks 2–4 subsample to `N_PRODUCTS = 700`–`800` whole products so they
execute in minutes; set it to `None` for the full 3,218-product panel. Notebook 1 uses the
full panel throughout. Subsampling is by **whole product** — dropping random rows would tear
holes in the lag structure and break every time-series feature. The full-scale numbers are
in [`../docs/ELASTICITY_MODELS.md`](../docs/ELASTICITY_MODELS.md); the qualitative
conclusions do not change with scale, with one instructive exception noted in §2.9 (the
R-learner needs more data than its constant-effect sibling).

**Runtime**, on a 24-core host with nothing else running: notebook 1 ~10 min, notebook 2
~15 min, notebook 3 ~90 min, notebook 4 ~45 min. Notebook 3 is the expensive one because it
runs twelve independent hyperparameter searches.

**Threads.** Do not set `n_jobs=-1` for LightGBM here. It measured 340× slower than
`n_jobs=8` on an idle host and 849× under load, because the trees are small and the threads
spend their time synchronising. `elasticity_lab.models.N_THREADS` handles it; override with
the `ELASTICITY_LAB_THREADS` environment variable.

**Data.** The first call downloads ~45 MB from the UCI repository and caches it under
`../data/` (gitignored). Everything after that reads the cached parquet.
