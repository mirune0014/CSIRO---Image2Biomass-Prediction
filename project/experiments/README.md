# Experiments Directory

- Put each experiment under a dedicated folder, e.g. `exp001/`, `exp002/`.
- Keep changes single-scoped (one change per experiment) in code or notebooks.
- Reference this path from your experiment report using `docs/EXPERIMENT_TEMPLATE.md`.
- Save any ad-hoc artifacts here that are not large (plots, small tables). Use `project/results/` for metrics/logs.

Structure example:

```
project/
  experiments/
    exp002/
      exp002.ipynb
      feature_ndvi.py
```

