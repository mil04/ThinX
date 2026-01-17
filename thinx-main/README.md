ThinX
=====

ThinX is an installable Python package for distribution compression and post-hoc explainability on tabular data. It combines data handling, compression pipelines, explainers, and evaluation utilities into a single standalone tool for reproducible experimentation.

Features
--------
- Data loading from OpenML and OpenXAI with ready-to-train models.
- Compression methods: kernel thinning, Stein thinning, influence-based selection,
  ARF (arfpy), and IID sampling.
- Explainers: SHAP (kernel/permutation), SAGE, SHAP-IQ, expected gradients,
  and influence (PyDVL).
- Evaluation utilities: MAE, Top-k agreement, pairwise overlap (SHAP-IQ),
  and MMD for compressed sets.

Installation
------------
This package is intended to be used from this repository:

```bash
pip install -e custom_packages/thinx-main
```

Notes:
- Python >= 3.10.
- Dependencies include `openxai`, `goodpoints`, and `sage`. This repo includes
  patched versions of `goodpoints` and `sage` under `custom_packages`.

Quickstart
----------
```python
import numpy as np
from thinx import DataLoader, Preprocessor, Explainer, Evaluator

# 1) Load data + model from OpenML
loader = DataLoader()
dataset_name, X_train, y_train, X_test, y_test, model, preprocessor = loader.load_from_openml(
    dataset_id=40983,
    model_name="xgboost",
)

# 2) Compress background data
prep = Preprocessor(
    X=X_train,
    y=y_train,
    model=model,
    compression_method="kernel_thinning",
    data_modification_method="none",
    seed=0,
)
X_bg, y_bg, indices, compress_time = prep.preprocess(
    g=4,
    num_bins=32,
    target_size=256,
    kernel="gaussian",
    delta=0.5,
)

# 3) Explain a small foreground subset
explainer = Explainer(
    model=model,
    explainer_name="shap",
    task_type="classification",
    strategy="kernel",
    seed=0,
)
X_fg = X_test[:50]
exp_vals, exp_time = explainer.explain(
    X_foreground=X_fg,
    X_background=X_bg,
)

# 4) Evaluate (example: compare against a reference explanation)
evaluator = Evaluator(
    ground_truth_explanation=exp_vals,
    ground_truth_points=X_train,
)
metrics = evaluator.evaluate_explanation(exp_vals, exp_time, num_samples=X_bg.shape[0])
```

Repository Notes
----------------
This package is part of a larger experimental codebase. It favors reproducibility
and explicit configurations over high-level automation. See `thinx/utils.py`
for helper functions and benchmark dataset IDs.
