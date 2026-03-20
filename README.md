# lazyopt

Lightweight Bayesian hyperparameter optimization using [HEBO](https://github.com/huawei-noah/HEBO).

## Install

```bash
git clone https://github.com/kegl/lazyopt.git
cd lazyopt
pip install -e ".[examples]"
```

## Quick start

Hyperparameters can be scattered across multiple files in your project. Each file declares its own `hp()` calls, and the optimizer collects them all. The example below has three files: feature engineering, model, and optimizer.

### Feature engineering ([`examples/feature_engineering.py`](examples/feature_engineering.py))

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from lazyopt import hp

poly_degree = hp("poly_degree", "int", 1, values=[1, 2, 3])
use_interactions = hp("use_interactions", "int", 0, values=[0, 1])


def transform(X):
    POLY_DEGREE = int(poly_degree)
    USE_INTERACTIONS = int(use_interactions)

    if POLY_DEGREE > 1:
        interaction_only = not bool(USE_INTERACTIONS)
        poly = PolynomialFeatures(
            degree=POLY_DEGREE, interaction_only=interaction_only, include_bias=False
        )
        X = poly.fit_transform(X)

    X = StandardScaler().fit_transform(X)
    return X
```

### Model ([`examples/lgbm_classifier.py`](examples/lgbm_classifier.py))

```python
from lightgbm import LGBMClassifier

from lazyopt import hp

# Hyperparameters with inline grids
lr = hp("learning_rate", "float", 0.1, values=[0.005, 0.01, 0.05, 0.1, 0.2, 0.3])
max_depth = hp("max_depth", "int", 5, values=[3, 4, 5, 6, 7, 8, 10])

# Hyperparameters whose grids are defined in hyperparams.yaml
n_estimators = hp("n_estimators", "int", 100)
num_leaves = hp("num_leaves", "int", 31)
min_child_samples = hp("min_child_samples", "int", 20)


def get_classifier():
    # Resolve proxies to trial values (must be inside a function, not at module level)
    LR = float(lr)
    MAX_DEPTH = int(max_depth)
    N_ESTIMATORS = int(n_estimators)
    NUM_LEAVES = int(num_leaves)
    MIN_CHILD_SAMPLES = int(min_child_samples)

    return LGBMClassifier(
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        n_estimators=N_ESTIMATORS,
        num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES,
        verbose=-1,
    )
```

Each file can be imported and used on its own — outside of optimization, all `hp()` proxies resolve to their defaults.

### Optimizer ([`examples/optimize.py`](examples/optimize.py))

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

from lazyopt import HyperOptimizer

from feature_engineering import transform
from lgbm_classifier import get_classifier

X, y = load_breast_cancer(return_X_y=True)


def objective():
    X_transformed = transform(X)
    clf = get_classifier()
    scores = cross_val_score(clf, X_transformed, y, cv=5, scoring="accuracy")
    return 1 - np.mean(scores)


if __name__ == "__main__":
    opt = HyperOptimizer(
        source_files=[
            "examples/feature_engineering.py",
            "examples/lgbm_classifier.py",
        ],
        yaml_config="examples/hyperparams.yaml",
        n_iterations=30,
        results_path="lgbm_results.csv",
        seed=42,
    )
    results = opt.run(objective)
    print(f"\n{results}")
```

The optimizer script has no `hp()` calls — it just defines the objective and points `source_files` to all files that contain hyperparameters. `HyperOptimizer` reads the search space by AST-parsing those files (it never imports them).

### Running the example

```bash
python examples/optimize.py
```

```
[1/30] score=0.026362  params={'learning_rate': 0.3, 'max_depth': 3, ...}
[2/30] score=0.040413  params={'learning_rate': 0.05, 'max_depth': 6, ...}
...
Best score: 0.022854
```

## How it works

**1. Declare hyperparameters** with `hp()` in your model or pipeline files. Each call takes a name, a type, and a default. You can provide the grid of candidate values inline:

```python
lr = hp("learning_rate", "float", 0.1, values=[0.005, 0.01, 0.05, 0.1, 0.2, 0.3])
```

or omit `values` and define the grid in a YAML file instead:

```python
n_estimators = hp("n_estimators", "int", 100)
```

```yaml
# hyperparams.yaml — namespace must match the source filename stem
lgbm_classifier:
  n_estimators:
    dtype: int
    default: 100
    values: [50, 100, 200, 300, 500]
```

**2. Resolve hyperparameters** by casting to the correct type inside a function. The convention is lowercase proxy, uppercase resolved value:

```python
def get_classifier():
    # Must be inside a function, not at module level
    LR = float(lr)
    MAX_DEPTH = int(max_depth)
    ...
```

The optimizer sets a trial context before each call to `objective()`, so `float(lr)` returns the HEBO-suggested value for that trial. At module level, `float(lr)` would just return the default once and never change.

**3. Return a score to minimize.** The objective calls the model and returns a scalar. Since we want to maximize accuracy, we return the error rate:

```python
def objective():
    X_transformed = transform(X)
    clf = get_classifier()
    scores = cross_val_score(clf, X_transformed, y, cv=5, scoring="accuracy")
    return 1 - np.mean(scores)
```

**4. Run the optimizer.** List all files that contain `hp()` calls in `source_files`:

```python
opt = HyperOptimizer(
    source_files=[
        "examples/feature_engineering.py",
        "examples/lgbm_classifier.py",
    ],
    yaml_config="examples/hyperparams.yaml",
    n_iterations=30,
    results_path="lgbm_results.csv",
)
results = opt.run(objective)
```

## Crash recovery and resume

Results are saved to `results_path` after every trial (atomically, so a crash mid-write can't corrupt the file). If the process crashes, re-run the same script — completed trials are reloaded and optimization continues from where it left off:

```python
# First run: completes 12 of 30 trials, then crashes
# Second run: loads 12 trials from results.csv, runs 13–30
results = opt.run(objective)
```

## API

| Symbol | Description |
|--------|-------------|
| `hp(name, dtype, default, values=None)` | Declare a hyperparameter |
| `HyperOptimizer(source_files, yaml_config=None, n_iterations=50, results_path="results.csv", seed=42)` | Create optimizer |
| `HyperOptimizer.run(objective)` | Run optimization, returns `TrialResults` |
| `TrialResults.best_score` | Best (minimum) score |
| `TrialResults.best_params` | Parameters of the best trial |
| `TrialResults.from_csv(path)` | Load results from a previous run |
| `TrialResults.to_csv(path)` / `to_json(path)` | Export results |
