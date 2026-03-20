# lazyopt

Lightweight Bayesian hyperparameter optimization using [HEBO](https://github.com/huawei-noah/HEBO).

## Install

```bash
git clone https://github.com/kegl/lazyopt.git
cd lazyopt
pip install -e ".[examples]"
```

## Quick start

The typical setup has two files: a **model file** that declares the hyperparameters and the objective, and an **optimizer script** that runs the search.

### Model file ([`examples/lgbm_classifier.py`](examples/lgbm_classifier.py))

```python
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

from lazyopt import hp

# Hyperparameters with inline grids
lr = hp("learning_rate", "float", 0.1, values=[0.005, 0.01, 0.05, 0.1, 0.2, 0.3])
max_depth = hp("max_depth", "int", 5, values=[3, 4, 5, 6, 7, 8, 10])

# Hyperparameters whose grids are defined in hyperparams.yaml
n_estimators = hp("n_estimators", "int", 100)
num_leaves = hp("num_leaves", "int", 31)
min_child_samples = hp("min_child_samples", "int", 20)

X, y = load_breast_cancer(return_X_y=True)


def objective():
    # Resolve proxies to trial values (must be inside objective, not at module level)
    LR = float(lr)
    MAX_DEPTH = int(max_depth)
    N_ESTIMATORS = int(n_estimators)
    NUM_LEAVES = int(num_leaves)
    MIN_CHILD_SAMPLES = int(min_child_samples)

    clf = LGBMClassifier(
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        n_estimators=N_ESTIMATORS,
        num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES,
        verbose=-1,
    )
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return 1 - np.mean(scores)
```

The model file is a regular Python module. It can be imported and used directly — outside of optimization, all `hp()` proxies resolve to their defaults.

### Optimizer script ([`examples/optimize.py`](examples/optimize.py))

```python
from lazyopt import HyperOptimizer

from lgbm_classifier import objective

if __name__ == "__main__":
    opt = HyperOptimizer(
        source_files=["examples/lgbm_classifier.py"],
        yaml_config="examples/hyperparams.yaml",
        n_iterations=30,
        results_path="lgbm_results.csv",
        seed=42,
    )
    results = opt.run(objective)
    print(f"\n{results}")
```

The optimizer script imports the objective and points `source_files` to the model file. `HyperOptimizer` reads the search space by AST-parsing the source file — it never imports it, so the two mechanisms are independent.

### Running the example

```bash
python examples/optimize.py
```

```
[1/30] score=0.026362  params={'learning_rate': 0.3, 'max_depth': 3, ...}
[2/30] score=0.040413  params={'learning_rate': 0.05, 'max_depth': 6, ...}
...
Best score: 0.022854
Best params: {'lgbm_classifier.learning_rate': 0.3, ...}
```

## How it works

**1. Declare hyperparameters** with `hp()` at module level. Each call takes a name, a type, and a default. You can provide the grid of candidate values inline:

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

**2. Resolve hyperparameters** inside the objective by casting to the correct type. The convention is lowercase proxy, uppercase resolved value:

```python
def objective():
    # Must be inside objective(), not at module level
    LR = float(lr)
    MAX_DEPTH = int(max_depth)
    ...
```

The optimizer sets a trial context before each call to `objective()`, so `float(lr)` returns the HEBO-suggested value for that trial. At module level, `float(lr)` would just return the default once and never change.

**3. Return a score to minimize.** Since we want to maximize accuracy, we return the error rate:

```python
    return 1 - np.mean(scores)
```

**4. Run the optimizer** from a separate script. Pass `source_files` pointing to wherever the `hp()` calls are, and optionally a `yaml_config` for grids defined externally:

```python
opt = HyperOptimizer(
    source_files=["examples/lgbm_classifier.py"],
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
opt = HyperOptimizer(
    source_files=["examples/lgbm_classifier.py"],
    yaml_config="examples/hyperparams.yaml",
    n_iterations=30,
    results_path="results.csv",
)
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
