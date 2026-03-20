# devai-hyperopt

Lightweight Bayesian hyperparameter optimization using HEBO, with a `contextvars.ContextVar`-based lazy proxy pattern.

## Install

```bash
git clone https://github.com/facebookresearch/devai-hyperopt.git
cd devai-hyperopt
pip install -e .
```

## How it works

### The proxy pattern

Declare hyperparameters at module level with `hp()`. Each call returns a `HyperProxy` — a lazy object that resolves to the **default** value normally, but to the **trial** value during optimization:

```python
from devai_hyperopt import hp

learning_rate = hp("learning_rate", "float", 0.1,
                   values=[0.01, 0.05, 0.1, 0.2])
```

Inside your objective function, cast the proxy to a concrete type:

```python
def objective():
    lr = float(learning_rate)   # resolves to trial value during optimization
    ...
```

Under the hood, `HyperOptimizer` sets a `contextvars.ContextVar` before calling `objective()`, so `float(learning_rate)` returns the HEBO-suggested value. Outside of a trial, it returns the default (`0.1`).

### Namespace auto-inference

The namespace is inferred from the caller's filename stem. For `my_model.py`, `hp("lr", ...)` registers as `my_model.lr`. This avoids collisions when multiple files declare hyperparameters.

### Search space

Grids can be declared inline (via `values=`) or in a YAML file as fallback:

```yaml
my_model:
  lr:
    dtype: float
    default: 0.1
    values: [0.01, 0.05, 0.1, 0.2]
```

The optimizer AST-parses your source files to collect the search space — no need to import your modules.

## Usage

```python
from devai_hyperopt import hp, HyperOptimizer

# Declare hyperparameters at module level
lr = hp("lr", "float", 0.1, values=[0.01, 0.05, 0.1, 0.2])
depth = hp("depth", "int", 5, values=[3, 5, 7, 9])

def objective():
    # Cast proxies to concrete types
    model = MyModel(lr=float(lr), depth=int(depth))
    return model.evaluate()  # return score to minimize

opt = HyperOptimizer(
    source_files=[__file__],
    n_iterations=30,
    results_path="results.csv",  # auto-saves after each trial
)
results = opt.run(objective)
print(results.best_score, results.best_params)
```

## Example

```bash
python examples/lgbm_classifier.py
```

Runs 30 HEBO iterations optimizing 5 LightGBM hyperparameters on the breast cancer dataset with 5-fold cross-validation.

## API

| Symbol | Description |
|--------|-------------|
| `hp(name, dtype, default, values=None)` | Declare a hyperparameter proxy |
| `HyperOptimizer(source_files, ..., results_path="results.csv", ...)` | Create optimizer |
| `HyperOptimizer.run(objective)` | Run optimization, returns `TrialResults` |
| `TrialResults.best_score` | Best (minimum) score |
| `TrialResults.best_params` | Parameters of the best trial |
| `TrialResults.from_csv(path)` | Load results from a previous run |
| `TrialResults.to_csv(path)` / `to_json(path)` | Export results |
| `get_registry()` | Return the current proxy registry |
| `clear_registry()` | Clear all registered proxies |

## Crash recovery and resume

Results are auto-saved to `results_path` after every trial. If the process crashes, just re-run the same script — completed trials are reloaded from the CSV and replayed into HEBO, and optimization continues from where it left off:

```python
# First run: completes 12 of 30 trials, then crashes
# Second run: loads 12 trials from results.csv, runs 13–30
opt = HyperOptimizer(
    source_files=[__file__],
    n_iterations=30,
    results_path="results.csv",
)
results = opt.run(objective)
```
