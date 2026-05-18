# Lightweight Hyperopt Framework: Design Pattern

## Declaration

Each file declares its hyperparameters in a preamble block at the top. Each hyper has two lines: the proxy declaration (lowercase) and the materialized value (uppercase):

```python
# HYPERPARAMETERS
learning_rate = hp("learning_rate", dtype=float, default=0.05,
                   values=[0.001, 0.01, 0.05, 0.1])  # grid inline
LEARNING_RATE = float(learning_rate)

max_depth = hp("max_depth", dtype=int, default=5)      # grid from YAML
MAX_DEPTH = int(max_depth)
```

The `hp()` call returns a lazy proxy object that resolves to the current trial's value during optimization, or the default in normal execution. The uppercase cast (`LEARNING_RATE = float(learning_rate)`) materializes the proxy into a real float/int, which is what the rest of the code uses. This avoids any `isinstance` issues with libraries that type-check their inputs.

If `values` is provided inline, that's the grid. If not, the framework looks it up in a shared YAML config file. Each developer can choose per-hyper whether to keep the grid local (self-documenting) or external (cleaner for large grids).

If a hyper is declared but no grid is specified (neither inline nor in YAML), the framework raises an error during optimization. In normal execution, it silently returns the default.

## Namespacing

The same variable name (e.g., `learning_rate`) can appear in different files. To avoid collisions, `hp()` automatically infers a namespace from the calling module using `inspect.stack()`. So `learning_rate` declared in `lgbm.py` becomes `lgbm.learning_rate` internally, and the same name in `transformer.py` becomes `transformer.learning_rate`. No extra parameter needed; the user just writes `hp("learning_rate", ...)` and the framework handles the rest.

This namespace is used consistently in the YAML config, in the optimizer's search space, and in trial result logging.

## YAML config

```yaml
# hyperparams.yaml

lgbm:
  learning_rate:
    type: float
    default: 0.05
    values: [0.001, 0.01, 0.05, 0.1, 0.5]
  max_depth:
    type: int
    default: 5
    values: [3, 5, 7, 10, 15, 20]
  n_estimators:
    type: int
    default: 400
    values: [50, 100, 200, 400, 700, 1000]

transformer:
  learning_rate:
    type: float
    default: 0.0001
    values: [0.00001, 0.0001, 0.001]
  num_heads:
    type: int
    default: 8
    values: [4, 8, 12, 16]
```

## Evaluation: lazy proxy

`hp()` returns a proxy object that resolves the value on access by checking the current trial context (a `contextvars.ContextVar`). The file only needs to be imported once; the proxy re-resolves on each trial.

The uppercase cast convention ensures the rest of the code works with plain Python types:

```python
# In the preamble: proxy
learning_rate = hp("learning_rate", dtype=float, default=0.05)
# Materialized: real float, safe to pass anywhere
LEARNING_RATE = float(learning_rate)

# Rest of the code uses LEARNING_RATE
optimizer = SGD(lr=LEARNING_RATE)
```

## Optimizer design

The optimizer has three responsibilities: collecting the search space, running trials, and reporting results.

### 1. Space collection

At startup, the optimizer scans all registered source files for `hp()` calls via AST parsing (reading the source statically without executing it). It extracts the name, dtype, default, and value grid from each call, namespaced by module. Grids not specified inline are looked up in the YAML config. This collected search space is then converted into a HEBO `DesignSpace`.

### 2. Trial loop

```
for each iteration:
    suggestion = hebo.suggest()          # dict of {qualified_name: value}
    score = run_trial(objective, suggestion)
    hebo.observe(suggestion, score)
```

`run_trial` sets the trial context (a `contextvars.ContextVar` holding the suggestion dict), calls the objective function, then resets the context. The objective function is a user-provided callable that returns a scalar score. The proxy objects resolve to the trial values at the moment they are cast in the preamble.

### 3. Result logging

Each trial logs the full qualified parameter dict and the score. Since names are always namespaced, results are unambiguous even when multiple files share parameter names. Results can be saved as a simple CSV or JSON for post-hoc analysis.

## Appendix: Design decisions

This design emerged from an iterative conversation with Claude as a sounding board. I steered the architecture; the discussion helped me sharpen the tradeoffs. Here's a summary of the choices I considered and why I landed where I did.

**Grid declaration: inline vs. YAML vs. hybrid.**
The starting point was the RAMP pattern (see [ramp-autods lgbm.py](https://github.com/paris-saclay-cds/ramp-autods/blob/master/rampds/workflow_elements/tabular_classifiers/lgbm.py)), where both the variable declaration and the value grid live in the same file. Self-documenting but makes files heavier, and some developers prefer grids in a separate config. I considered four options: (1) decorator-based with external grids, (2) type-hint style with a registry, (3) centralized config object, (4) hybrid where inline grids take precedence and missing grids fall back to YAML. I chose the hybrid because it preserves locality when you want it, while allowing separation when you don't. The cost is two places to look for a grid; in practice this hasn't been a problem at the current scale, but for a much larger codebase I'd reconsider.

**Evaluation: re-exec vs. lazy proxy.**
The core question was how the optimizer injects trial values into `hp()` calls. Re-exec (Option A) re-runs the target file on each trial, so `hp()` picks up values naturally. The lazy proxy (Option B) returns an object that resolves on access by checking a context variable. I considered both seriously, but ruled out re-exec: in an arbitrary codebase, the target file may have side effects at the top level (loading data, initializing models, writing to disk), and re-executing on every trial is unsafe and wasteful. The lazy proxy avoids this entirely since the file is imported once.

**The casting convention.**
The lazy proxy introduces a type-check problem: the proxy isn't a real float/int, so `isinstance` checks in downstream libraries would fail. I considered adding `__class__` overrides to make the proxy look like a real number to `isinstance`, but this gets fragile fast (different libraries check types in different ways). Instead I adopted the RAMP convention of explicit casting: lowercase for the proxy (`learning_rate`), uppercase for the materialized value (`LEARNING_RATE = float(learning_rate)`). The rest of the code uses the uppercase name and never interacts with the proxy directly. It's a one-line cost per hyperparameter, in exchange for robust type behavior everywhere downstream.

**Namespacing.**
When multiple files declare a hyper with the same name (e.g., two different `learning_rate`), there's a collision risk. I initially considered requiring an explicit namespace argument (`hp("learning_rate", namespace="lgbm", ...)`), but rejected it because it adds boilerplate to every call. Instead, `hp()` automatically infers the namespace from the calling module via `inspect.stack()`, so `learning_rate` in `lgbm.py` becomes `lgbm.learning_rate`. The `inspect.stack()` approach has a small magic-cost (one more thing to understand if you debug it) but reads cleaner at the call site. The same namespace structure is used in the YAML config and in the optimizer's result logs.

**Space collection: AST parsing vs. dry-run import.**
The optimizer needs to discover all `hp()` declarations before running any trials. Two options: (1) parse the source files statically using Python's `ast` module, or (2) import the files in a "dry run" with no trial context and have `hp()` register itself as a side effect. I chose AST parsing because dry-run imports execute all top-level code, which can trigger the same unwanted side effects that led me to reject the re-exec evaluation strategy. The two decisions reinforce each other: the framework never executes user code except inside `objective()`.
