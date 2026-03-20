"""LightGBM classifier model with hyperparameter declarations."""

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
