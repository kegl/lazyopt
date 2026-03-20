"""Run hyperparameter optimization on the LightGBM classifier."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

from lazyopt import HyperOptimizer

from lgbm_classifier import get_classifier

X, y = load_breast_cancer(return_X_y=True)


def objective():
    clf = get_classifier()
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return 1 - np.mean(scores)


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
