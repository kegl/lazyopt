"""LightGBM classifier hyperparameter optimization on breast cancer dataset."""

from devai_hyperopt import hp, HyperOptimizer

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
import numpy as np

learning_rate = hp(
    "learning_rate", "float", 0.1,
    values=[0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
)
max_depth = hp(
    "max_depth", "int", 5,
    values=[3, 4, 5, 6, 7, 8, 10],
)
n_estimators = hp(
    "n_estimators", "int", 100,
    values=[50, 100, 200, 300, 500],
)
num_leaves = hp(
    "num_leaves", "int", 31,
    values=[15, 20, 31, 40, 50, 63],
)
min_child_samples = hp(
    "min_child_samples", "int", 20,
    values=[5, 10, 20, 30, 50],
)


def objective():
    X, y = load_breast_cancer(return_X_y=True)

    clf = LGBMClassifier(
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        verbose=-1,
    )
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return -np.mean(scores)


if __name__ == "__main__":
    opt = HyperOptimizer(
        source_files=[__file__],
        n_iterations=30,
        results_path="lgbm_results.csv",
        seed=42,
    )
    results = opt.run(objective)
    print(f"\n{results}")
