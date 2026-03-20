"""Run hyperparameter optimization on the LightGBM classifier."""

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
