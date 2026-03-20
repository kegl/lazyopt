"""HyperOptimizer: HEBO-driven Bayesian hyperparameter optimization."""

import numpy as np
import pandas as pd
from hebo.optimizers.hebo import HEBO

from .context import _trial_context
from .results import TrialResults
from .space import collect_search_space, build_hebo_space


class HyperOptimizer:
    """Bayesian hyperparameter optimizer using HEBO.

    Collects the search space from source files (and optional YAML),
    runs a trial loop calling the user's objective function with
    ContextVar-based proxy resolution.
    """

    def __init__(
        self,
        source_files: list[str],
        yaml_config: str | None = None,
        n_iterations: int = 50,
        seed: int = 42,
    ):
        self.source_files = source_files
        self.yaml_config = yaml_config
        self.n_iterations = n_iterations
        self.seed = seed

        self.params = collect_search_space(source_files, yaml_config)
        self.space, self.index_to_value = build_hebo_space(self.params)

    def run(self, objective) -> TrialResults:
        """Run the optimization loop.

        Parameters
        ----------
        objective : callable
            A no-argument function that returns a scalar score to minimize.
            Inside objective(), hp() proxies resolve to trial values
            via ContextVar.

        Returns
        -------
        TrialResults
        """
        np.random.seed(self.seed)
        opt = HEBO(self.space, scramble_seed=self.seed)
        results = TrialResults()

        for i in range(self.n_iterations):
            suggestion = opt.suggest(n_suggestions=1)

            values_dict = {}
            for p in self.params:
                qname = p["qualified_name"]
                idx = int(suggestion[qname].iloc[0])
                idx = np.clip(idx, 0, len(self.index_to_value[qname]) - 1)
                values_dict[qname] = self.index_to_value[qname][idx]

            score = self._run_trial(objective, values_dict)

            indices_dict = {}
            for p in self.params:
                qname = p["qualified_name"]
                val = values_dict[qname]
                idx = self.index_to_value[qname].index(val)
                indices_dict[qname] = idx

            indices_df = pd.DataFrame([indices_dict])
            score_array = np.array([[score]])
            opt.observe(indices_df, score_array)

            results.add(i, values_dict, score)

            short_params = {
                p["name"]: values_dict[p["qualified_name"]] for p in self.params
            }
            print(
                f"[{i + 1}/{self.n_iterations}] "
                f"score={score:.6f}  params={short_params}"
            )

        print(f"\nBest score: {results.best_score:.6f}")
        print(f"Best params: {results.best_params}")
        return results

    @staticmethod
    def _run_trial(objective, values_dict: dict):
        """Execute objective with the trial context active."""
        token = _trial_context.set(values_dict)
        try:
            score = objective()
        finally:
            _trial_context.reset(token)
        return score
