"""HyperOptimizer: HEBO-driven Bayesian hyperparameter optimization."""

import numpy as np
import pandas as pd
from hebo.optimizers.hebo import HEBO

from .context import _trial_context
from .results import TrialResults
from .space import build_hebo_space, collect_search_space


class HyperOptimizer:
    """Bayesian hyperparameter optimizer using HEBO.

    Collects the search space from source files (and optional YAML),
    runs a trial loop calling the user's objective function with
    ContextVar-based proxy resolution.

    Results are auto-saved after every trial. If `results_path` points
    to an existing CSV from a previous run, past observations are
    replayed into HEBO and optimization resumes from where it left off.
    """

    def __init__(
        self,
        source_files: list[str],
        yaml_config: str | None = None,
        n_iterations: int = 50,
        results_path: str = "results.csv",
        seed: int = 42,
    ):
        self.source_files = source_files
        self.yaml_config = yaml_config
        self.n_iterations = n_iterations
        self.results_path = results_path
        self.seed = seed

        self.params = collect_search_space(source_files, yaml_config)
        self.space, self.index_to_value = build_hebo_space(self.params)

    def _value_to_index(self, qname: str, value) -> int:
        """Map an actual parameter value back to its grid index."""
        values_list = self.index_to_value[qname]
        for i, v in enumerate(values_list):
            if isinstance(v, float) and isinstance(value, float):
                if abs(v - value) < 1e-15:
                    return i
            elif v == value:
                return i
        return values_list.index(value)

    def _replay_observations(self, opt: HEBO, results: TrialResults) -> None:
        """Feed all past trials into HEBO so it resumes with full history."""
        if results.n_trials == 0:
            return

        indices_rows = []
        scores = []
        for _, params, score in results._trials:
            row = {}
            for p in self.params:
                qname = p["qualified_name"]
                row[qname] = self._value_to_index(qname, params[qname])
            indices_rows.append(row)
            scores.append(score)

        indices_df = pd.DataFrame(indices_rows)
        score_array = np.array(scores).reshape(-1, 1)
        opt.observe(indices_df, score_array)

    def run(self, objective) -> TrialResults:
        """Run the optimization loop.

        If ``results_path`` points to an existing CSV, past trials are
        loaded and replayed into HEBO before continuing. New results
        are appended and the CSV is re-written after every trial.

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

        results = TrialResults.from_csv(self.results_path)
        n_completed = results.n_trials

        if n_completed > 0:
            self._replay_observations(opt, results)
            print(
                f"Resumed from {self.results_path}: {n_completed} past trials loaded."
            )

        if n_completed >= self.n_iterations:
            print(
                f"Already completed {n_completed}/{self.n_iterations} "
                f"iterations. Nothing to do."
            )
            return results

        for i in range(n_completed, self.n_iterations):
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
                indices_dict[qname] = self._value_to_index(qname, values_dict[qname])

            indices_df = pd.DataFrame([indices_dict])
            score_array = np.array([[score]])
            opt.observe(indices_df, score_array)

            results.add(i, values_dict, score)
            results.to_csv(self.results_path)

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
