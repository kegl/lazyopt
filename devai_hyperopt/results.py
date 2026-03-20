"""TrialResults: structured logging of optimization trials."""

import json
import pandas as pd


class TrialResults:
    """Stores and reports hyperparameter optimization results."""

    def __init__(self):
        self._trials: list[tuple[int, dict, float]] = []

    def add(self, iteration: int, params: dict, score: float) -> None:
        self._trials.append((iteration, params, score))

    @property
    def best_score(self) -> float:
        if not self._trials:
            raise ValueError("No trials recorded.")
        return min(t[2] for t in self._trials)

    @property
    def best_params(self) -> dict:
        if not self._trials:
            raise ValueError("No trials recorded.")
        best = min(self._trials, key=lambda t: t[2])
        return best[1]

    @property
    def all_trials(self) -> pd.DataFrame:
        rows = []
        for it, params, score in self._trials:
            row = {"iteration": it, "score": score, **params}
            rows.append(row)
        return pd.DataFrame(rows)

    def to_csv(self, path: str) -> None:
        self.all_trials.to_csv(path, index=False)

    def to_json(self, path: str) -> None:
        records = []
        for it, params, score in self._trials:
            records.append(
                {
                    "iteration": it,
                    "params": params,
                    "score": score,
                }
            )
        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=str)

    def __repr__(self) -> str:
        n = len(self._trials)
        if n == 0:
            return "TrialResults(0 trials)"
        return f"TrialResults({n} trials, best_score={self.best_score:.6f})"
