"""TrialResults: structured logging of optimization trials."""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["TrialResults"]


class TrialResults:
    """Stores and reports hyperparameter optimization results."""

    def __init__(self) -> None:
        self._trials: list[tuple[int, dict[str, Any], float]] = []

    def add(self, iteration: int, params: dict[str, Any], score: float) -> None:
        self._trials.append((iteration, params, score))

    @classmethod
    def from_csv(cls, path: str) -> TrialResults:
        """Load results from a previously saved CSV file.

        Handles missing files, empty files, and corrupted CSVs gracefully.
        """
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return cls()
        try:
            df = pd.read_csv(path)
            if df.empty:
                return cls()
            obj = cls()
            param_cols = [c for c in df.columns if c not in ("iteration", "score")]
            for _, row in df.iterrows():
                it = int(row["iteration"])
                score = float(row["score"])
                params = {col: row[col] for col in param_cols}
                obj._trials.append((it, params, score))
            return obj
        except Exception as e:
            warnings.warn(
                f"Corrupted results file {path}: {e}. Starting fresh.",
                stacklevel=2,
            )
            return cls()

    @property
    def n_trials(self) -> int:
        return len(self._trials)

    @property
    def best_score(self) -> float:
        if not self._trials:
            raise ValueError("No trials recorded.")
        return min(t[2] for t in self._trials)

    @property
    def best_params(self) -> dict[str, Any]:
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
        """Write results to CSV atomically (write-to-temp + rename)."""
        dir_path = os.path.dirname(os.path.abspath(path))
        with tempfile.NamedTemporaryFile(
            mode="w", dir=dir_path, delete=False, suffix=".tmp"
        ) as f:
            self.all_trials.to_csv(f, index=False)
            temp_path = f.name
        os.replace(temp_path, path)

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
        s = "trial" if n == 1 else "trials"
        return f"TrialResults({n} {s}, best_score={self.best_score:.6f})"
