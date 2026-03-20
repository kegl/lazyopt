"""Tests for devai_hyperopt.results."""

import json
import pytest

from devai_hyperopt.results import TrialResults


class TestTrialResults:
    def test_add_and_best(self):
        r = TrialResults()
        r.add(0, {"lr": 0.1}, 0.5)
        r.add(1, {"lr": 0.01}, 0.3)
        r.add(2, {"lr": 1.0}, 0.7)
        assert r.best_score == 0.3
        assert r.best_params == {"lr": 0.01}

    def test_empty_raises(self):
        r = TrialResults()
        with pytest.raises(ValueError):
            _ = r.best_score
        with pytest.raises(ValueError):
            _ = r.best_params

    def test_all_trials_dataframe(self):
        r = TrialResults()
        r.add(0, {"a": 1}, 0.5)
        r.add(1, {"a": 2}, 0.3)
        df = r.all_trials
        assert len(df) == 2
        assert list(df.columns) == ["iteration", "score", "a"]

    def test_to_csv(self, tmp_path):
        r = TrialResults()
        r.add(0, {"x": 1}, 0.5)
        path = str(tmp_path / "out.csv")
        r.to_csv(path)
        with open(path) as f:
            content = f.read()
        assert "iteration" in content
        assert "0.5" in content

    def test_to_json(self, tmp_path):
        r = TrialResults()
        r.add(0, {"x": 1}, 0.5)
        path = str(tmp_path / "out.json")
        r.to_json(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["score"] == 0.5

    def test_repr(self):
        r = TrialResults()
        assert "0 trials" in repr(r)
        r.add(0, {}, 0.5)
        assert "1 trials" in repr(r)
        assert "0.5" in repr(r)
