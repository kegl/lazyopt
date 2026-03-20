"""Tests for lazyopt.results."""

import json

import pytest

from lazyopt.results import TrialResults


class TestTrialResults:
    def test_add_and_best(self):
        r = TrialResults()
        r.add(0, {"lr": 0.1}, 0.5)
        r.add(1, {"lr": 0.01}, 0.3)
        r.add(2, {"lr": 1.0}, 0.7)
        assert r.best_score == 0.3
        assert r.best_params == {"lr": 0.01}

    def test_n_trials(self):
        r = TrialResults()
        assert r.n_trials == 0
        r.add(0, {"a": 1}, 0.5)
        assert r.n_trials == 1

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


class TestFromCsv:
    def test_round_trip(self, tmp_path):
        r = TrialResults()
        r.add(0, {"m.lr": 0.1, "m.depth": 3}, 0.5)
        r.add(1, {"m.lr": 0.01, "m.depth": 5}, 0.3)
        path = str(tmp_path / "results.csv")
        r.to_csv(path)

        loaded = TrialResults.from_csv(path)
        assert loaded.n_trials == 2
        assert loaded.best_score == 0.3
        assert loaded.best_params["m.lr"] == 0.01

    def test_missing_file_returns_empty(self, tmp_path):
        loaded = TrialResults.from_csv(str(tmp_path / "nonexistent.csv"))
        assert loaded.n_trials == 0

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        loaded = TrialResults.from_csv(str(path))
        assert loaded.n_trials == 0

    def test_preserves_dtypes(self, tmp_path):
        r = TrialResults()
        r.add(0, {"m.lr": 0.01, "m.depth": 5}, 0.42)
        path = str(tmp_path / "typed.csv")
        r.to_csv(path)

        loaded = TrialResults.from_csv(path)
        params = loaded.best_params
        assert isinstance(params["m.lr"], float)
