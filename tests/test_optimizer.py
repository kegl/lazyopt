"""Tests for lazyopt.optimizer."""

import pytest

from lazyopt.context import _trial_context
from lazyopt.optimizer import HyperOptimizer
from lazyopt.results import TrialResults


@pytest.fixture
def quadratic_source(tmp_path):
    code = """
from lazyopt import hp

x = hp("x", "float", 0.0, values=[-2.0, -1.0, 0.0, 1.0, 2.0])
"""
    p = tmp_path / "quad.py"
    p.write_text(code)
    return str(p)


def _quad_objective():
    ctx = _trial_context.get()
    x = ctx["quad.x"]
    return (x - 1.0) ** 2


class TestHyperOptimizer:
    def test_run_minimizes(self, quadratic_source, tmp_path):
        opt = HyperOptimizer(
            source_files=[quadratic_source],
            n_iterations=10,
            results_path=str(tmp_path / "results.csv"),
            seed=42,
        )
        results = opt.run(_quad_objective)
        assert results.best_score <= 1.0
        assert len(results.all_trials) == 10

    def test_auto_saves_after_each_trial(self, quadratic_source, tmp_path):
        results_path = str(tmp_path / "auto.csv")
        opt = HyperOptimizer(
            source_files=[quadratic_source],
            n_iterations=3,
            results_path=results_path,
            seed=42,
        )
        opt.run(_quad_objective)

        loaded = TrialResults.from_csv(results_path)
        assert loaded.n_trials == 3

    def test_resume_continues_from_checkpoint(self, quadratic_source, tmp_path):
        results_path = str(tmp_path / "resume.csv")

        opt1 = HyperOptimizer(
            source_files=[quadratic_source],
            n_iterations=5,
            results_path=results_path,
            seed=42,
        )
        opt1.run(_quad_objective)

        loaded_after_first = TrialResults.from_csv(results_path)
        assert loaded_after_first.n_trials == 5

        opt2 = HyperOptimizer(
            source_files=[quadratic_source],
            n_iterations=10,
            results_path=results_path,
            seed=42,
        )
        results = opt2.run(_quad_objective)
        assert results.n_trials == 10

        loaded_final = TrialResults.from_csv(results_path)
        assert loaded_final.n_trials == 10

    def test_resume_skips_if_already_done(self, quadratic_source, tmp_path):
        results_path = str(tmp_path / "done.csv")
        opt = HyperOptimizer(
            source_files=[quadratic_source],
            n_iterations=3,
            results_path=results_path,
            seed=42,
        )
        opt.run(_quad_objective)

        opt2 = HyperOptimizer(
            source_files=[quadratic_source],
            n_iterations=3,
            results_path=results_path,
            seed=42,
        )
        results = opt2.run(_quad_objective)
        assert results.n_trials == 3

    def test_context_reset_after_trial(self, quadratic_source, tmp_path):
        def objective():
            return 0.0

        opt = HyperOptimizer(
            source_files=[quadratic_source],
            n_iterations=2,
            results_path=str(tmp_path / "ctx.csv"),
            seed=0,
        )
        opt.run(objective)
        assert _trial_context.get() is None
