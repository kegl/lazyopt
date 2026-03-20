"""Tests for devai_hyperopt.optimizer."""

import pytest

from devai_hyperopt.optimizer import HyperOptimizer
from devai_hyperopt.context import _trial_context


@pytest.fixture
def quadratic_source(tmp_path):
    code = """
from devai_hyperopt import hp

x = hp("x", "float", 0.0, values=[-2.0, -1.0, 0.0, 1.0, 2.0])
"""
    p = tmp_path / "quad.py"
    p.write_text(code)
    return str(p)


class TestHyperOptimizer:
    def test_run_minimizes(self, quadratic_source):
        def objective():
            ctx = _trial_context.get()
            x = ctx["quad.x"]
            return (x - 1.0) ** 2

        opt = HyperOptimizer(
            source_files=[quadratic_source],
            n_iterations=10,
            seed=42,
        )
        results = opt.run(objective)
        assert results.best_score <= 1.0
        assert len(results.all_trials) == 10

    def test_context_reset_after_trial(self, quadratic_source):
        def objective():
            return 0.0

        opt = HyperOptimizer(
            source_files=[quadratic_source],
            n_iterations=2,
            seed=0,
        )
        opt.run(objective)
        assert _trial_context.get() is None
