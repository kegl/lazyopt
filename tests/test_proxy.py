"""Tests for devai_hyperopt.proxy."""

import pytest
from devai_hyperopt.proxy import HyperProxy, clear_registry, get_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


class TestHyperProxy:
    def test_resolve_default(self):
        p = HyperProxy("lr", "model", "float", 0.1, [0.01, 0.1, 1.0])
        assert float(p) == 0.1

    def test_resolve_trial_value(self):
        from devai_hyperopt.context import _trial_context

        p = HyperProxy("lr", "model", "float", 0.1, [0.01, 0.1, 1.0])
        token = _trial_context.set({"model.lr": 0.01})
        try:
            assert float(p) == 0.01
        finally:
            _trial_context.reset(token)
        assert float(p) == 0.1

    def test_int_cast(self):
        p = HyperProxy("depth", "model", "int", 5, [3, 5, 7])
        assert int(p) == 5

    def test_str_cast(self):
        p = HyperProxy("act", "model", "str", "relu", ["relu", "tanh"])
        assert str(p) == "relu"

    def test_bool_cast(self):
        p = HyperProxy("flag", "model", "bool", True, [True, False])
        assert bool(p) is True

    def test_comparisons(self):
        p = HyperProxy("lr", "model", "float", 0.1, [0.01, 0.1, 1.0])
        assert p == 0.1
        assert p != 0.5
        assert p < 1.0
        assert p <= 0.1
        assert p > 0.01
        assert p >= 0.1

    def test_repr(self):
        p = HyperProxy("lr", "model", "float", 0.1)
        r = repr(p)
        assert "model.lr" in r
        assert "0.1" in r

    def test_hash_stable(self):
        p = HyperProxy("lr", "model", "float", 0.1)
        assert hash(p) == hash("model.lr")

    def test_index_dunder(self):
        p = HyperProxy("idx", "model", "int", 3, [1, 2, 3])
        lst = [10, 20, 30, 40]
        assert lst[p] == 40


class TestRegistry:
    def test_hp_registers(self):
        from devai_hyperopt.proxy import _registry

        p = HyperProxy("lr", "test_ns", "float", 0.1)
        _registry[p.qualified_name] = p
        reg = get_registry()
        assert "test_ns.lr" in reg
        assert reg["test_ns.lr"] is p

    def test_clear(self):
        from devai_hyperopt.proxy import _registry

        _registry["x.y"] = HyperProxy("y", "x", "float", 1.0)
        clear_registry()
        assert len(get_registry()) == 0
