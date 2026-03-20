"""Tests for lazyopt.proxy."""

import warnings

import pytest

from lazyopt.context import _trial_context
from lazyopt.proxy import HyperProxy, clear_registry, get_registry, hp


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

    def test_float_cast_error_message(self):
        p = HyperProxy("mode", "model", "str", "auto", ["auto", "manual"])
        with pytest.raises(TypeError, match="model.mode"):
            float(p)

    def test_int_cast_error_message(self):
        p = HyperProxy("mode", "model", "str", "auto", ["auto", "manual"])
        with pytest.raises(TypeError, match="model.mode"):
            int(p)

    def test_comparisons(self):
        p = HyperProxy("lr", "model", "float", 0.1, [0.01, 0.1, 1.0])
        assert p == 0.1
        assert p != 0.5
        assert p < 1.0
        assert p <= 0.1
        assert p > 0.01
        assert p >= 0.1

    def test_comparison_between_proxies(self):
        p1 = HyperProxy("a", "m", "float", 0.5)
        p2 = HyperProxy("b", "m", "float", 0.3)
        assert p1 > p2
        assert p2 < p1
        assert p1 != p2

    def test_repr(self):
        p = HyperProxy("lr", "model", "float", 0.1)
        r = repr(p)
        assert "model.lr" in r
        assert "0.1" in r

    def test_hash_stable(self):
        p = HyperProxy("lr", "model", "float", 0.1)
        assert hash(p) == hash("model.lr")

    def test_index_dunder_with_int(self):
        p = HyperProxy("idx", "model", "int", 3, [1, 2, 3])
        lst = [10, 20, 30, 40]
        assert lst[p] == 40

    def test_index_dunder_rejects_float(self):
        p = HyperProxy("ratio", "model", "float", 0.5)
        with pytest.raises(TypeError, match="model.ratio"):
            [1, 2, 3][p]

    def test_arithmetic_add(self):
        p = HyperProxy("x", "m", "float", 0.5)
        assert p + 1.0 == 1.5
        assert 1.0 + p == 1.5

    def test_arithmetic_sub(self):
        p = HyperProxy("x", "m", "float", 0.5)
        assert p - 0.1 == pytest.approx(0.4)
        assert 1.0 - p == 0.5

    def test_arithmetic_mul(self):
        p = HyperProxy("x", "m", "float", 0.1)
        assert p * 10 == pytest.approx(1.0)
        assert 10 * p == pytest.approx(1.0)

    def test_arithmetic_div(self):
        p = HyperProxy("x", "m", "float", 1.0)
        assert p / 2 == 0.5
        assert 2 / p == 2.0

    def test_neg_and_abs(self):
        p = HyperProxy("x", "m", "float", -3.0)
        assert -p == 3.0
        assert abs(p) == 3.0


class TestHpFunction:
    def test_hp_infers_namespace(self):
        proxy = hp("test_param", "float", 0.5)
        assert proxy.namespace == "test_proxy"
        assert "test_proxy.test_param" in get_registry()

    def test_hp_explicit_namespace(self):
        proxy = hp("lr", "float", 0.1, namespace="custom_ns")
        assert proxy.qualified_name == "custom_ns.lr"
        assert "custom_ns.lr" in get_registry()

    def test_hp_warns_on_duplicate(self):
        hp("dup", "float", 0.1, namespace="ns")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hp("dup", "float", 0.2, namespace="ns")
            assert len(w) == 1
            assert "already registered" in str(w[0].message)


class TestRegistry:
    def test_get_returns_copy(self):
        hp("x", "float", 1.0, namespace="r")
        reg = get_registry()
        reg["extra"] = None
        assert "extra" not in get_registry()

    def test_clear(self):
        hp("y", "float", 1.0, namespace="r")
        clear_registry()
        assert len(get_registry()) == 0
