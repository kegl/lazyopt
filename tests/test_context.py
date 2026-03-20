"""Tests for devai_hyperopt.context."""

from devai_hyperopt.context import _trial_context


def test_default_is_none():
    assert _trial_context.get() is None


def test_set_and_reset():
    token = _trial_context.set({"a": 1})
    assert _trial_context.get() == {"a": 1}
    _trial_context.reset(token)
    assert _trial_context.get() is None
