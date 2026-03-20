"""Tests for lazyopt.context."""

from lazyopt.context import _trial_context, trial_scope


def test_default_is_none():
    assert _trial_context.get() is None


def test_set_and_reset():
    token = _trial_context.set({"a": 1})
    assert _trial_context.get() == {"a": 1}
    _trial_context.reset(token)
    assert _trial_context.get() is None


def test_trial_scope_context_manager():
    with trial_scope({"x": 42}):
        assert _trial_context.get() == {"x": 42}
    assert _trial_context.get() is None


def test_trial_scope_resets_on_exception():
    try:
        with trial_scope({"x": 1}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert _trial_context.get() is None
