"""ContextVar-based trial context for hyperparameter resolution."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

__all__ = ["trial_scope"]

_trial_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "_trial_context", default=None
)


@contextmanager
def trial_scope(params: dict[str, Any]):
    """Context manager that activates a trial parameter set.

    Usage::

        with trial_scope({"model.lr": 0.01}):
            score = objective()
    """
    token = _trial_context.set(params)
    try:
        yield
    finally:
        _trial_context.reset(token)
