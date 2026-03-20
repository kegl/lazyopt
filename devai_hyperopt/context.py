"""ContextVar-based trial context for hyperparameter resolution."""

from contextvars import ContextVar

_trial_context: ContextVar[dict | None] = ContextVar("_trial_context", default=None)
