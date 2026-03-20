"""lazyopt: lightweight Bayesian hyperparameter optimization."""

from .context import trial_scope
from .optimizer import HyperOptimizer
from .proxy import HyperProxy, clear_registry, get_registry, hp
from .results import TrialResults

__all__ = [
    "hp",
    "HyperProxy",
    "HyperOptimizer",
    "TrialResults",
    "trial_scope",
    "get_registry",
    "clear_registry",
]
