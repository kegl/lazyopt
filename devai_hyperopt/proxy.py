"""HyperProxy class and hp() registration function."""

import inspect
from pathlib import Path

from .context import _trial_context

_registry: dict[str, "HyperProxy"] = {}


class HyperProxy:
    """Lazy proxy for a hyperparameter.

    Resolves to the trial value when a trial context is active,
    otherwise resolves to the default value. Supports transparent
    casting via __float__, __int__, __str__, etc.
    """

    def __init__(self, name: str, namespace: str, dtype: str,
                 default, values: list | None = None):
        self.name = name
        self.namespace = namespace
        self.qualified_name = f"{namespace}.{name}"
        self.dtype = dtype
        self.default = default
        self.values = values

    def _resolve(self):
        ctx = _trial_context.get()
        if ctx is not None and self.qualified_name in ctx:
            return ctx[self.qualified_name]
        return self.default

    def __float__(self):
        return float(self._resolve())

    def __int__(self):
        return int(self._resolve())

    def __str__(self):
        return str(self._resolve())

    def __bool__(self):
        return bool(self._resolve())

    def __repr__(self):
        return (
            f"HyperProxy({self.qualified_name!r}, "
            f"resolved={self._resolve()!r})"
        )

    def __eq__(self, other):
        if isinstance(other, HyperProxy):
            return self._resolve() == other._resolve()
        return self._resolve() == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, HyperProxy):
            return self._resolve() < other._resolve()
        return self._resolve() < other

    def __le__(self, other):
        if isinstance(other, HyperProxy):
            return self._resolve() <= other._resolve()
        return self._resolve() <= other

    def __gt__(self, other):
        if isinstance(other, HyperProxy):
            return self._resolve() > other._resolve()
        return self._resolve() > other

    def __ge__(self, other):
        if isinstance(other, HyperProxy):
            return self._resolve() >= other._resolve()
        return self._resolve() >= other

    def __hash__(self):
        return hash(self.qualified_name)

    def __index__(self):
        return int(self._resolve())


def hp(name: str, dtype: str, default, values: list | None = None) -> HyperProxy:
    """Declare a hyperparameter and register it.

    Namespace is auto-inferred from the caller's filename stem.
    """
    frame = inspect.stack()[1]
    namespace = Path(frame.filename).stem

    proxy = HyperProxy(name, namespace, dtype, default, values)
    _registry[proxy.qualified_name] = proxy
    return proxy


def get_registry() -> dict[str, "HyperProxy"]:
    """Return the global proxy registry."""
    return dict(_registry)


def clear_registry() -> None:
    """Clear the global proxy registry."""
    _registry.clear()
