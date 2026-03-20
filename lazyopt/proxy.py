"""HyperProxy class and hp() registration function."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any

from .context import _trial_context

__all__ = ["HyperProxy", "hp", "get_registry", "clear_registry"]

_registry: dict[str, HyperProxy] = {}


class HyperProxy:
    """Lazy proxy for a hyperparameter.

    Resolves to the trial value when a trial context is active,
    otherwise resolves to the default value. Supports transparent
    casting via __float__, __int__, __str__, and arithmetic operators.

    Note: ``__hash__`` is based on ``qualified_name`` (identity), while
    ``__eq__`` compares resolved values. This means HyperProxy objects
    should not be used as dict keys or set members when trial context
    may change between insertions and lookups.
    """

    __slots__ = ("name", "namespace", "qualified_name", "dtype", "default", "values")

    def __init__(
        self,
        name: str,
        namespace: str,
        dtype: str,
        default: Any,
        values: list | None = None,
    ) -> None:
        self.name = name
        self.namespace = namespace
        self.qualified_name = f"{namespace}.{name}"
        self.dtype = dtype
        self.default = default
        self.values = values

    def _resolve(self) -> Any:
        ctx = _trial_context.get()
        if ctx is not None and self.qualified_name in ctx:
            return ctx[self.qualified_name]
        return self.default

    # --- type coercion ---

    def __float__(self) -> float:
        val = self._resolve()
        try:
            return float(val)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Cannot convert {self.qualified_name}={val!r} to float"
            ) from e

    def __int__(self) -> int:
        val = self._resolve()
        try:
            return int(val)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Cannot convert {self.qualified_name}={val!r} to int"
            ) from e

    def __str__(self) -> str:
        return str(self._resolve())

    def __bool__(self) -> bool:
        return bool(self._resolve())

    def __repr__(self) -> str:
        return f"HyperProxy({self.qualified_name!r}, resolved={self._resolve()!r})"

    def __index__(self) -> int:
        val = self._resolve()
        if isinstance(val, int):
            return val
        raise TypeError(f"Cannot use {self.qualified_name}={val!r} as an index")

    # --- comparison ---

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HyperProxy):
            return self._resolve() == other._resolve()
        return self._resolve() == other

    def __lt__(self, other: object) -> bool:
        if isinstance(other, HyperProxy):
            return self._resolve() < other._resolve()
        return self._resolve() < other

    def __le__(self, other: object) -> bool:
        if isinstance(other, HyperProxy):
            return self._resolve() <= other._resolve()
        return self._resolve() <= other

    def __gt__(self, other: object) -> bool:
        if isinstance(other, HyperProxy):
            return self._resolve() > other._resolve()
        return self._resolve() > other

    def __ge__(self, other: object) -> bool:
        if isinstance(other, HyperProxy):
            return self._resolve() >= other._resolve()
        return self._resolve() >= other

    def __hash__(self) -> int:
        return hash(self.qualified_name)

    # --- arithmetic ---

    def __add__(self, other: Any) -> Any:
        return self._resolve() + (
            other._resolve() if isinstance(other, HyperProxy) else other
        )

    def __radd__(self, other: Any) -> Any:
        return other + self._resolve()

    def __sub__(self, other: Any) -> Any:
        return self._resolve() - (
            other._resolve() if isinstance(other, HyperProxy) else other
        )

    def __rsub__(self, other: Any) -> Any:
        return other - self._resolve()

    def __mul__(self, other: Any) -> Any:
        return self._resolve() * (
            other._resolve() if isinstance(other, HyperProxy) else other
        )

    def __rmul__(self, other: Any) -> Any:
        return other * self._resolve()

    def __truediv__(self, other: Any) -> Any:
        return self._resolve() / (
            other._resolve() if isinstance(other, HyperProxy) else other
        )

    def __rtruediv__(self, other: Any) -> Any:
        return other / self._resolve()

    def __neg__(self) -> Any:
        return -self._resolve()

    def __abs__(self) -> Any:
        return abs(self._resolve())


def hp(
    name: str,
    dtype: str,
    default: Any,
    values: list | None = None,
    *,
    namespace: str | None = None,
) -> HyperProxy:
    """Declare a hyperparameter and register it.

    Parameters
    ----------
    name : str
        The parameter name.
    dtype : str
        Type descriptor (``"float"``, ``"int"``, ``"str"``).
    default : Any
        Default value used outside of a trial context.
    values : list, optional
        Grid of candidate values. Can also be supplied via YAML config.
    namespace : str, optional
        Override auto-inferred namespace (defaults to caller's filename stem).
    """
    if namespace is None:
        frame = inspect.stack()[1]
        namespace = Path(frame.filename).stem

    proxy = HyperProxy(name, namespace, dtype, default, values)

    if proxy.qualified_name in _registry:
        warnings.warn(
            f"Hyperparameter {proxy.qualified_name!r} is already registered "
            f"and will be overwritten.",
            stacklevel=2,
        )
    _registry[proxy.qualified_name] = proxy
    return proxy


def get_registry() -> dict[str, HyperProxy]:
    """Return a copy of the global proxy registry."""
    return dict(_registry)


def clear_registry() -> None:
    """Clear the global proxy registry."""
    _registry.clear()
