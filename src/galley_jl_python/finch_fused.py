from collections.abc import Callable
from typing import Any

from finch.finch_fused import jit as _finch_jit

from .compiled import compute, lazy
from .tensor import Tensor


def _defer(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_defer(item) for item in value)
    return lazy(value) if isinstance(value, Tensor) else value


def _compute(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_compute(item) for item in value)
    return compute(value) if isinstance(value, Tensor) else value


def _maybedefer(values: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_defer(value) for value in values)


def jit(f: Callable[..., Any], /, ctx: Any = None) -> Callable[..., Any]:
    transformed = _finch_jit(f, ctx=ctx)
    transformed.__globals__.update(
        defer=_defer,
        compute=_compute,
        maybedefer=_maybedefer,
    )
    return transformed
