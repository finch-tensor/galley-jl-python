from collections.abc import Callable
from typing import Any

from .dataflow import insert_lazy_and_compute
from .parser import fused_function_to_python_function, parse_fused_function


def jit(f: Callable[..., Any], /, ctx: Any = None) -> Callable[..., Any]:
    """
    A decorator that compiles a function to defer and compute Galley tensors, so
    that tensor operations between control flow are fused into one computation.

    Parameters:
    - f: The function to compile. It can use basic python control flow and
        operations (e.g. while, for, if), but not generators, classes or recursion.
    - ctx: Unused; accepted for compatibility.

    Example usage:
    @jit
    def my_function(A, B, C):
        D = A @ B
        while some_condition(D):
            D = D + C
        return D

    is transformed into

    def my_function(A, B, C):
        A, B = maybedefer((A, B))
        D = A @ B
        D, = compute((D,))
        while some_condition(D):
            C, D = maybedefer((C, D))
            D = D + C
            D, = compute((D,))
        D, = compute((D,))
        return D
    """
    fused_fn = parse_fused_function(f)
    transformed_fn = insert_lazy_and_compute(fused_fn)
    return fused_function_to_python_function(transformed_fn)
