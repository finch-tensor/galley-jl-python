"""
A JIT that inserts `defer` and `compute` calls into Python functions so that
Galley fuses the tensor operations between control flow.

Adapted from `finch.finch_fused` in finch-tensor (MIT licensed).
"""

from .jit import jit

__all__ = ["jit"]
