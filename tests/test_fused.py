import numpy as np
from numpy.testing import assert_allclose

import galley_jl_python as galley


def assert_galley_allclose(actual, expected):
    assert isinstance(actual, galley.Tensor)
    assert_allclose(actual.todense(), expected)


def test_jit_straight_line_uses_galley_tensors():
    @galley.jit
    def opt_fn(A, B):
        return galley.matmul(A, B)

    A = galley.Tensor(np.array([[1, 2], [3, 4]]))
    B = galley.Tensor(np.array([[5, 6], [7, 8]]))

    assert_galley_allclose(opt_fn(A, B), np.matmul(A.todense(), B.todense()))


def test_jit_return_expr_uses_galley_tensors():
    @galley.jit
    def opt_fn(A, B):
        return galley.matmul(A, B), galley.add(A, B)

    A = galley.Tensor(np.array([[1, 2], [3, 4]]))
    B = galley.Tensor(np.array([[5, 6], [7, 8]]))

    actual_matmul, actual_add = opt_fn(A, B)
    assert_galley_allclose(actual_matmul, np.matmul(A.todense(), B.todense()))
    assert_galley_allclose(actual_add, A.todense() + B.todense())


def test_jit_scalar_loop_uses_galley_tensors():
    @galley.jit
    def opt_fn(A, n):
        B = A
        for _i in range(n):
            B = galley.add(B, A)
        return B

    A = galley.Tensor(np.array([[1, 0], [0, 1]], dtype=float))

    assert_galley_allclose(opt_fn(A, 3), 4 * A.todense())


def test_jit_if_branch_uses_galley_tensors():
    @galley.jit
    def opt_fn(A, B, use_matmul):
        if use_matmul:
            return galley.matmul(A, B)
        return galley.add(A, B)

    A = galley.Tensor(np.array([[1, 2], [3, 4]]))
    B = galley.Tensor(np.array([[1, 0], [0, 1]]))

    assert_galley_allclose(opt_fn(A, B, True), np.matmul(A.todense(), B.todense()))
    assert_galley_allclose(opt_fn(A, B, False), A.todense() + B.todense())


def test_jit_while_loop_uses_galley_tensors():
    @galley.jit
    def opt_fn(A, B, n):
        C = A
        while n > 0:
            C = galley.add(C, B)
            n = n - 1
        return C

    A = galley.Tensor(np.array([[1, 2], [3, 4]]))
    B = galley.Tensor(np.array([[1, 0], [0, 1]]))

    assert_galley_allclose(opt_fn(A, B, 3), A.todense() + 3 * B.todense())
