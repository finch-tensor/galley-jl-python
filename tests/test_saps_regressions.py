"""Regressions found running the Sparse Array Programming Suite (SAPS)."""

import numpy as np
import scipy.sparse as sps
from numpy.testing import assert_equal

import galley_jl_python as galley


def _dense(x):
    return np.asarray(galley.compute(x).todense())


def test_reflected_operators():
    # finite_difference: `1 - u`; ccsd: `2 * V`; simhash: `2 ** arange(n)`.
    x = galley.asarray(np.array([1.0, 2.0, 4.0]))
    assert_equal(_dense(1 - x), [0.0, -1.0, -3.0])
    assert_equal(_dense(2 * x), [2.0, 4.0, 8.0])
    assert_equal(_dense(8 / x), [8.0, 4.0, 2.0])
    assert_equal(_dense(2**x), [2.0, 4.0, 16.0])
    i = galley.asarray(np.array([1, 2, 3]))
    assert_equal(_dense(7 % i), [0, 1, 1])
    assert_equal(_dense(6 & i), [0, 2, 2])


def test_matrix_transpose_attribute():
    a = np.arange(6.0).reshape(2, 3)
    assert_equal(_dense(galley.asarray(a).T), a.T)


def test_logical_not():
    # model_counting: einop's `not` resolves to galley.logical_not.
    x = np.array([0, 1, 2])
    assert_equal(_dense(galley.logical_not(galley.asarray(x))), [True, False, False])
    b = galley.asarray(np.array([True, False]))
    assert_equal(_dense(galley.einop("y[i] = not b[i]", b=b)), [False, True])


def test_logical_operators_on_integers():
    x, y = np.array([0, 1, 2, 3]), np.array([2, 0, 1, 4])
    gx, gy = galley.asarray(x), galley.asarray(y)
    assert_equal(_dense(galley.logical_and(gx, gy)), np.logical_and(x, y))
    assert_equal(_dense(galley.logical_or(gx, gy)), np.logical_or(x, y))
    assert_equal(_dense(galley.logical_xor(gx, gy)), np.logical_xor(x, y))
    assert_equal(_dense(galley.logical_and(True, gx)), np.logical_and(True, x))


def test_asarray_tuple():
    # dae_bdf: `xp.asarray(tspan)` with a tuple.
    assert_equal(_dense(galley.asarray((0.0, 1.0))), [0.0, 1.0])


def test_boolean_sparse_constructors():
    # bfs, pagerank, connected_components: Boolean SciPy matrices failed with
    # "Finch currently only supports isbits defaults" (np.bool_ fill).
    a = sps.random(4, 5, density=0.4, random_state=0, format="csr") > 0.5
    for matrix in (a.tocsr(), a.tocsc(), a.tocoo()):
        assert_equal(_dense(galley.asarray(matrix)), a.toarray())


def test_numpy_constructor_with_array_fill():
    # sparse.COO keeps its fill as a 0-d array.
    a = np.array([[True, False], [False, False]])
    assert_equal(_dense(galley.asarray(a, fill_value=np.asarray(False))), a)


def test_coo_constructor_beyond_two_dimensions():
    coords = (np.array([0, 1]), np.array([2, 0]), np.array([1, 3]))
    data = np.array([5.0, 7.0])
    expected = np.zeros((2, 3, 4))
    expected[coords] = data
    # Coordinates sorted with the first axis outermost, as galley's 2-D COO
    # constructor takes them, in reversed (column-major) order.
    tensor = galley.Tensor.construct_coo(
        coords[::-1], data, expected.shape[::-1], order=galley.Tensor.row_major
    )
    assert_equal(_dense(tensor), expected)


def test_maximum_minimum():
    # gcn: `xp.maximum(x, 0)`; einop's min and max resolve to these too.
    x = np.array([-1.0, 2.0, -3.0])
    gx = galley.asarray(x)
    assert_equal(_dense(galley.maximum(gx, 0.0)), np.maximum(x, 0.0))
    assert_equal(_dense(galley.minimum(0.0, gx)), np.minimum(0.0, x))
    a = galley.asarray(np.array([[3.0, 1.0], [2.0, 5.0]]))
    assert_equal(_dense(galley.einop("y[i] max= a[i, j]", a=a)), [3.0, 5.0])
