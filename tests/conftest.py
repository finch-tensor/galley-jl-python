import pytest

import numpy as np

# Start Julia here, on the main thread, in every pytest process. Otherwise the
# pytest-xdist controller first imports galley when it rebuilds a
# PerformanceWarning sent by a worker, which starts Julia on execnet's receiver
# thread, and Julia's exit hook then deadlocks at interpreter shutdown.
import galley_jl_python  # noqa: E402, F401


def pytest_addoption(parser):
    parser.addoption(
        "--array-api",
        "--array-api-pytest-args",
        dest="array_api_pytest_args",
        action="append",
        default=[],
        help=(
            "Arguments forwarded to the nested array-api-tests pytest run. "
            "Repeat this option to pass multiple groups."
        ),
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def arr1d():
    return np.arange(100)


@pytest.fixture
def arr2d():
    return np.array(
        [
            [0, 0, 3, 2, 0],
            [1, 0, 0, 1, 0],
            [0, 5, 0, 0, 0],
        ]
    )


@pytest.fixture
def arr3d():
    return np.array(
        [
            [[0, 1, 0, 0], [1, 0, 0, 3]],
            [[4, 0, -1, 0], [2, 2, 0, 0]],
            [[0, 0, 0, 0], [1, 5, 0, 3]],
        ]
    )
