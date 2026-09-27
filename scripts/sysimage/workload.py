"""Representative workload traced to build the Julia sysimage.

Every Julia method compiled while this runs is recorded (via juliacall's
``trace_compile`` option) and baked into the sysimage by ``build.py``. Add
operations here to cover more of the library ahead of time.
"""

import numpy as np

import galley_jl_python as gl


def run_ops(A, B, C, v):
    gl.matmul(A, B)
    gl.sum(gl.matmul(gl.matmul(A, B), C))
    gl.add(A, B)
    gl.multiply(A, B)
    gl.subtract(A, B)
    gl.sum(A, axis=0)
    gl.sum(A, axis=1)
    gl.max(A)
    gl.tensordot(A, v, axes=1)
    gl.permute_dims(A, (1, 0))
    A[0]
    A[0, 0]
    A[1:3, :]
    A.todense()

    @gl.jit
    def sum_abc(A, B, C):
        return gl.sum(gl.matmul(gl.matmul(A, B), C))

    @gl.jit
    def chain(A, B, C):
        return gl.add(gl.matmul(A, B), C)

    sum_abc(A, B, C)
    chain(A, B, C)


def main():
    rng = np.random.default_rng(0)
    for scheduler in (gl.GalleyScheduler(), gl.DefaultScheduler()):
        gl.set_optimizer(scheduler)
        for n in (50, 400):
            A, B, C = (
                gl.random((n, n), density=0.02, random_state=s) for s in (1, 2, 3)
            )
            v = gl.asarray(rng.random(n))
            run_ops(A, B, C, v)
        for fmt in ("coo", "csr", "csc", "dense"):
            for dtype in (np.float64, np.int64):
                dense = (rng.random((60, 60)) < 0.05).astype(dtype)
                X = gl.asarray(dense, format=fmt)
                gl.matmul(X, X)
                gl.sum(X)
                gl.add(X, X)
    gl.set_optimizer(gl.GalleyScheduler())


if __name__ == "__main__":
    main()
