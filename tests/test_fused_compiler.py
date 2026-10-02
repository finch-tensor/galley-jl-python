import ast
import operator
import sys
import textwrap

import pytest

import numpy as np

import galley_jl_python as gl
from galley_jl_python import add, matmul
from galley_jl_python.fused import nodes as fzd
from galley_jl_python.fused.cfg_builder import (
    fused_build_cfg,
    fused_desugar,
    number_statements,
)
from galley_jl_python.fused.dataflow import LivenessAnalysis
from galley_jl_python.fused.parser import (
    fused_function_to_python_ast,
    parse_fused_function,
)


def test_jit_does_not_depend_on_finch_tensor():
    @gl.jit
    def opt_fn(A, B):
        return gl.add(A, B)

    A = gl.Tensor(np.array([[1, 2], [3, 4]]))
    opt_fn(A, A)

    assert not any(name == "finch" or name.startswith("finch.") for name in sys.modules)


def test_parse_simple_function_with_control_flow_and_calls():
    def simple_fn(fn, n):
        total = 0
        for i in range(n):
            if i < n:  # noqa: SIM108
                total = fn(total, i, scale=2)
            else:
                total = total - 1
        while total < n:
            total = total + 1
        return total

    result = parse_fused_function(simple_fn)

    expected = fzd.Function(
        fzd.Literal("simple_fn"),
        (fzd.Variable("fn"), fzd.Variable("n")),
        fzd.Block(
            (
                fzd.Assign(fzd.Variable("total"), fzd.Literal(0)),
                fzd.For(
                    fzd.Variable("i"),
                    fzd.Call(fzd.Literal(range), (fzd.Variable("n"),)),
                    fzd.Block(
                        (
                            fzd.If(
                                fzd.Compare(
                                    fzd.Variable("i"),
                                    fzd.Literal(operator.lt),
                                    fzd.Variable("n"),
                                ),
                                fzd.Block(
                                    (
                                        fzd.Assign(
                                            fzd.Variable("total"),
                                            fzd.Call(
                                                fzd.Variable("fn"),
                                                (
                                                    fzd.Variable("total"),
                                                    fzd.Variable("i"),
                                                ),
                                                (
                                                    fzd.Keyword(
                                                        "scale",
                                                        fzd.Literal(2),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )
                                ),
                                fzd.Block(
                                    (
                                        fzd.Assign(
                                            fzd.Variable("total"),
                                            fzd.BinaryOp(
                                                fzd.Variable("total"),
                                                fzd.Literal(operator.sub),
                                                fzd.Literal(1),
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
                fzd.While(
                    fzd.Compare(
                        fzd.Variable("total"),
                        fzd.Literal(operator.lt),
                        fzd.Variable("n"),
                    ),
                    fzd.Block(
                        (
                            fzd.Assign(
                                fzd.Variable("total"),
                                fzd.BinaryOp(
                                    fzd.Variable("total"),
                                    fzd.Literal(operator.add),
                                    fzd.Literal(1),
                                ),
                            ),
                        )
                    ),
                ),
                fzd.Return((fzd.Variable("total"),)),
            )
        ),
    )

    assert result == expected


def test_parse_rejects_local_function_definitions():
    def with_local_fn(x):
        def inner(y):
            return y + 1

        return inner(x)

    with pytest.raises(ValueError, match="Local functions are not supported"):
        parse_fused_function(with_local_fn)


def test_parse_rejects_for_else_blocks():
    def with_for_else(n):
        for i in range(n):
            n = n + i
        else:
            n = n + 1
        return n

    with pytest.raises(ValueError, match="For-else blocks are not supported"):
        parse_fused_function(with_for_else)


def test_parse_rejects_while_else_blocks():
    def with_while_else(n):
        while n < 3:
            n = n + 1
        else:
            n = n + 2
        return n

    with pytest.raises(ValueError, match="While-else blocks are not supported"):
        parse_fused_function(with_while_else)


def test_parse_reverse_parse_is_lossless_on_supported_subset():
    """
    Round-tripping recovers the source, with numeric literals left as literals.
    """

    def roundtrip_fn(n):
        total = 0
        for i in range(n):
            if i < n:  # noqa: SIM108
                total = total + i
            else:
                total = total - 1
        while total < n:
            total = total + 1
        return total

    expected_source = textwrap.dedent("""\
        def roundtrip_fn(n):
            total = 0
            for i in range(n):
                if i < n:
                    total = total + i
                else:
                    total = total - 1
            while total < n:
                total = total + 1
            return total
        """)
    expected_fn = ast.parse(expected_source).body[0]

    fused_fn = parse_fused_function(roundtrip_fn)
    roundtrip_fn_ast = fused_function_to_python_ast(fused_fn)

    assert ast.dump(expected_fn, include_attributes=False) == ast.dump(
        roundtrip_fn_ast,
        include_attributes=False,
    )


def test_cfg_builder():
    def simple_fn(fn, n):
        total = 0
        for i in range(n):
            if i < n:  # noqa: SIM108
                total = fn(total, i)
            else:
                total = total - 1
        while total < n:
            total = total + 1
        return total

    fused_fn = parse_fused_function(simple_fn)
    numbered_fn, _ = number_statements(fused_fn)
    desugared_fn = fused_desugar(numbered_fn)
    cfg = fused_build_cfg(desugared_fn)

    # We won't assert on the exact structure of the CFG here, but we can at least
    # check that it has the expected number of blocks. The exact number of blocks
    # may depend on how the CFG builder handles certain constructs, so this is a
    # somewhat loose check.
    assert (
        len(cfg.blocks) >= 5
    )  # Entry block, for loop block, if block, while block, return block


def _build_liveness(fn):
    """Helper: parse, number, desugar, build CFG, run liveness."""
    fused_fn = parse_fused_function(fn)
    numbered_fn, _ = number_statements(fused_fn)
    desugared_fn = fused_desugar(numbered_fn)
    cfg = fused_build_cfg(desugared_fn)
    liveness = LivenessAnalysis(cfg)
    liveness.analyze()
    return liveness, cfg


def _all_live_names(liveness, cfg):
    """Union of all live variable names across all blocks (in and out)."""
    names = set()
    for block in cfg.blocks.values():
        names |= {v.name for v in liveness.output_states[block.id]}
        names |= {v.name for v in liveness.input_states[block.id]}
    return names


def test_liveness_straight_line():
    """Parameters must appear live at the function entry block."""

    def fn(a, b):
        c = add(a, b)
        return c  # noqa: RET504

    liveness, cfg = _build_liveness(fn)

    # After desugaring, the function body is a single block.
    # live-IN (output_states) of that block = {a, b}, since both are used.
    # c is defined and consumed in the same block so it never crosses a block
    # boundary and does not appear in any block-boundary state.
    all_live_in = set()
    for block in cfg.blocks.values():
        all_live_in |= {v.name for v in liveness.output_states[block.id]}

    assert "a" in all_live_in
    assert "b" in all_live_in
    assert "c" not in all_live_in


def test_liveness_dead_variable():
    """A variable assigned but never used afterwards must not be live after."""

    def fn(a, b):
        unused = add(a, b)  # noqa: F841
        c = matmul(a, b)
        return c  # noqa: RET504

    liveness, cfg = _build_liveness(fn)

    exit_block = list(cfg.blocks.values())[-1]
    live_at_exit = {v.name for v in liveness.input_states[exit_block.id]}
    assert "unused" not in live_at_exit


def test_liveness_loop_carried():
    """Loop-carried variables must be live at the top of the loop body."""

    def fn(n):
        total = 0
        for _i in range(n):
            total = total + 1
        return total

    liveness, cfg = _build_liveness(fn)
    names = _all_live_names(liveness, cfg)

    assert "total" in names
    assert "n" in names


def test_liveness_multi_loop_carried():
    """Multiple loop-carried variables must all be live inside the loop."""

    def fn(A, B, C, n):
        D = matmul(A, B)
        E = add(A, C)
        for _i in range(n):
            D = add(D, E)
        return D

    liveness, cfg = _build_liveness(fn)
    names = _all_live_names(liveness, cfg)

    assert "D" in names
    assert "E" in names
    assert "n" in names


def test_liveness_if_branch_merges():
    """Variables used in either branch must be live before the if."""

    def fn(cond, a, b):
        if cond:  # noqa: SIM108
            result = add(a, b)
        else:
            result = matmul(a, b)
        return result

    liveness, cfg = _build_liveness(fn)
    names = _all_live_names(liveness, cfg)

    assert "a" in names
    assert "b" in names
    assert "result" in names


def _increment(x):
    return x + 1


def _decrement(x):
    return x - 1


_decrement.__name__ = "_increment"


def test_jit_keeps_bitwise_and_logical_operators_apart():
    @gl.jit
    def opt_fn(a, b):
        return a | b, a & b, a or b, a and b

    assert opt_fn(1, 2) == (3, 0, 1, 2)


def test_jit_loop_variable_is_defined_by_the_loop():
    @gl.jit
    def opt_fn(n):
        total = 0
        for i in range(n):
            total = total + i
        return total

    assert opt_fn(4) == 6


def test_jit_break():
    @gl.jit
    def opt_fn(limit):
        i = 0
        while i < 10:
            i = i + 1
            if i > limit:
                break
        return i

    assert opt_fn(3) == 4


def test_jit_expression_statement(capsys):
    @gl.jit
    def opt_fn(x):
        print(x)
        return x

    assert opt_fn(5) == 5
    assert capsys.readouterr().out == "5\n"


def test_jit_conditional_expression():
    @gl.jit
    def opt_fn(c):
        return 1 if c else 2

    assert opt_fn(True) == 1
    assert opt_fn(False) == 2


def test_jit_distinct_callables_with_the_same_name():
    @gl.jit
    def opt_fn(x):
        return _increment(x), _decrement(x)

    assert opt_fn(5) == (6, 4)
