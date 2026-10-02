"""
Term, rewriting and dataflow utilities used by the fused JIT.

Adapted from `finch.symbolic` in finch-tensor (MIT licensed), keeping only what
the fused JIT needs.
"""

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from inspect import isbuiltin, isclass, isfunction
from typing import Any, Generic, Self, TypeVar


class Term:
    @abstractmethod
    def head(self) -> Callable[..., Self]:
        """Return the head type of the S-expression."""
        ...

    @classmethod
    @abstractmethod
    def make_term(cls, head: Callable[..., Self], *children: "Term") -> Self:
        """
        Construct a new term in the same family of terms with the given head type and
        children. This function should satisfy
        `x == x.make_term(x.head(), *x.children)`
        """
        ...


@dataclass(frozen=True, eq=True)
class TermTree(Term, ABC):
    @property
    @abstractmethod
    def children(self) -> list[Term]:
        """Return the children (AKA tail) of the S-expression."""
        ...


class NamedTerm(Term, ABC):
    """
    A named program node, e.g. a variable, function, or index.
    """

    @property
    @abstractmethod
    def symbol(self) -> str: ...


def qual_repr(val: Any) -> str:
    if isbuiltin(val) or isclass(val) or isfunction(val):
        return f"{val.__module__}.{val.__qualname__}"
    return repr(val)


def literal_repr(name: str, fields: dict[str, Any]) -> str:
    return (
        name + "(" + ", ".join([f"{k}={qual_repr(v)}" for k, v in fields.items()]) + ")"
    )


def PostOrderDFS(node: Term) -> Iterator[Term]:
    if isinstance(node, TermTree):
        for arg in node.children:
            yield from PostOrderDFS(arg)
    yield node


T = TypeVar("T", bound=Term)

RwCallable = Callable[[T], T | None]


def default_rewrite(x: T | None, y: T) -> T:
    return x if x is not None else y


class Rewrite(Generic[T]):
    """
    A rewriter which returns the original argument even if `rw` returns nothing.
    """

    def __init__(self, rw: RwCallable):
        self.rw = rw

    def __call__(self, x: T) -> T:
        return default_rewrite(self.rw(x), x)


class PostWalk:
    """
    A rewriter which recursively rewrites the arguments of each node using
    `rw`, then rewrites the resulting node. If all rewriters return `nothing`,
    returns `nothing`.
    """

    def __init__(self, rw: RwCallable):
        self.rw = rw

    def __call__(self, x: T) -> T | None:
        if isinstance(x, TermTree):
            args = x.children
            new_args = list(map(self, args))
            if all(arg is None for arg in new_args):
                return self.rw(x)
            y = x.make_term(
                x.head(), *map(lambda x1, x2: default_rewrite(x1, x2), new_args, args)
            )
            assert isinstance(y, type(x))
            return default_rewrite(self.rw(y), y)
        return self.rw(x)


class Chain(Generic[T]):
    """
    A rewriter which rewrites using each rewriter in `rws`. If all rewriters
    return `nothing`, return `nothing`.
    """

    def __init__(self, rws: Iterable[RwCallable]):
        self.rws = rws

    def __call__(self, x: T) -> T | None:
        is_success = False
        for rw in self.rws:
            y = rw(x)
            if y is not None:
                is_success = True
                x = y
        if is_success:
            return x
        return None


class Namespace:
    """
    A namespace for managing variable names and aesthetic fresh variable
    generation.  You can construct a namespace from an existing tree of `Term`s
    to avoid name collisions.
    """

    def __init__(self, root=None):
        self.counts = defaultdict(int)
        if root is not None:
            for node in PostOrderDFS(root):
                if isinstance(node, NamedTerm):
                    self.freshen(node.symbol)

    def freshen(self, *tags) -> str:
        """
        Generate a fresh variable name based on the provided tags.
        """
        name = "_".join(str(tag) for tag in tags)
        m = re.match(r"^(.*)_(\d+)$", name)
        if m is None:
            tag = name
            n = 1
        else:
            tag = m.group(1)
            n = int(m.group(2))
        n = max(self.counts[tag] + 1, n)
        self.counts[tag] = n
        if n == 1:
            return tag
        return f"{tag}_{n}"


class Context(ABC):
    def __init__(self, namespace=None, preamble=None, epilogue=None):
        self.namespace = namespace if namespace is not None else Namespace()
        self.preamble = preamble if preamble is not None else []
        self.epilogue = epilogue if epilogue is not None else []

    def exec(self, thunk: Any):
        self.preamble.append(thunk)

    def block(self):
        """
        Create a new block. Preambles and epilogues will stay within this block.
        """
        blk = self.__class__()
        blk.namespace = self.namespace
        blk.preamble = []
        blk.epilogue = []
        return blk

    @abstractmethod
    def emit(self):
        """
        Emit the code in this context.
        """
        ...


class BasicBlock:
    """Linear sequence of statements with a single entry and exit."""

    def __init__(self, id: str) -> None:
        self.id = id
        self.statements: list = []
        self.successors: list[BasicBlock] = []
        self.predecessors: list[BasicBlock] = []

    def add_statement(self, statement) -> None:
        self.statements.append(statement)

    def add_successor(self, successor: "BasicBlock") -> None:
        if successor not in self.successors:
            self.successors.append(successor)

        if self not in successor.predecessors:
            successor.predecessors.append(self)

    def __str__(self) -> str:
        succ_names = [succ.id for succ in self.successors]
        lines = [f"{self.id}: #succs=[{', '.join(succ_names)}]"]
        lines.extend(f"    {stmt}" for stmt in self.statements)
        return "\n".join(lines)


class ControlFlowGraph:
    """Collection of BasicBlocks plus explicit ENTRY and EXIT nodes."""

    def __init__(self) -> None:
        self.block_counter = 0
        self.block_name = ""
        self.blocks: dict[str, BasicBlock] = {}

        self.entry_block = self.new_block_custom("ENTRY")
        self.exit_block = self.new_block_custom("EXIT")

    def new_block(self) -> BasicBlock:
        return self.new_block_custom(self.block_name)

    def new_block_custom(self, name: str) -> BasicBlock:
        bid = f"{name}_{self.block_counter}"
        self.block_counter += 1
        block = BasicBlock(bid)
        self.blocks[bid] = block
        return block

    def __str__(self) -> str:
        return "\n\n".join(str(block) for block in self.blocks.values())


class DataFlowAnalysis(ABC):
    """
    Base class for performing data flow analyses over a ControlFlowGraph with a
    work-list algorithm, in either a "forward" or "backward" direction.

    `input_states` and `output_states` map each basic block id to the lattice
    state at the entry and exit of the block.
    """

    def __init__(self, cfg: ControlFlowGraph):
        self.cfg: ControlFlowGraph = cfg
        self.input_states: dict[str, dict] = {
            block.id: {} for block in cfg.blocks.values()
        }
        self.output_states: dict[str, dict] = {
            block.id: {} for block in cfg.blocks.values()
        }

    @abstractmethod
    def stmt_str(self, stmt, state: dict) -> str: ...

    @abstractmethod
    def transfer(self, stmts, state: dict) -> dict: ...

    @abstractmethod
    def join(self, state_1: dict, state_2: dict) -> dict: ...

    @abstractmethod
    def direction(self) -> str: ...

    def analyze(self) -> None:
        forward = self.direction() == "forward"
        work_list: list[BasicBlock] = list(self.cfg.blocks.values())
        while work_list:
            block = work_list.pop(0)
            neighbors = block.predecessors if forward else block.successors
            if not neighbors:
                input_state = {}
            else:
                neighbor_outputs = [
                    self.output_states.get(neighbor.id, {}) for neighbor in neighbors
                ]
                input_state = neighbor_outputs[0].copy()
                for neighbor_output in neighbor_outputs[1:]:
                    input_state = self.join(input_state, neighbor_output)

            self.input_states[block.id] = input_state
            output_state = self.transfer(block.statements, input_state)
            if output_state != self.output_states.get(block.id, {}):
                self.output_states[block.id] = output_state
                for dependent in block.successors if forward else block.predecessors:
                    if dependent not in work_list:
                        work_list.append(dependent)
