"""
Tree representation for symbolic regression expressions.
Nodes can be binary operators (+, -, *, /) or terminals (constants, x, x**a).
"""
import random
import math
from copy import deepcopy


# -- Node types ----------------------------------------------------------------

BINARY_OPS = ['+', '-', '*', '/']

# Terminal types
TERMINAL_X     = 'x'
TERMINAL_CONST = 'const'
TERMINAL_POW   = 'pow'      # x**a,  a ∈ {2,3,4,5}

MAX_POWER = 5


class Node:
    """A node in the expression tree."""

    def __init__(self, node_type, value=None, left=None, right=None):
        """
        Parameters
        ----------
        node_type : str
            One of BINARY_OPS, TERMINAL_X, TERMINAL_CONST, TERMINAL_POW.
        value : int | float | None
            - For TERMINAL_CONST: the float constant.
            - For TERMINAL_POW:   the integer exponent (≥ 2).
            - For operators / x:  None.
        left, right : Node | None
            Children (only for binary operators).
        """
        self.node_type = node_type
        self.value     = value
        self.left      = left
        self.right     = right

    # -- Evaluation ------------------------------------------------------------

    def evaluate(self, x: float) -> float:
        """Evaluate the expression at a given x.  Returns NaN on errors."""
        try:
            return self._eval(x)
        except Exception:
            return float('nan')

    def _eval(self, x: float) -> float:
        if self.node_type == TERMINAL_X:
            return x
        if self.node_type == TERMINAL_CONST:
            return self.value
        if self.node_type == TERMINAL_POW:
            return x ** self.value

        # Binary operator
        lv = self.left._eval(x)
        rv = self.right._eval(x)

        if self.node_type == '+':
            return lv + rv
        if self.node_type == '-':
            return lv - rv
        if self.node_type == '*':
            return lv * rv
        if self.node_type == '/':
            if abs(rv) < 1e-10:
                raise ZeroDivisionError
            return lv / rv
        raise ValueError(f"Unknown node type: {self.node_type}")

    # -- String conversion (phenotype) -----------------------------------------

    def to_string(self, parent_op: str | None = None) -> str:
        """Convert tree to a Python-expression string."""
        if self.node_type == TERMINAL_X:
            return 'x'
        if self.node_type == TERMINAL_CONST:
            v = self.value
            # Represent as integer when possible
            if v == int(v) and abs(v) < 1e9:
                return str(int(v))
            return f'{v:.6g}'
        if self.node_type == TERMINAL_POW:
            return f'x**{self.value}'

        ls = self.left.to_string(self.node_type)
        rs = self.right.to_string(self.node_type)

        expr = f'{ls} {self.node_type} {rs}'

        # Parenthesise when needed to preserve precedence
        if parent_op in ('*', '/') and self.node_type in ('+', '-'):
            expr = f'({expr})'
        elif self.node_type == '/' and parent_op == '/':
            # Right-associativity issue: a / (b / c)
            expr = f'({expr})'

        return expr

    # -- Size / depth ----------------------------------------------------------

    def size(self) -> int:
        """Total number of nodes."""
        if self.node_type in (TERMINAL_X, TERMINAL_CONST, TERMINAL_POW):
            return 1
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        """Maximum depth (root = 0)."""
        if self.node_type in (TERMINAL_X, TERMINAL_CONST, TERMINAL_POW):
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    # -- Utilities -------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.node_type in (TERMINAL_X, TERMINAL_CONST, TERMINAL_POW)

    def clone(self) -> 'Node':
        return deepcopy(self)

    def __repr__(self):
        return self.to_string()


# -- Random tree generation ----------------------------------------------------

def _random_terminal(const_range: tuple[float, float] = (-10.0, 10.0)) -> Node:
    """Return a random terminal node."""
    choice = random.random()
    if choice < 0.35:
        return Node(TERMINAL_X)
    elif choice < 0.60:
        exp = random.randint(2, MAX_POWER)
        return Node(TERMINAL_POW, value=exp)
    else:
        # Mix of integers (interpretable) and floats
        if random.random() < 0.5:
            v = float(random.randint(int(const_range[0]), int(const_range[1])))
        else:
            v = random.uniform(const_range[0], const_range[1])
        if abs(v) < 1e-6:
            v = 1.0
        return Node(TERMINAL_CONST, value=v)


def grow(max_depth: int, current_depth: int = 0,
         const_range: tuple[float, float] = (-10.0, 10.0)) -> Node:
    """Grow initialisation: nodes can be terminals at any depth."""
    if current_depth >= max_depth:
        return _random_terminal(const_range)

    if current_depth > 0 and random.random() < 0.4:
        return _random_terminal(const_range)

    op = random.choice(BINARY_OPS)
    left  = grow(max_depth, current_depth + 1, const_range)
    right = grow(max_depth, current_depth + 1, const_range)
    return Node(op, left=left, right=right)


def full(max_depth: int, current_depth: int = 0,
         const_range: tuple[float, float] = (-10.0, 10.0)) -> Node:
    """Full initialisation: all branches reach max depth."""
    if current_depth >= max_depth:
        return _random_terminal(const_range)

    op = random.choice(BINARY_OPS)
    left  = full(max_depth, current_depth + 1, const_range)
    right = full(max_depth, current_depth + 1, const_range)
    return Node(op, left=left, right=right)


def ramped_half_and_half(max_depth: int,
                         const_range: tuple[float, float] = (-10.0, 10.0)) -> Node:
    """Ramped half-and-half: varied depths, mix of grow & full."""
    depth = random.randint(2, max_depth)
    if random.random() < 0.5:
        return grow(depth, const_range=const_range)
    return full(depth, const_range=const_range)


# -- Node-list helpers (for variation operators) -------------------------------

def collect_nodes(node: Node, path: list | None = None) -> list[tuple[list, Node]]:
    """
    Return a flat list of (path, node) for every node in the tree.
    `path` encodes how to reach the node from the root
    via 'left'/'right' strings.
    """
    if path is None:
        path = []
    result = [(path, node)]
    if not node.is_terminal():
        result += collect_nodes(node.left,  path + ['left'])
        result += collect_nodes(node.right, path + ['right'])
    return result


def get_node(root: Node, path: list) -> Node:
    """Follow a path list to retrieve a node."""
    cur = root
    for step in path:
        cur = getattr(cur, step)
    return cur


def set_node(root: Node, path: list, new_node: Node) -> Node:
    """Return a deep-copy of the tree with the node at `path` replaced."""
    root = root.clone()
    if not path:
        return new_node
    cur = root
    for step in path[:-1]:
        cur = getattr(cur, step)
    setattr(cur, path[-1], new_node)
    return root