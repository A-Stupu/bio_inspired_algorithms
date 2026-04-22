"""
Genetic operators for tree-based genetic programming.

Mutation
--------
- subtree_mutation  : replace a random subtree with a new random one
- point_mutation    : replace a terminal's value (constant nudge / new const)
- hoist_mutation    : replace a node with one of its subtrees (shrinks tree)

Crossover
---------
- subtree_crossover : exchange a random subtree between two parents
"""
import random
from src.tree import (
    Node, BINARY_OPS, TERMINAL_X, TERMINAL_CONST, TERMINAL_POW,
    grow, ramped_half_and_half,
    collect_nodes, get_node, set_node,
    MAX_POWER,
)


# -- helpers -------------------------------------------------------------------

def _pick_random_node(root: Node) -> tuple[list, Node]:
    """Return a (path, node) chosen uniformly at random."""
    nodes = collect_nodes(root)
    return random.choice(nodes)


def _pick_internal_node(root: Node) -> tuple[list, Node] | None:
    """Return a random internal (non-terminal) node, or None if tree is a single leaf."""
    internals = [(p, n) for p, n in collect_nodes(root) if not n.is_terminal()]
    if not internals:
        return None
    return random.choice(internals)


# -- Mutation ------------------------------------------------------------------

def subtree_mutation(root: Node, max_depth: int = 4,
                     const_range: tuple[float, float] = (-10.0, 10.0)) -> Node:
    """
    Subtree mutation: pick a random node and replace it with a
    freshly generated random subtree.
    """
    path, _ = _pick_random_node(root)
    new_subtree = ramped_half_and_half(max_depth, const_range)
    return set_node(root, path, new_subtree)


def point_mutation(root: Node,
                   const_range: tuple[float, float] = (-10.0, 10.0),
                   sigma: float = 1.0) -> Node:
    """
    Point mutation: modify a randomly chosen terminal node.
    - CONST    -> multi-scale Gaussian nudge or full resample (float-aware)
    - X        -> replace with a random terminal
    - POW      -> change exponent by +/-1
    - operator -> randomly pick another operator
    """
    root = root.clone()
    path, node = _pick_random_node(root)

    if node.node_type == TERMINAL_CONST:
        r = random.random()
        if r < 0.50:
            # Fine-grained nudge: scale sigma to magnitude so we can
            # converge toward irrational values (pi, e, sqrt(2), ...)
            scale = max(abs(node.value) * 0.1, sigma * 0.1)
            node.value += random.gauss(0, scale)
        elif r < 0.80:
            # Coarse nudge (original behaviour)
            node.value += random.gauss(0, sigma)
        else:
            # Full resample as float (not just integer)
            node.value = random.uniform(const_range[0], const_range[1])
            if abs(node.value) < 1e-6:
                node.value = 1.0

    elif node.node_type == TERMINAL_POW:
        delta = random.choice([-1, 1])
        node.value = max(2, min(MAX_POWER, node.value + delta))

    elif node.node_type == TERMINAL_X:
        # Replace x with a random terminal (const or pow)
        new_node = get_node(
            set_node(root, path,
                     Node(TERMINAL_CONST,
                          value=random.uniform(const_range[0], const_range[1]))),
            path
        )

    elif node.node_type in BINARY_OPS:
        ops = [o for o in BINARY_OPS if o != node.node_type]
        node.node_type = random.choice(ops)

    return root


def hoist_mutation(root: Node) -> Node:
    """
    Hoist mutation: replace a random internal node with one of its subtrees.
    This naturally reduces tree size (fights bloat).
    Falls back to the original tree if no internal node exists.
    """
    result = _pick_internal_node(root)
    if result is None:
        return root.clone()

    path, node = result
    # Pick left or right child
    child = node.left if random.random() < 0.5 else node.right
    return set_node(root, path, child)


def constant_folding(root: Node) -> Node:
    """
    Simplification: if both children of an operator are constants,
    evaluate and replace with a constant node.  Applied recursively.
    """
    root = root.clone()
    _fold(root)
    return root


def _fold(node: Node):
    """In-place recursive constant folding."""
    if node.is_terminal():
        return
    _fold(node.left)
    _fold(node.right)

    if node.left.node_type == TERMINAL_CONST and node.right.node_type == TERMINAL_CONST:
        try:
            dummy = Node(node.node_type, left=node.left, right=node.right)
            val = dummy.evaluate(0.0)
            if not (val != val) and abs(val) < 1e6:   # not NaN and not huge
                node.node_type = TERMINAL_CONST
                node.value     = val
                node.left      = None
                node.right     = None
        except Exception:
            pass


# -- Crossover -----------------------------------------------------------------

def subtree_crossover(parent1: Node, parent2: Node,
                      max_depth: int = 8) -> tuple[Node, Node]:
    """
    Standard subtree crossover:
    - Pick a random subtree in each parent.
    - Exchange them.
    - Enforce max_depth by falling back to the original parent if exceeded.

    Returns two children.
    """
    p1, p2 = parent1.clone(), parent2.clone()

    path1, _ = _pick_random_node(p1)
    path2, _ = _pick_random_node(p2)

    subtree1 = get_node(p1, path1).clone()
    subtree2 = get_node(p2, path2).clone()

    child1 = set_node(p1, path1, subtree2)
    child2 = set_node(p2, path2, subtree1)

    # Depth guard — revert to parent if too deep
    if child1.depth() > max_depth:
        child1 = parent1.clone()
    if child2.depth() > max_depth:
        child2 = parent2.clone()

    return child1, child2


# -- Constant optimisation (local search) -------------------------------------

def optimise_constants(root: Node,
                       data: list[tuple[float, float]],
                       n_steps: int = 40,
                       sigma_init: float = 0.5) -> Node:
    """
    Hill-climbing local search on the constant leaves of a fixed tree structure.

    For each constant node, try a Gaussian perturbation; keep it if MSE improves.
    Multiple passes with decaying sigma (simulated annealing-like schedule).

    This is especially useful for challenge instances that involve irrational
    constants (pi, e, sqrt(2), ...) which the main GP struggles to pin down.

    Parameters
    ----------
    root      : tree whose structure is kept fixed
    data      : training data
    n_steps   : total number of perturbation attempts per constant per pass
    sigma_init: initial standard deviation for perturbations
    """
    from src.fitness import mse as _mse

    best = root.clone()
    best_mse = _mse(best, data)

    # Collect paths to all constant nodes
    const_paths = [p for p, n in collect_nodes(best) if n.node_type == TERMINAL_CONST]
    if not const_paths:
        return best

    sigma = sigma_init
    for step in range(n_steps):
        # Decay sigma geometrically
        sigma = sigma_init * (0.3 ** (step / n_steps))

        for path in const_paths:
            node = get_node(best, path)
            orig = node.value

            # Try a perturbation
            node.value = orig + random.gauss(0, max(sigma, abs(orig) * 0.05))
            new_mse = _mse(best, data)

            if new_mse < best_mse:
                best_mse = new_mse          # accept
            else:
                node.value = orig           # reject

    return best