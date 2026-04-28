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


# helpers

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


# Mutation

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


# Crossover

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


# Constant optimisation (local search)

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


# Structure-aware rational seeding

def make_rational_seed(a: float, b: float, c: float, d: float) -> 'Node':
    """
    Build the tree (a + b*x) / (c + d*x^2) explicitly.
    Used to seed the population for ratio instances.
    Structure: div( add(const_a, mul(const_b, x)), add(const_c, mul(const_d, pow2)) )
    """
    num = Node('+',
               left=Node(TERMINAL_CONST, value=a),
               right=Node('*',
                          left=Node(TERMINAL_CONST, value=b),
                          right=Node(TERMINAL_X)))
    den = Node('+',
               left=Node(TERMINAL_CONST, value=c),
               right=Node('*',
                          left=Node(TERMINAL_CONST, value=d),
                          right=Node(TERMINAL_POW, value=2)))
    return Node('/', left=num, right=den)


def seed_rational_population(data: list, n: int = 20,
                              const_range: tuple = (-10.0, 10.0)) -> list:
    """
    Generate n rational seeds of the form (a+b*x)/(c+d*x^2) with
    constants optimised by hill-climbing.

    All seeds have b != 0 to ensure the x term in the numerator is present —
    this is critical for functions like (3+x)/(2+x^2) where a purely
    constant numerator gives a structurally different (worse) approximation.
    """
    import random as _random
    seeds = []
    for _ in range(n):
        a = _random.uniform(*const_range)
        # b is always non-zero so numerator always contains x
        b = _random.uniform(0.1, const_range[1]) * _random.choice([-1, 1])
        c = abs(_random.uniform(0.5, const_range[1]))  # positive denominator const
        d = abs(_random.uniform(0.1, 3.0))
        node = make_rational_seed(a, b, c, d)
        node = optimise_constants(node, data, n_steps=80, sigma_init=1.0)
        seeds.append(node)
    return seeds


# Factored polynomial seeding

def make_factored_poly(roots: list[float], leading: float = 1.0) -> 'Node':
    """
    Build the tree  leading * (x + roots[0]) * (x + roots[1]) * ...
    Roots are stored as additive offsets: (x + r) where r = -actual_root.
    """
    xn = lambda: Node(TERMINAL_X)
    cn = lambda v: Node(TERMINAL_CONST, value=v)

    # Start with (x + roots[0])
    tree = Node('+', left=xn(), right=cn(roots[0]))
    for r in roots[1:]:
        factor = Node('+', left=xn(), right=cn(r))
        tree   = Node('*', left=tree, right=factor)
    # Multiply by leading coefficient
    return Node('*', left=tree, right=cn(leading))


def seed_factored_population(data: list, degree: int, n: int = 30,
                              root_range: tuple = (-8.0, 8.0),
                              n_opt_steps: int = 200) -> list:
    """
    Generate `n` factored polynomial seeds of a given degree:
        leading * (x + r1) * (x + r2) * ... * (x + r_degree)
    Constants are randomly initialised then hill-climbed.

    Useful for polynomial instances where the GP builds bloated trees
    instead of compact factored forms (e.g. challenge_a_03, challenge_b_01).
    """
    import random as _r
    seeds = []
    for _ in range(n):
        roots   = [_r.uniform(*root_range) for _ in range(degree)]
        leading = _r.uniform(-2.0, 2.0)
        if abs(leading) < 0.05:
            leading = 1.0
        node = make_factored_poly(roots, leading)
        node = optimise_constants(node, data, n_steps=n_opt_steps, sigma_init=1.0)
        seeds.append(node)
    return seeds


# Gradient-based constant optimisation

def optimise_constants_gradient(root: 'Node', data: list,
                                 n_steps: int = 800,
                                 lr: float = 0.01) -> 'Node':
    """
    Adam gradient descent on constant leaves, keeping tree structure fixed.

    More effective than hill-climbing for rational expressions where the
    hill-climber gets trapped on ridges (e.g. (a+bx)/(c+x^2) where the
    true optimum requires precise co-tuning of a, b, c).

    Falls back to optimise_constants (hill-climbing) if fewer than 2
    constant nodes are present.
    """
    try:
        import numpy as np
    except ImportError:
        return optimise_constants(root, data, n_steps, sigma_init=0.3)

    root = root.clone()
    const_paths = [p for p, n in collect_nodes(root)
                   if n.node_type == TERMINAL_CONST]
    if len(const_paths) < 2:
        return optimise_constants(root, data, n_steps)

    xs = np.array([x for x, _ in data])
    ys = np.array([y for _, y in data])

    def get_params():
        return np.array([get_node(root, p).value for p in const_paths])

    def set_params(params):
        for p, v in zip(const_paths, params):
            get_node(root, p).value = float(v)

    def mse_and_numerical_grad(params, eps=1e-5):
        set_params(params)
        preds = np.array([root.evaluate(x) for x in xs])
        if not np.all(np.isfinite(preds)):
            return float('inf'), np.zeros_like(params)
        res = preds - ys
        mse = float(np.mean(res**2))
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy(); p_plus[i] += eps
            set_params(p_plus)
            pp = np.array([root.evaluate(x) for x in xs])
            if np.all(np.isfinite(pp)):
                grad[i] = (np.mean((pp - ys)**2) - mse) / eps
        set_params(params)
        return mse, grad

    params = get_params()
    m  = np.zeros_like(params)
    v  = np.zeros_like(params)
    b1, b2, ep = 0.9, 0.999, 1e-8
    best_mse, best_params = float('inf'), params.copy()

    for step in range(1, n_steps + 1):
        mse, grad = mse_and_numerical_grad(params)
        if not np.isfinite(mse):
            break
        if mse < best_mse:
            best_mse   = mse
            best_params = params.copy()
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        mh = m / (1 - b1**step)
        vh = v / (1 - b2**step)
        params = params - lr * mh / (np.sqrt(vh) + ep)

    set_params(best_params)
    return root


# Degree-(2,1) rational seed for exponential-like instances

def make_21_rational(a: float, b: float, c: float, d: float) -> 'Node':
    """
    Build the tree  (a + b*x + c*x^2) / (1 + d*x).
    Pole is at x = -1/d; keep d small and negative to push pole outside [0,2].
    Best Padé approximant for 2^(1+1.5x) on [0,2]:
        a=2.011, b=1.435, c=0.981, d=-0.225  →  RMSE≈0.0086
    """
    xn  = lambda: Node(TERMINAL_X)
    cn  = lambda v: Node(TERMINAL_CONST, value=v)
    px2 = Node(TERMINAL_POW, value=2)
    num = Node('+',
               left=Node('+', left=cn(a),
                               right=Node('*', left=cn(b), right=xn())),
               right=Node('*', left=cn(c), right=px2))
    den = Node('+', left=cn(1.0),
                     right=Node('*', left=cn(d), right=xn()))
    return Node('/', left=num, right=den)


def seed_21_rational_population(data: list, n: int = 20) -> list:
    """
    Generate n seeds of the form (a+b*x+c*x^2)/(1+d*x) refined by Adam
    gradient descent.  Designed for rapidly-growing functions like
    exponentials that are outside the polynomial/integer grammar.
    """
    import random as _r
    seeds = []
    # Always include the analytical Padé starting point
    anchor = make_21_rational(2.011, 1.435, 0.981, -0.2252)
    anchor = optimise_constants_gradient(anchor, data, n_steps=800, lr=0.005)
    seeds.append(anchor)

    for _ in range(n - 1):
        # Perturb around the anchor
        a = _r.uniform(1.5, 3.0)
        b = _r.uniform(0.5, 3.0)
        c = _r.uniform(0.3, 2.0)
        d = _r.uniform(-0.4, -0.05)   # negative → pole outside [0,∞)
        node = make_21_rational(a, b, c, d)
        node = optimise_constants_gradient(node, data, n_steps=600, lr=0.005)
        seeds.append(node)
    return seeds


# Linear-denominator ratio seeding  c/(d+x)

def make_linear_ratio(c: float, d: float) -> 'Node':
    """Build the tree  c / (d + x)."""
    return Node('/',
                left=Node(TERMINAL_CONST, value=c),
                right=Node('+',
                           left=Node(TERMINAL_CONST, value=d),
                           right=Node(TERMINAL_X)))


def seed_linear_ratio_population(data: list, n: int = 20) -> list:
    """
    Generate n seeds of the form c/(d+x) refined by gradient descent.
    Designed for ratio instances like 3/(x+2) where the denominator
    is linear — the standard rational seeder (quadratic denom) misses this.
    """
    import random as _r
    seeds = []
    ys = [y for _, y in data]
    # Rough estimate: c ≈ y(0)*(d+0) = y(0)*d, d from slope
    y0  = ys[0]
    yend = ys[-1]
    x0, xend = data[0][0], data[-1][0]
    # c/(d+x): at x=0: c/d=y0; at x=xend: c/(d+xend)=yend
    # -> d = xend*yend/(y0-yend); c = y0*d
    try:
        d_est = xend * yend / (y0 - yend)
        c_est = y0 * d_est
    except ZeroDivisionError:
        d_est, c_est = 2.0, 3.0

    # Always include the analytically estimated seed
    anchor = make_linear_ratio(c_est, d_est)
    anchor = optimise_constants_gradient(anchor, data, n_steps=600, lr=0.01)
    seeds.append(anchor)

    for _ in range(n - 1):
        c = _r.uniform(0.5, max(10, abs(c_est) * 2))
        d = _r.uniform(0.1, max(5,  abs(d_est) * 2))
        node = make_linear_ratio(c, d)
        node = optimise_constants_gradient(node, data, n_steps=400, lr=0.01)
        seeds.append(node)
    return seeds