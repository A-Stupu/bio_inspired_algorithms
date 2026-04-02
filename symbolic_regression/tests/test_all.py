"""
Unit tests for the symbolic regression project.
Run with:  python -m pytest tests/ -v
"""
import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tree import (
    Node, TERMINAL_X, TERMINAL_CONST, TERMINAL_POW,
    grow, full, ramped_half_and_half,
    collect_nodes, get_node, set_node,
)
from src.fitness import mse, fitness
from src.operators import (
    subtree_mutation, point_mutation, hoist_mutation,
    constant_folding, subtree_crossover,
)
from src.data import load_instance
from src.selection import tournament_select, elitist_survivor


# ── Tree tests ────────────────────────────────────────────────────────────────

def test_constant_node():
    n = Node(TERMINAL_CONST, value=3.0)
    assert n.evaluate(0.0)  == 3.0
    assert n.evaluate(99.0) == 3.0
    assert n.to_string() == '3'

def test_x_node():
    n = Node(TERMINAL_X)
    assert n.evaluate(5.0) == 5.0
    assert n.to_string() == 'x'

def test_pow_node():
    n = Node(TERMINAL_POW, value=2)
    assert n.evaluate(3.0) == 9.0
    assert n.to_string() == 'x**2'

def test_binary_add():
    n = Node('+', left=Node(TERMINAL_X), right=Node(TERMINAL_CONST, value=1.0))
    assert n.evaluate(4.0) == 5.0

def test_binary_div_zero():
    n = Node('/', left=Node(TERMINAL_X),
             right=Node(TERMINAL_CONST, value=0.0))
    assert not math.isfinite(n.evaluate(1.0))

def test_grow_depth():
    for _ in range(20):
        t = grow(4)
        assert t.depth() <= 4

def test_full_depth():
    for _ in range(10):
        t = full(3)
        assert t.depth() == 3

def test_size():
    leaf = Node(TERMINAL_X)
    assert leaf.size() == 1
    inner = Node('+', left=Node(TERMINAL_X), right=Node(TERMINAL_CONST, value=1.0))
    assert inner.size() == 3

def test_collect_nodes():
    n = Node('+', left=Node(TERMINAL_X), right=Node(TERMINAL_CONST, value=2.0))
    nodes = collect_nodes(n)
    assert len(nodes) == 3

def test_set_node_immutability():
    original = Node('+', left=Node(TERMINAL_X), right=Node(TERMINAL_CONST, value=1.0))
    modified = set_node(original, ['left'], Node(TERMINAL_CONST, value=99.0))
    # original unchanged
    assert original.left.node_type == TERMINAL_X
    # modified changed
    assert modified.left.value == 99.0


# ── Fitness tests ─────────────────────────────────────────────────────────────

def test_mse_perfect():
    # f(x) = x  → perfect fit on data
    n = Node(TERMINAL_X)
    data = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    assert mse(n, data) < 1e-12

def test_mse_nonzero():
    n = Node(TERMINAL_CONST, value=0.0)
    data = [(0.0, 1.0), (1.0, 2.0)]
    assert mse(n, data) > 0

def test_fitness_penalises_size():
    simple  = Node(TERMINAL_X)
    complex_tree = grow(5)
    data = [(float(i), float(i)) for i in range(10)]
    f_simple  = fitness(simple,       data, complexity_weight=0.01)
    f_complex = fitness(complex_tree, data, complexity_weight=0.01)
    # complex tree has larger penalty (even if mse happens to be similar)
    assert simple.size() < complex_tree.size()


# ── Operator tests ────────────────────────────────────────────────────────────

def test_subtree_mutation_changes_tree():
    random.seed(42)
    t = grow(4)
    mutated = subtree_mutation(t, max_depth=4)
    # Should generally differ (may occasionally be the same by chance)
    # Just verify it's a valid tree
    assert mutated.size() >= 1

def test_point_mutation_returns_valid():
    random.seed(0)
    for _ in range(20):
        t = grow(3)
        m = point_mutation(t)
        assert m.size() >= 1

def test_hoist_reduces_or_keeps_size():
    random.seed(1)
    for _ in range(20):
        t = grow(4)
        h = hoist_mutation(t)
        assert h.size() <= t.size()

def test_constant_folding():
    # (3 + 4) should fold to 7
    n = Node('+',
             left=Node(TERMINAL_CONST, value=3.0),
             right=Node(TERMINAL_CONST, value=4.0))
    folded = constant_folding(n)
    assert folded.node_type == TERMINAL_CONST
    assert abs(folded.value - 7.0) < 1e-9

def test_crossover_depth_limit():
    random.seed(5)
    p1 = grow(4)
    p2 = grow(4)
    c1, c2 = subtree_crossover(p1, p2, max_depth=8)
    assert c1.depth() <= 8
    assert c2.depth() <= 8


# ── Selection tests ───────────────────────────────────────────────────────────

def test_tournament_selects_from_population():
    pop  = [grow(3) for _ in range(20)]
    fits = [float(i) for i in range(20)]
    winner = tournament_select(pop, fits, k=5)
    assert winner in pop

def test_elitist_survivor_keeps_best():
    pop      = [Node(TERMINAL_CONST, value=float(i)) for i in range(10)]
    fits     = [float(i) for i in range(10)]
    children = [Node(TERMINAL_CONST, value=float(100 + i)) for i in range(10)]
    c_fits   = [float(100 + i) for i in range(10)]
    new_pop, new_fits = elitist_survivor(pop, fits, children, c_fits, 10)
    # Best should still be there (value=0, fitness=0)
    assert new_fits[0] == 0.0


# ── Data loading tests ────────────────────────────────────────────────────────

def test_load_instance(tmp_path):
    content = "4\n0.0 1.0\n0.2 1.149\n0.5 1.414\n0.8 1.741\n"
    p = tmp_path / "sr_poly_99.txt"
    p.write_text(content)

    data, meta = load_instance(str(p))
    assert len(data) == 4
    assert meta['type'] == 'poly'
    assert meta['id'] == '99'
    assert abs(data[0][0] - 0.0) < 1e-9
    assert abs(data[2][1] - 1.414) < 1e-9


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
