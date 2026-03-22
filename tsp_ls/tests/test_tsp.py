# tests/test_tsp.py
# =================
# Basic unit tests for TSP parsing, tour cost, and delta-cost operators.
#
# Run (from tsp_ls/ or project root):
#   python -m tests.test_tsp
#   python tests/test_tsp.py

import os
import tempfile
import math

from src.tsp import (
    read_tsplib_from_file,
    read_tsplib_dimension,
    tour_cost,
    delta_cost_2opt,
    delta_cost_vertex_switch,
    delta_cost_or_opt,
)
from src.operators import apply_2opt, apply_vertex_switching, apply_or_opt


def _write_tmp_tsplib(contents):
    fd, path = tempfile.mkstemp(suffix=".tsp")
    with os.fdopen(fd, "w") as f:
        f.write(contents)
    return path


def test_read_tsplib_and_dimension():
    tsp_text = """
NAME: square4
TYPE: TSP
COMMENT: 4 points on a square
DIMENSION: 4
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 0 1
3 1 1
4 1 0
EOF
""".lstrip()

    path = _write_tmp_tsplib(tsp_text)
    try:
        n = read_tsplib_dimension(path)
        assert n == 4

        instance = read_tsplib_from_file(path)
        assert instance.n == 4
        assert len(instance.vertex_coords) == 4
        assert instance.edge_weight_type == "EUC_2D"
    finally:
        os.remove(path)


def test_tour_cost_square():
    # Square of side 1, tour around the square => total cost 4
    tsp_text = """
NAME: square4
TYPE: TSP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 0 1
3 1 1
4 1 0
EOF
""".lstrip()

    path = _write_tmp_tsplib(tsp_text)
    try:
        instance = read_tsplib_from_file(path)
        tour = [0, 1, 2, 3]
        cost = tour_cost(tour, instance)
        assert cost == 4
    finally:
        os.remove(path)


def _delta_matches_recompute(tour, instance, delta_fn, apply_fn, move):
    before = tour_cost(tour, instance)
    delta = delta_fn(tour, instance, *move)
    apply_fn(tour, move)
    after = tour_cost(tour, instance)
    return math.isclose(after - before, delta, rel_tol=0, abs_tol=1e-9)


def test_delta_cost_2opt():
    tsp_text = """
NAME: rect5
TYPE: TSP
DIMENSION: 5
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 0 2
3 2 2
4 2 0
5 1 1
EOF
""".lstrip()

    path = _write_tmp_tsplib(tsp_text)
    try:
        instance = read_tsplib_from_file(path)
        tour = [0, 1, 2, 3, 4]
        move = (1, 3)
        assert _delta_matches_recompute(
            tour[:], instance, lambda t, inst, i, j: delta_cost_2opt(t, inst, i, j),
            apply_2opt, move
        )
    finally:
        os.remove(path)


def test_delta_cost_vertex_switch():
    tsp_text = """
NAME: rect5
TYPE: TSP
DIMENSION: 5
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 0 2
3 2 2
4 2 0
5 1 1
EOF
""".lstrip()

    path = _write_tmp_tsplib(tsp_text)
    try:
        instance = read_tsplib_from_file(path)
        tour = [0, 1, 2, 3, 4]
        move = (0, 3)
        assert _delta_matches_recompute(
            tour[:], instance, lambda t, inst, i, j: delta_cost_vertex_switch(t, inst, i, j),
            apply_vertex_switching, move
        )
    finally:
        os.remove(path)


def test_delta_cost_or_opt():
    tsp_text = """
NAME: rect5
TYPE: TSP
DIMENSION: 5
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 0 2
3 2 2
4 2 0
5 1 1
EOF
""".lstrip()

    path = _write_tmp_tsplib(tsp_text)
    try:
        instance = read_tsplib_from_file(path)
        tour = [0, 1, 2, 3, 4]
        move = (1, 3)  # relocate vertex at index 1 to position 3
        assert _delta_matches_recompute(
            tour[:], instance, lambda t, inst, i, j: delta_cost_or_opt(t, inst, i, j),
            apply_or_opt, move
        )
    finally:
        os.remove(path)


if __name__ == "__main__":
    # Run tests in a simple, dependency-free way
    test_read_tsplib_and_dimension()
    test_tour_cost_square()
    test_delta_cost_2opt()
    test_delta_cost_vertex_switch()
    test_delta_cost_or_opt()
    print("All tests passed.")
