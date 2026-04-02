"""
Fitness functions for symbolic regression.

Primary metric : Mean Squared Error (MSE).
A complexity penalty is added to discourage bloat.
"""
import math
from src.tree import Node


def mse(node: Node, data: list[tuple[float, float]]) -> float:
    """Compute Mean Squared Error on the data set."""
    total = 0.0
    for x, y in data:
        pred = node.evaluate(x)
        if not math.isfinite(pred):
            return float('inf')
        total += (pred - y) ** 2
    return total / len(data)


def fitness(node: Node, data: list[tuple[float, float]],
            complexity_weight: float = 0.005) -> float:
    """
    Fitness = MSE + complexity_weight * size.

    Lower is better.

    Parameters
    ----------
    node              : expression tree
    data              : list of (x, y) pairs
    complexity_weight : penalty coefficient for tree size (bloat control)

    Usage:
    fitness(node, data, 0.005)
    """
    error = mse(node, data)
    if not math.isfinite(error):
        return float('inf')
    penalty = complexity_weight * node.size()
    return error + penalty


def rmse(node: Node, data: list[tuple[float, float]]) -> float:
    """Root Mean Squared Error — useful for reporting."""
    return math.sqrt(mse(node, data))
