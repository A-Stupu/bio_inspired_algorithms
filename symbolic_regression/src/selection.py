"""
Selection methods for the genetic programming evolutionary loop.

All selectors take a population (list of Node) and a fitnesses list (same order,
lower = better) and return one or two individuals.
"""
import random
import math


# -- Tournament selection ------------------------------------------------------

def tournament_select(population: list, fitnesses: list[float],
                      k: int = 7) -> object:
    """
    k-tournament selection: sample k individuals and return the best one.
    """
    indices = random.sample(range(len(population)), min(k, len(population)))
    best = min(indices, key=lambda i: fitnesses[i])
    return population[best]


def tournament_select_pair(population: list, fitnesses: list[float],
                           k: int = 7) -> tuple:
    """Select two distinct parents via tournament."""
    p1 = tournament_select(population, fitnesses, k)
    # Ensure parents are different individuals (by index)
    attempts = 0
    while True:
        p2 = tournament_select(population, fitnesses, k)
        if p2 is not p1 or attempts > 5:
            break
        attempts += 1
    return p1, p2


# -- Fitness-proportionate (roulette wheel) ------------------------------------

def roulette_select(population: list, fitnesses: list[float]) -> object:
    """
    Fitness-proportionate selection.
    Converts minimisation fitnesses to weights via 1 / (1 + f).
    """
    weights = [1.0 / (1.0 + f) if math.isfinite(f) else 0.0 for f in fitnesses]
    total = sum(weights)
    if total == 0:
        return random.choice(population)
    r = random.uniform(0, total)
    cumulative = 0.0
    for ind, w in zip(population, weights):
        cumulative += w
        if cumulative >= r:
            return ind
    return population[-1]


# -- Over-selection ------------------------------------------------------------

def over_selection(population: list, fitnesses: list[float],
                   top_fraction: float = 0.2,
                   top_prob: float = 0.8) -> object:
    """
    Over-selection: split population into top x% and bottom (1-x)%.
    Select from the top group with probability top_prob.

    Designed for large populations (thousands).
    """
    n = len(population)
    sorted_idx = sorted(range(n), key=lambda i: fitnesses[i])
    top_size = max(1, int(n * top_fraction))

    if random.random() < top_prob:
        idx = random.choice(sorted_idx[:top_size])
    else:
        idx = random.choice(sorted_idx[top_size:])
    return population[idx]


# -- Survivor selection --------------------------------------------------------

def elitist_survivor(population: list, fitnesses: list[float],
                     children: list, child_fitnesses: list[float],
                     n: int) -> tuple[list, list[float]]:
    """
    (µ + λ) selection: keep the best n individuals from parents ∪ children.
    """
    combined      = list(zip(population + children, fitnesses + child_fitnesses))
    combined.sort(key=lambda x: x[1])
    combined = combined[:n]
    new_pop, new_fit = zip(*combined)
    return list(new_pop), list(new_fit)


def generational_survivor(children: list, child_fitnesses: list[float],
                           elite_count: int,
                           population: list, fitnesses: list[float]
                           ) -> tuple[list, list[float]]:
    """
    Generational replacement with elitism:
    keep the top `elite_count` individuals from the previous generation
    and fill the rest with children.
    """
    # Sort parents and take elites
    parent_pairs = sorted(zip(population, fitnesses), key=lambda x: x[1])
    elites   = [p for p, _ in parent_pairs[:elite_count]]
    e_fits   = [f for _, f in parent_pairs[:elite_count]]

    # Fill the rest with children (best first)
    child_pairs = sorted(zip(children, child_fitnesses), key=lambda x: x[1])
    fill = len(population) - elite_count
    filled_children = [p for p, _ in child_pairs[:fill]]
    filled_fits     = [f for _, f in child_pairs[:fill]]

    return elites + filled_children, e_fits + filled_fits
