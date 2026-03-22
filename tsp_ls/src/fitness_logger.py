# src/fitness_logger.py — Instrumented Versions of GLS and SA
# =============================================================
#
# Variants of greedy_local_search_optimized and simulated_annealing_optimized
# that record the cost history at each iteration.
#
# These functions are intended for analysis/visualization only.
# They do not replace the versions in algorithms.py used in run.py.
#
# Each function returns:
#     (best_tour, best_cost, history)
#
# where history is a list of (iteration: int, cost: float) recorded
# at each iteration (GLS: at each improvement; SA: at each acceptance).

import math
import random

from .tsp import tour_cost

EPSILON = 1e-9

# Instrumented GLS --------------------------------------------------------------

def gls_with_history(
    tour,
    instance,
    generate_moves,
    delta_fn,
    apply_fn,
    strategy="first",
    max_iter=10_000,
):
    """
    Optimized GLS (delta-cost) with cost logging at each improvement.

    Returns: (tour, cost, history)
    history: list of (iteration, cost) — one point at each improvement
             + the initial point (0, init_cost)
    """
    current_cost = tour_cost(tour, instance)
    history = [(0, current_cost)]
    iteration = 0

    for _ in range(max_iter):
        improved = False
        iteration += 1

        if strategy == "first":
            for move in generate_moves(tour):
                delta = delta_fn(tour, instance, move)
                if delta < -EPSILON:
                    apply_fn(tour, move)
                    current_cost = tour_cost(tour, instance)
                    history.append((iteration, current_cost))
                    improved = True
                    break

        elif strategy == "best":
            best_move  = None
            best_delta = -EPSILON
            for move in generate_moves(tour):
                delta = delta_fn(tour, instance, move)
                if delta < best_delta:
                    best_delta = delta
                    best_move  = move
            if best_move is not None:
                apply_fn(tour, best_move)
                current_cost = tour_cost(tour, instance)
                history.append((iteration, current_cost))
                improved = True

        if not improved:
            break

    return tour, current_cost, history

# Instrumented SA --------------------------------------------------------------

RESYNC_EVERY = 500

def sa_with_history(
    tour,
    instance,
    generate_random_move,
    delta_fn,
    apply_fn,
    T,
    min_T,
    update_T,
    max_iter,
    log_every=100,
):
    """
    Optimized SA (delta-cost) with current cost logging every
    `log_every` iterations (+ at each improvement of the best).

    Args:
        log_every: sampling frequency for history (in iterations)
                   → keep ~500 points max for readable plots

    Returns: (best_tour, best_cost, history)
    history: list of (iteration, cost) — sampled current cost
    """
    current_cost = tour_cost(tour, instance)
    best         = tour[:]
    best_cost    = current_cost
    history      = [(0, current_cost)]

    for i in range(1, max_iter + 1):
        if T < min_T:
            break

        move  = generate_random_move(tour)
        delta = delta_fn(tour, instance, move)

        if delta < 0 or random.random() < math.exp(-delta / T):
            apply_fn(tour, move)
            current_cost += delta

            if current_cost < best_cost:
                best      = tour[:]
                best_cost = current_cost

        # Periodic resync to avoid floating-point drift
        if i % RESYNC_EVERY == 0:
            current_cost = tour_cost(tour, instance)

        # History sampling
        if i % log_every == 0:
            history.append((i, current_cost))

        T = update_T(T)

    # Ensure the last point is recorded
    if history[-1][0] != i:
        history.append((i, current_cost))

    return best, best_cost, history
