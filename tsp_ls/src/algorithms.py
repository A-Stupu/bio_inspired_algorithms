# algorithms.py
import math
import random
from .tsp import tour_cost

EPSILON = 1e-9
RESYNC_EVERY = 500  # resync exact cost every N iterations in SA to prevent floating-point drift

### greedy local search algorithms ###

def greedy_local_search_naive_best_improvement(
        init_solution,
        fitness,
        get_neighbors,
        max_iter=1_000
):
    """
    Best-improvement greedy local search.
    At each step, evaluates ALL neighbors and moves to the best one.
    Stops when no neighbor improves the current solution, or after max_iter iterations.
    """
    x = init_solution()
    fx = fitness(x)

    for _ in range(max_iter):
        best_neighbor = None
        best_value = float("inf")

        for c in get_neighbors(x):
            fc = fitness(c)
            if fc < best_value:
                best_value = fc
                best_neighbor = c

        if best_value < fx:
            x = best_neighbor
            fx = best_value
        else:
            break

    return x, fx

def greedy_local_search_naive_first_improvement(
        init_solution,
        fitness,
        get_neighbors,
        max_iter=1_000
):
    """
    First-improvement greedy local search.
    At each step, moves to the FIRST neighbor that improves the current solution.
    Stops when no neighbor improves, or after max_iter iterations.
    """
    x = init_solution()
    fx = fitness(x)

    for _ in range(max_iter):
        improved = False

        for c in get_neighbors(x):
            fc = fitness(c)
            if fc < fx:
                x = c
                fx = fc
                improved = True
                break

        if not improved:
            break

    return x, fx

def greedy_local_search_optimized(
        tour,
        instance,
        generate_moves,   # iterable of moves
        delta_fn,
        apply_fn,
        strategy="first",
        max_iter=10_000   # ← safety cap against infinite loops
):
    """
    Optimized greedy local search using delta-cost evaluation (avoids full tour recomputation).
    current_cost is recomputed exactly after each move to avoid floating-point drift from accumulated delta additions.

    Args:
        tour:           mutable list of city indices (modified in-place)
        instance:       TSPInstance with distance matrix
        generate_moves: callable(tour) -> iterable of moves
        delta_fn:       callable(tour, instance, move) -> float (cost change)
        apply_fn:       callable(tour, move) -> None (applies move in-place)
        strategy:       "first" (first improvement) or "best" (best improvement)
        max_iter:       maximum number of improving moves (guards infinite loops)

    Returns:
        (tour, cost)
    """
    current_cost = tour_cost(tour, instance)

    for _ in range(max_iter):
        improved = False

        if strategy == "first":
            for move in generate_moves(tour):
                delta = delta_fn(tour, instance, move)
                if delta < -EPSILON:          # ← epsilon guard
                    apply_fn(tour, move)
                    current_cost = tour_cost(tour, instance)  # exact resync
                    improved = True
                    break                     # restart from new position

        elif strategy == "best":
            best_move = None
            best_delta = -EPSILON             # ← epsilon guard

            for move in generate_moves(tour):
                delta = delta_fn(tour, instance, move)
                if delta < best_delta:
                    best_delta = delta
                    best_move = move

            if best_move is not None:
                apply_fn(tour, best_move)
                current_cost = tour_cost(tour, instance)  # exact resync
                improved = True

        if not improved:
            break

    return tour, current_cost

### Simulated Annealing algorithms ###

def simulated_annealing_naive(
        init_solution,
        fitness,
        initial_temp,
        min_temp,
        update_temp,
        random_neighbor,
        max_iter=50_000
):
    """
    Naive simulated annealing.
    At each step, picks a random neighbor and accepts it with probability exp(-delta/T).

    Args:
        init_solution:   callable() -> initial tour (list)
        fitness:         callable(tour) -> float
        initial_temp:    starting temperature
        min_temp:        stopping temperature
        update_temp:     callable(T) -> new T  (use make_cooling_schedule())
        random_neighbor: callable(tour) -> new tour
        max_iter:        hard cap on iterations (guards against slow cooling schedules)

    Returns:
        (best_tour, best_cost)
    """
    x = init_solution()
    fx = fitness(x)
    best = x[:]
    best_fx = fx
    T = initial_temp

    for _ in range(max_iter):
        if T <= min_temp:
            break

        c = random_neighbor(x)
        fc = fitness(c)
        delta = fc - fx

        if delta < 0 or random.random() < math.exp(-delta / T):
            x = c
            fx = fc
            if fx < best_fx:
                best = x[:]
                best_fx = fx

        T = update_temp(T)

    return best, best_fx

def simulated_annealing_optimized(
        tour,
        instance,
        generate_random_move,
        delta_fn,
        apply_fn,
        T,
        min_T,
        update_T,
        max_iter
):
    """
    Optimized simulated annealing using delta-cost evaluation.
    Keeps track of the best solution found during the search.
    current_cost is resynced exactly every RESYNC_EVERY iterations to prevent floating-point drift from turning negative.

    Args:
        tour:                 mutable list of city indices (modified in-place)
        instance:             TSPInstance with distance matrix
        generate_random_move: callable(tour) -> move
        delta_fn:             callable(tour, instance, move) -> float
        apply_fn:             callable(tour, move) -> None
        T:                    initial temperature
        min_T:                stopping temperature
        update_T:             callable(T) -> new T
        max_iter:             maximum number of iterations

    Returns:
        (best_tour, best_cost)
    """
    current_cost = tour_cost(tour, instance)
    best = tour[:]
    best_cost = current_cost

    for i in range(max_iter):
        if T < min_T:
            break

        move = generate_random_move(tour)
        delta = delta_fn(tour, instance, move)

        if delta < 0 or random.random() < math.exp(-delta / T):
            apply_fn(tour, move)
            current_cost += delta

            if current_cost < best_cost:
                best = tour[:]
                best_cost = current_cost

        # Periodic exact resync to prevent floating-point drift
        if i % RESYNC_EVERY == 0:
            current_cost = tour_cost(tour, instance)

        T = update_T(T)

    return best, best_cost

### Simulated Annealing - cooling schedules ###

def make_cooling_schedule(method, T0, alpha=None):
    """
    Returns an update_T callable based on the chosen cooling method.

    Args:
        method: "exponential" | "polynomial" | "logarithmic"
        T0:     initial temperature (used as reference for polynomial/logarithmic)
        alpha:  cooling parameter (meaning depends on method, see below)

    Cooling behaviours:
        exponential:  T_k = alpha * T_{k-1}          (alpha in (0,1), e.g. 0.99, 0.995, 0.999)
        polynomial:   T_k = T0 / (k+1)^alpha          (alpha > 0, e.g. 0.5, 1.0, 2.0)
        logarithmic:  T_k = T0 / log(k+2)^alpha       (alpha > 0, e.g. 0.5, 1.0, 2.0)

    Returns a stateful callable: update_T() -> new T
    """
    k = [0]  # mutable counter (closure trick)

    if method == "exponential":
        if alpha is None:
            alpha = 0.995

        def update_T(T):
            return T * alpha

    elif method == "polynomial":
        if alpha is None:
            alpha = 1.0

        def update_T(T):
            k[0] += 1
            return T0 / ((k[0] + 1) ** alpha)

    elif method == "logarithmic":
        if alpha is None:
            alpha = 1.0

        def update_T(T):
            k[0] += 1
            return T0 / (math.log(k[0] + 2) ** alpha)

    else:
        raise ValueError(f"Unknown cooling method: {method}")

    return update_T
