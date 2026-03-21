# algorithms.py
import math
import random
from .tsp import tour_cost

### greedy local search algorithms ###

def greedy_local_search_naive_best_improvement(
        init_solution, 
        fitness, 
        get_neighbors
):
    """
    Best-improvement greedy local search.
    At each step, evaluates ALL neighbors and moves to the best one.
    Stops when no neighbor improves the current solution.
    """
    x = init_solution()
    fx = fitness(x)

    while True:
        neighbors = get_neighbors(x)

        best_neighbor = None
        best_value = float("inf")

        for c in neighbors:
            fc = fitness(c)
            if fc < best_value:
                best_value = fc
                best_neighbor = c

        if best_value < fitness(x):
            x = best_neighbor
            fx = best_value
        else:
            break
    
    return x, fx


def greedy_local_search_naive_first_improvement(
        init_solution, 
        fitness, 
        get_neighbors
):
    """
    First-improvement greedy local search.
    At each step, moves to the FIRST neighbor that improves the current solution.
    Stops when no neighbor improves.
    """
    x = init_solution()
    fx = fitness(x)
    while True:
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
        strategy="first"
):
    """
    Optimized greedy local search using delta-cost evaluation (avoids full tour recomputation).
 
    Args:
        tour:           mutable list of city indices (modified in-place)
        instance:       TSPInstance with distance matrix
        generate_moves: callable(tour) -> iterable of moves
        delta_fn:       callable(tour, instance, move) -> float (cost change)
        apply_fn:       callable(tour, move) -> None (applies move in-place)
        strategy:       "first" (first improvement) or "best" (best improvement)
 
    Returns:
        (tour, cost)
    """
    current_cost = tour_cost(tour, instance)
    improved = True
 
    while improved:
        improved = False
 
        if strategy == "first":
            for move in generate_moves(tour):
                delta = delta_fn(tour, instance, move)
                if delta < 0:
                    apply_fn(tour, move)
                    current_cost += delta
                    improved = True
                    break
 
        elif strategy == "best":
            best_move = None
            best_delta = 0
 
            for move in generate_moves(tour):
                delta = delta_fn(tour, instance, move)
                if delta < best_delta:
                    best_delta = delta
                    best_move = move
 
            if best_move is not None:
                apply_fn(tour, best_move)
                current_cost += best_delta
                improved = True
 
    return tour, current_cost


### Simulated Annealing algorithms ###

def simulated_annealing_naive(
        init_solution, 
        fitness,
        initial_temp, 
        min_temp,
        update_temp,
        random_neighbor
):
    """
    Naive simulated annealing.
    At each step, picks a random neighbor and accepts it with probability exp(-delta/T).
 
    Args:
        init_solution:  callable() -> initial tour (list)
        fitness:        callable(tour) -> float
        initial_temp:   starting temperature
        min_temp:       stopping temperature
        update_temp:    callable(T) -> new T  (use make_cooling_schedule())
        random_neighbor: callable(tour) -> new tour
 
    Returns:
        (best_tour, best_cost)
    """
    x = init_solution()
    fx = fitness(x)
    T = initial_temp
    while T > min_temp:
        c = random_neighbor(x)
        fc = fitness(c)
        delta = fc - fx

        if delta < 0 or random.random() < math.exp(-delta / T):
            x = c
            fx = fc
        
        T = update_temp(T)
    
    return x, fx
    

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

    for _ in range(max_iter):
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