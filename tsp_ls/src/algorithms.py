# algorithms.py
import math
import random
from .tsp import tour_cost

### greedy local search algorithms ###

# slides 32 & 33 /83, Local Search:

# Algorithm GreedyLocalSearch
# Input: domain D, fitness f
# Output: solution x belongs to D
# 1: x_0 := RandomSolution(D)
# 2: i := 1
# 3: while true do
# 4: N := Neighbors(x, D)
# 5: xi := argmin{f (c) : c belongs to N}
# 6: if f (x_i ) >= f (x_{i−1}) then
# 7: break
# 8: end if
# 9: i := i + 1
# 10: end while
# 11: return x_i

# def greedy_local_search(D, f):
#     x = D.sample()
#     while True:
#         N = D.neighbors(x)
#         fc, c = min( (f(c), c) for c in N )
#         if f(c) < f(x):
#             x = c
#         else:
#             break
#     return x


def greedy_local_search_naive_best_neighbor(
        init_solution, 
        fitness, 
        get_neighbors
        ):
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


def greedy_local_search_naive_first_neighbor(
        init_solution, 
        fitness, 
        get_neighbors
    ):
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

# Algorithm SimulatedAnnealing
# Input: domain D, fitness f , temperature T
# Output: solution x belongs to D
# 1: x := RandomSolution(D)
# 2: while T approx not equal to 0 do
# 3: c := RandomNeighbor(x, D)
# 4: if (f (c) < f (x)) OR (random( ) <= e−(f (c)−f (x))/T ) then
# 5: x := c
# 6: end if
# 7: T := UpdateTemp(T )
# 8: end while
# 9: return x


def simulated_annealing_naive(
        init_solution, 
        fitness,
        initial_temp, 
        min_temp,
        update_temp,
        random_neighbor
    ):
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

# see also: slide 36/53 on temperature reduction

# Temperature reduction
# 1. Geometric: Tk = αTk−1
# 2. Polynomial: Tk = T0/(k + 1)
# • Or equivalently Tk = Tk−1/
# (
# 1 + Tk−1
# T0
# )
# 3. Logarithmic: Tk = T0/ log(k + 2)
# • Usually with stage length 1