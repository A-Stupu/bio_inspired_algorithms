# operators.py
import random

### Neighborhood operators ###

def vertex_switching_neighbors(tour):
    """
    Yielf all possible neighbors by swapping each pair of vertices.
    """
    n = len(tour)

    for a in range(n):
        for b in range(a + 1, n):
            neighbor = tour.copy()
            neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
            yield neighbor


def two_opt_neighbors(tour):
    """
    Yield all possible neighbors by making 2-opt exchanges.
    """
    n = len(tour)

    for i in range(n - 1):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            neighbor = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
            yield neighbor


def or_opt_neighbors(tour, seg_len=1):
    """
    Yield all neighbors via Or-opt (relocate a segment of length seg_len).
    """
    n = len(tour)

    for i in range(n):
        segment = [tour[(i + k) % n] for k in range(seg_len)]
        indices = [(i + k) % n for k in range(seg_len)]
        rest = [tour[j] for j in range(n) if j not in indices]
        
        # Insert segment at every position in rest
        for pos in range(len(rest) + 1):
            neighbor = rest[:pos] + segment + rest[pos:]
            if neighbor != tour:
                yield neighbor

def get_neighbors(tour, operator):
    """
    Dispatch to the right neighbor generator.
    """
    if operator == "vertex switching":
        return vertex_switching_neighbors(tour)
    elif operator == "2-opt":
        return two_opt_neighbors(tour)
    elif operator == "or-opt":
        return or_opt_neighbors(tour)
    else:
        raise ValueError(f"Unknown operator: {operator}")



### single neighbor generators ###

# Used Used, for example, to randomly generate a neighbor 
# for the naive simulated annealing algorithm

def vertex_switching(tour, a=None, b=None):
    """
    Generates a new tour with the exchange of two vertices
    """
    neighbor = tour.copy()
    n=len(neighbor)

    # If a and b are not provided, choose a random pair
    if a is None or b is None:
        a, b = random.sample(range(n), 2)

    neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
    return neighbor


def two_opt(tour, i=None, j=None):
    """
    Generates a new tour with the 2-opt exchange applied
    """
    neighbor = tour.copy()
    n = len(neighbor)

    # If i and j are not provided, a valid random pair is chosen
    if i is None or j is None:
        i, j = sorted(random.sample(range(n), 2))
        if j - i < 2:
            j = (i+2) % n

    neighbor[i:j] = neighbor[i:j][::-1]
    return neighbor

def or_opt(tour, seg_len=1):
    """
    Generates a random neighbor using Or-opt (shifts a segment of length seg_len).
    """
    n = len(tour)
    # Choose randomly the edge to move
    i = random.randint(0, n - 1)
    # Extracts the egde
    segment = []
    for k in range(seg_len):
        segment.append(tour[(i + k) % n])

    # Creates a copy of the tour without the edge
    rest = [tour[j] for j in range(n) if j % n not in range(i, i + seg_len)]
    # Choose randomly the insertion position
    pos = random.randint(0, len(rest))
    # Create the new tour
    neighbor = rest[:pos] + segment + rest[pos:]
    return neighbor


### Utilities for optimized (delta cost) algorithms ###

def random_2opt_move(n):
    """
    Generates a pair of random indices (i, j) for a 2-opt exchange
    """
    while True:
        i, j = sorted(random.sample(range(n), 2))
        if j - i >= 2 and not (i == 0 and j == n-1):
            break
    return i, j

def apply_2opt(tour, i, j):
    """
    In place 2-opt exchange
    """
    tour[i:j+1] = tour[i:j+1][::-1]


### references ###

# https://deepwiki.com/rciemi/simulated-annealing-tsp-py/2.2-neighborhood-generation-strategies
# https://tsp-basics.blogspot.com/2017/03/or-opt.html