# operators.py
import random
from .tsp import delta_cost_or_opt, tour_cost

### Neighborhood operators ###

def vertex_switching_neighbors(tour):
    """
    Yield all possible neighbors by swapping each pair of vertices.
    """
    n = len(tour)
    for a in range(n):
        for b in range(a + 1, n):
            neighbor = tour[:]
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
                continue  # avoid same tour
            neighbor = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
            yield neighbor

def or_opt_neighbors(tour, seg_len=1):
    """
    Yield all neighbors via Or-opt (relocate a segment of length seg_len).
    Uses a set for O(1) exclusion instead of O(n) list membership test.
    """
    n = len(tour)
    for i in range(n):
        # Indices of the segment to relocate (handles wrap-around)
        indices = set((i + k) % n for k in range(seg_len))
        segment = [tour[(i + k) % n] for k in range(seg_len)]
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
    if operator == "swap":
        return vertex_switching_neighbors(tour)
    elif operator == "2-opt":
        return two_opt_neighbors(tour)
    elif operator == "or-opt":
        return or_opt_neighbors(tour)
    else:
        raise ValueError(f"Unknown operator: {operator}")

### single neighbor generators ###

# Used, for example, to randomly generate a neighbor
# for the naive simulated annealing algorithm

def vertex_switching(tour, a=None, b=None):
    """
    Generates a new tour with the exchange of two vertices.
    """
    neighbor = tour[:]
    n = len(neighbor)
    # If a and b are not provided, choose a random pair
    if a is None or b is None:
        a, b = random.sample(range(n), 2)
    neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
    return neighbor

def two_opt(tour, i=None, j=None):
    """
    Generates a new tour with the 2-opt exchange applied.
    """
    neighbor = tour[:]
    n = len(neighbor)
    # If i and j are not provided, a valid random pair is chosen
    if i is None or j is None:
        i, j = sorted(random.sample(range(n), 2))
        while j - i < 2:
            i, j = sorted(random.sample(range(n), 2))
    neighbor[i:j] = neighbor[i:j][::-1]
    return neighbor

def or_opt(tour, seg_len=1):
    """
    Generates a random neighbor using Or-opt (shifts a segment of length seg_len).
    """
    n = len(tour)
    # Choose randomly the edge to move
    i = random.randint(0, n - 1)
    # Extracts the segment
    segment = [tour[(i + k) % n] for k in range(seg_len)]
    indices = set((i + k) % n for k in range(seg_len))
    rest = [tour[j] for j in range(n) if j not in indices]
    # Choose randomly the insertion position
    pos = random.randint(0, len(rest))
    # Create the new tour
    return rest[:pos] + segment + rest[pos:]

def random_neighbor(tour):
    op = random.choice(["swap", "2-opt", "or-opt"])
    if op == "swap":
        return vertex_switching(tour)
    elif op == "2-opt":
        return two_opt(tour)
    else:
        return or_opt(tour)

### move-based (delta cost) approach utilities ###

def generate_vertex_switching_moves(tour):
    n = len(tour)
    for i in range(n):
        for j in range(i + 1, n):
            yield (i, j)

def apply_vertex_switching(tour, move):
    i, j = move
    tour[i], tour[j] = tour[j], tour[i]

def generate_2opt_moves(tour):
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # avoid same tour
            yield (i, j)

def apply_2opt(tour, move):
    i, j = move
    tour[i:j+1] = tour[i:j+1][::-1]

def generate_or_opt_moves(tour):
    n = len(tour)
    for i in range(n):
        for insert_pos in range(n):
            if insert_pos == i or insert_pos == (i + 1) % n:
                continue  # no real change
            yield (i, insert_pos)

def apply_or_opt(tour, move):
    i, insert_pos = move
    node = tour[i]
    # remove the vertex
    del tour[i]
    # adjust the position if needed
    if insert_pos > i:
        insert_pos -= 1
    tour.insert(insert_pos, node)

# note:
# run first
# delta = delta_fn(tour, instance, move)
# then
# apply_fn(tour, move)

### other ###

def or_opt_neighbors_k(k):
    """Returns a neighbors function for Or-opt with segment length k."""
    def fn(tour):
        return or_opt_neighbors(tour, seg_len=k)
    fn.__name__ = f"or_opt_{k}"
    return fn

def or_opt_single_k(k):
    """Returns a random neighbor function for Or-opt with segment length k."""
    def fn(tour):
        return or_opt(tour, seg_len=k)
    fn.__name__ = f"or_opt_{k}"
    return fn

def or_opt_delta_fn_k(k):
    """
    Delta function for Or-opt (only seg_len=1 is analytically supported here;
    for k>1 we fall back to full recomputation via the naive approach).
    """
    if k == 1:
        def fn(tour, instance, move):
            i, insert_pos = move
            return delta_cost_or_opt(tour, instance, i, insert_pos)
    else:
        def fn(tour, instance, move):
            # Full recomputation fallback for seg_len > 1
            i, insert_pos = move
            node = tour[i]
            trial = tour[:]
            del trial[i]
            adjusted = insert_pos - 1 if insert_pos > i else insert_pos
            trial.insert(adjusted, node)
            return tour_cost(trial, instance) - tour_cost(tour, instance)
    fn.__name__ = f"delta_or_opt_{k}"
    return fn

def generate_random_move_factory(operator_name, seg_len=1):
    """Returns a function that generates a random move (as indices) for SA optimized."""
    def fn(tour):
        n = len(tour)
        if operator_name == "swap":
            i, j = random.sample(range(n), 2)
            return (min(i, j), max(i, j))
        elif operator_name == "2-opt":
            i, j = sorted(random.sample(range(n), 2))
            while j - i < 2:
                i, j = sorted(random.sample(range(n), 2))
            return (i, j)
        elif operator_name == "or-opt":
            i = random.randint(0, n - 1)
            insert_pos = random.randint(0, n - 1)
            while insert_pos == i or insert_pos == (i + 1) % n:
                insert_pos = random.randint(0, n - 1)
            return (i, insert_pos)
    fn.__name__ = f"random_move_{operator_name}"
    return fn



### references ###

# https://deepwiki.com/rciemi/simulated-annealing-tsp-py/2.2-neighborhood-generation-strategies
# https://tsp-basics.blogspot.com/2017/03/or-opt.html
