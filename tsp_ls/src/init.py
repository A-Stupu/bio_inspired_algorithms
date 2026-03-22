# init.py — Tour Initialization Strategies
# =========================================
#
# Three initialization strategies for TSP:
#
#   random_tour(n)                  — random permutation (baseline)
#   nearest_neighbor_tour(instance) — nearest neighbor heuristic
#   greedy_edge_tour(instance)      — construction by sorted edges
#
# All return a 0-based index list of length n,
# representing a valid Hamiltonian tour.

import random

# 1. Random --------------------------------------------------------------------

def random_tour(n):
    """
    Generates a random tour of n vertices (0-based indices).
    Serves as a baseline: no distance information is used.

    Complexity: O(n)
    """
    tour = list(range(n))
    random.shuffle(tour)
    return tour

# 2. Nearest Neighbor -------------------------------------------------------------

def nearest_neighbor_tour(instance, start=None):
    """
    Builds a tour using the nearest neighbor heuristic.

    Principle: From the current vertex, always go to the nearest unvisited vertex.
    Repeat until all vertices are visited, then return to the starting vertex.

    Args:
        instance: TSPInstance with distance_matrix and n
        start: index of the starting vertex (random if None)

    Returns:
        list of 0-based indices (Hamiltonian tour)

    Complexity: O(n²) — acceptable up to ~10,000 vertices
    Quality: typically 20-25% above optimal
    """
    n = instance.n
    dist = instance.distance_matrix

    if start is None:
        start = random.randint(0, n - 1)

    visited = [False] * n
    tour = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = tour[-1]
        best_dist = float("inf")
        best_vertex = -1

        for vertex in range(n):
            if not visited[vertex] and dist[current][vertex] < best_dist:
                best_dist = dist[current][vertex]
                best_vertex = vertex

        tour.append(best_vertex)
        visited[best_vertex] = True

    return tour

# 3. Greedy Edge -----------------------------------------------------------------

def greedy_edge_tour(instance):
    """
    Builds a tour by adding edges in increasing order of length,
    subject to two constraints:
      - each vertex has at most degree 2 (at most 2 edges in the tour)
      - no premature cycles (except to close the tour at the end)

    Principle:
      1. Sort all edges (i, j) by increasing distance
      2. Add an edge if it does not violate any constraint
      3. Continue until n edges are added (complete tour)

    Args:
        instance: TSPInstance with distance_matrix and n

    Returns:
        list of 0-based indices (Hamiltonian tour)

    Complexity: O(n² log n) for sorting + O(n²) for construction
    Quality: often better than Nearest Neighbor
    """
    n = instance.n
    dist = instance.distance_matrix

    # Sort all edges by increasing distance
    edges = sorted(
        ((dist[i][j], i, j) for i in range(n) for j in range(i + 1, n)),
        key=lambda e: e[0]
    )

    degree = [0] * n          # degree of each vertex in the partial tour
    adj = [[] for _ in range(n)]  # adjacency list of the partial tour

    # Union-Find to detect premature cycles
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    edges_added = 0

    for _, i, j in edges:
        if edges_added == n:
            break

        # Constraint 1: max degree 2
        if degree[i] >= 2 or degree[j] >= 2:
            continue

        # Constraint 2: no premature cycles
        # (except if it's the last edge to close the tour)
        if edges_added < n - 1 and find(i) == find(j):
            continue

        # Add the edge
        adj[i].append(j)
        adj[j].append(i)
        degree[i] += 1
        degree[j] += 1
        union(i, j)
        edges_added += 1

    # Reconstruct the tour from the adjacency list
    # Start from an endpoint (degree 1) or any vertex if the tour is closed
    tour = _reconstruct_tour(adj, n)

    return tour

def _reconstruct_tour(adj, n):
    """
    Reconstructs an ordered tour from an adjacency list
    where each vertex has exactly degree 2.

    In case of failure (incomplete graph, which should not happen),
    completes with missing vertices in order.
    """
    visited = [False] * n

    # Find a starting vertex (degree 2 = complete tour)
    start = 0
    tour = [start]
    visited[start] = True

    prev = -1
    current = start

    for _ in range(n - 1):
        neighbors = adj[current]
        moved = False
        for nxt in neighbors:
            if nxt != prev and not visited[nxt]:
                tour.append(nxt)
                visited[nxt] = True
                prev = current
                current = nxt
                moved = True
                break
        if not moved:
            break

    # Safety: add missing vertices if construction failed
    if len(tour) < n:
        missing = [v for v in range(n) if not visited[v]]
        tour.extend(missing)

    return tour

# Registry ----------------------------------------------------------------------

INIT_STRATEGIES = {
    "random":           lambda instance: random_tour(instance.n),
    "nearest_neighbor": lambda instance: nearest_neighbor_tour(instance),
    "greedy_edge":      lambda instance: greedy_edge_tour(instance),
}
# Dictionary {name -> callable(instance) -> tour}.
# Used by run.py and test scripts to iterate over strategies.

# Quick validation (callable directly) ------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from src.tsp import read_tsplib_from_file, tour_cost

    tsp_path = "DB/bioalg-proj01-tsplib/berlin52.tsp"
    instance = read_tsplib_from_file(tsp_path)
    instance.name = "berlin52"

    print(f"Instance: berlin52  (n={instance.n})\n")
    print(f"{'Strategy':<20}  {'Initial Cost':>14}")
    print("-" * 38)

    for name, fn in INIT_STRATEGIES.items():
        costs = [tour_cost(fn(instance), instance) for _ in range(10)]
        avg = sum(costs) / len(costs)
        best = min(costs)
        print(f"{name:<20}  avg={avg:>8.0f}  best={best:>8.0f}")
