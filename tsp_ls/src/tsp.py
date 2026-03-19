def parse_tsplib(filepath):
    """Parse a TSPLIB .tsp file and return list of (x, y) coordinates."""
    coords = {}
    edge_weight_type = "EUC_2D"
    in_coord_section = False
 
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key, val = key.strip().upper(), val.strip()
                if key == "EDGE_WEIGHT_TYPE":
                    edge_weight_type = val
            elif line == "NODE_COORD_SECTION":
                in_coord_section = True
            elif line in ("EOF", "TOUR_SECTION"):
                break
            elif in_coord_section:
                parts = line.split()
                idx, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                coords[idx] = (x, y)
 
    # Build ordered list (0-indexed internally)
    nodes = [coords[i] for i in sorted(coords.keys())]
    return nodes, edge_weight_type

class Tour:
    def __init__(self, vertices):
        self.vertices = vertices # list of tour vertex indices

    def cost(self, distance_matrix):
        """Calculate the total cost of the tour."""
        total = 0.0
        n = len(self.vertices)
        for i in range(n):
            a = self.vertices
            b = self.vertices[(i + 1) % n]
            total += distance_matrix[a][b]
        return total
    def distances(self, a, b, distance_matrix):
        return distance_matrix[a][b] # distance between 2 vertices

def delta_2opt(tour, dist, i, j):
    """
    Compute the delta (cost change) for a 2-opt swap 
    between edges (i,i+1) and (j,j+1)
    Args:
        tour: list of vertex indices
        dist: distance_matrix
        i, j: indices
    Returns:
        float: delta (if negative -> improvement)
    """
    n = len(tour)
    a = tour[i - 1]
    b = tour[i]
    c = tour[j]
    d = tour[(j + 1) % n]

    # Current cost: distance(a,b) + distance(c,d)
    # Cost after swap: distance(a,c) + distance(b,d)
    return (
        - dist[a][b]
        - dist[c][d]
        + dist[a][c]
        + dist[b][d]
    )