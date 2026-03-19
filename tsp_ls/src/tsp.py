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