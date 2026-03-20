import math
from dataclasses import dataclass, field


@dataclass
class TSPInstance:
	"""
	Representation of a traveling salesman problem instance.
	Stores the number of nodes, their coordinates, and distance information.
	"""
	n                : int
	vertex_names     : list[str] = None
	vertex_coords    : list[tuple[float, float]] = None
	edge_weight_type : str = None
	distance_matrix  : list[list[float]] = field(default_factory=list)


def read_list_int(line):
	"""Helper function to parse a line and extract integers."""
	return [int(x) for x in line.split()]


def euclidean_distance(p1, p2):
	"""
	Calculate the Euclidean distance between two 2D points.
	
	Args:
	    p1: Tuple (x1, y1) representing the first point
	    p2: Tuple (x2, y2) representing the second point
	
	Returns:
	    Float representing the Euclidean distance
	"""
	return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_distance_matrix(instance):
	"""
	Compute the distance matrix for all pairs of vertices.
	Applied rounding is determined by the edge_weight_type field.
	
	Args:
	    instance: TSPInstance object with vertex coordinates set
	
	Returns:
	    None (modifies instance.distance_matrix in-place)
	"""
	n = instance.n
	instance.distance_matrix = [[0.0] * n for _ in range(n)]
	
	for i in range(n):
		for j in range(i + 1, n):
			distance = euclidean_distance(
				instance.vertex_coords[i], 
				instance.vertex_coords[j]
			)
			
			# Apply rounding based on edge weight type
			if instance.edge_weight_type == "CEIL_2D":
				distance = math.ceil(distance)
			else:  # Default to EUC_2D: round to nearest integer
				distance = round(distance)
			
			instance.distance_matrix[i][j] = distance
			instance.distance_matrix[j][i] = distance


def read_tsplib_from_file(fname):
	"""
	Parse a TSPLIB format file and populate a TSPInstance.
	
	The TSPLIB format contains:
	- Header fields with metadata (NAME, DIMENSION, EDGE_WEIGHT_TYPE, etc.)
	- NODE_COORD_SECTION with vertex coordinates in format: index x y
	
	Args:
	    fname: Path to the TSPLIB .tsp file
	
	Returns:
	    TSPInstance object populated with data from the file
	"""
	lines = open(fname).readlines()
	
	instance = TSPInstance(n=0)
	vertex_coords_dict = {}
	in_coord_section = False
	
	for line in lines:
		line = line.strip()
		
		# Skip empty lines
		if not line:
			continue
		
		# Parse header fields (key: value format)
		if ":" in line:
			key, _, value = line.partition(":")
			key = key.strip().upper()
			value = value.strip()
			
			if key == "DIMENSION":
				instance.n = int(value)
				instance.vertex_names = [""] * instance.n
				instance.vertex_coords = [(math.inf, math.inf)] * instance.n
			elif key == "EDGE_WEIGHT_TYPE":
				instance.edge_weight_type = value
		
		# Start reading node coordinates
		elif line == "NODE_COORD_SECTION":
			in_coord_section = True
		
		# End of coordinate section
		elif line in ("EOF", "TOUR_SECTION"):
			break
		
		# Parse vertex coordinates
		elif in_coord_section:
			parts = line.split()
			if len(parts) >= 3 and parts[0].isnumeric():
				vertex_id = int(parts[0])
				x = float(parts[1])
				y = float(parts[2])
				
				if 0 < vertex_id <= instance.n:
					# Store with 1-based index as key, convert to 0-based later
					vertex_coords_dict[vertex_id] = (x, y)
	
	# Convert 1-based indices to 0-based and populate coordinates
	for vertex_id in range(1, instance.n + 1):
		if vertex_id in vertex_coords_dict:
			instance.vertex_coords[vertex_id - 1] = vertex_coords_dict[vertex_id]
	
	# Compute the distance matrix
	compute_distance_matrix(instance)
	
	return instance


def tour_cost(tour, instance):
	"""
	Calculate the total cost (length) of a tour.
	
	Args:
	    tour: List of vertex indices (0-based) representing a complete tour
	    instance: TSPInstance object with precomputed distance matrix
	
	Returns:
	    Float representing the total tour cost
	"""
	total_cost = 0.0
	n = len(tour)
	
	for i in range(n):
		current_vertex = tour[i]
		next_vertex = tour[(i + 1) % n]
		total_cost += instance.distance_matrix[current_vertex][next_vertex]
	
	return total_cost


def delta_cost_2opt(tour, instance, i, j):
	"""
	Calculate the change in cost when performing a 2-opt move.
	A 2-opt move reverses the tour segment between positions i and j.
	
	Args:
	    tour: List of vertex indices representing a tour
	    instance: TSPInstance object with distance matrix
	    i: First position (0-based, inclusive)
	    j: Second position (0-based, inclusive), must satisfy i < j
	
	Returns:
	    Float representing the change in cost (negative means improvement)
	"""
	n = len(tour)
	
	# Vertices at the edges of the segment to reverse
	vertex_a = tour[i]
	vertex_b = tour[(i + 1) % n]
	vertex_c = tour[j]
	vertex_d = tour[(j + 1) % n]
	
	# Cost of edges being removed
	current_cost = instance.distance_matrix[vertex_a][vertex_b] + \
	               instance.distance_matrix[vertex_c][vertex_d]
	
	# Cost of edges after reversal
	new_cost = instance.distance_matrix[vertex_a][vertex_c] + \
	           instance.distance_matrix[vertex_b][vertex_d]
	
	return new_cost - current_cost


def delta_cost_vertex_switch(tour, instance, i, j):
	"""
	Calculate the change in cost when swapping two vertices in the tour.
	Vertices at positions i and j exchange their positions in the tour.
	
	Args:
	    tour: List of vertex indices representing a tour
	    instance: TSPInstance object with distance matrix
	    i: First position (0-based)
	    j: Second position (0-based)
	
	Returns:
	    Float representing the change in cost (negative means improvement)
	"""
	n = len(tour)
	
	# Get vertices and their neighbors
	vertex_i = tour[i]
	vertex_j = tour[j]
	vertex_prev_i = tour[(i - 1) % n]
	vertex_next_i = tour[(i + 1) % n]
	vertex_prev_j = tour[(j - 1) % n]
	vertex_next_j = tour[(j + 1) % n]
	
	# Check if vertices are adjacent
	is_adjacent = (abs(i - j) == 1) or (abs(i - j) == n - 1)
	
	if is_adjacent:
		# Adjacent vertices: fewer edges to recalculate
		current_cost = (instance.distance_matrix[vertex_prev_i][vertex_i] +
		                instance.distance_matrix[vertex_i][vertex_j] +
		                instance.distance_matrix[vertex_j][vertex_next_j])
		
		new_cost = (instance.distance_matrix[vertex_prev_i][vertex_j] +
		            instance.distance_matrix[vertex_j][vertex_i] +
		            instance.distance_matrix[vertex_i][vertex_next_j])
	else:
		# Non-adjacent vertices: more edges to recalculate
		current_cost = (instance.distance_matrix[vertex_prev_i][vertex_i] +
		                instance.distance_matrix[vertex_i][vertex_next_i] +
		                instance.distance_matrix[vertex_prev_j][vertex_j] +
		                instance.distance_matrix[vertex_j][vertex_next_j])
		
		new_cost = (instance.distance_matrix[vertex_prev_i][vertex_j] +
		            instance.distance_matrix[vertex_j][vertex_next_i] +
		            instance.distance_matrix[vertex_prev_j][vertex_i] +
		            instance.distance_matrix[vertex_i][vertex_next_j])
	
	return new_cost - current_cost


def delta_cost_or_opt(tour, instance, i, insert_pos):
	"""
	Calculate the change in cost when relocating a single vertex.
	The vertex at position i is removed and inserted at position insert_pos.
	This is Or-opt with segment length 1.
	
	Args:
	    tour: List of vertex indices representing a tour
	    instance: TSPInstance object with distance matrix
	    i: Position of the vertex to relocate (0-based)
	    insert_pos: Position where the vertex will be inserted (0-based)
	
	Returns:
	    Float representing the change in cost (negative means improvement)
	"""
	n = len(tour)
	
	# No change if inserting at the same position
	if i == insert_pos or i == (insert_pos - 1) % n:
		return 0.0
	
	# Get the vertex to move and its neighbors
	vertex_to_move = tour[i]
	vertex_prev_i = tour[(i - 1) % n]
	vertex_next_i = tour[(i + 1) % n]
	
	# Get neighbors at the insertion position
	vertex_prev_insert = tour[(insert_pos - 1) % n]
	vertex_next_insert = tour[insert_pos % n]
	
	# Cost of removing the vertex from position i
	removal_cost = (instance.distance_matrix[vertex_prev_i][vertex_to_move] +
	                instance.distance_matrix[vertex_to_move][vertex_next_i] -
	                instance.distance_matrix[vertex_prev_i][vertex_next_i])
	
	# Cost of inserting the vertex at position insert_pos
	insertion_cost = (instance.distance_matrix[vertex_prev_insert][vertex_to_move] +
	                  instance.distance_matrix[vertex_to_move][vertex_next_insert] -
	                  instance.distance_matrix[vertex_prev_insert][vertex_next_insert])
	
	return removal_cost + insertion_cost


if __name__ == "__main__":
	# Example usage
	tsp_file = "data/DB/bioalg-proj01-tsplib/berlin52.tsp"
	instance = read_tsplib_from_file(tsp_file)
	
	print(f"TSP Instance: {tsp_file}")
	print(f"Number of vertices: {instance.n}")
	print(f"Edge weight type: {instance.edge_weight_type}")
	print(f"Sample coordinates: {instance.vertex_coords[:3]}")