import random
 
 
def random_tour(n):
    """
    Generate a random tour of n verttices (0-indexed).
    Returns a shuffled list of all vertices indices.
    """
    tour = list(range(n))
    random.shuffle(tour)
    return tour
 