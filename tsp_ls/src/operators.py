# operators.py
import random
# Neighborhood operators

def vertex_switching(tour):
    tour_copy = tour.copy()
    a, b = random.sample(range(len(tour_copy)), 2)
    tour_copy[a], tour_copy[b] = tour_copy[b], tour_copy[a]
    return tour_copy


def two_opt(tour):
    tour_copy = tour.copy()
    a, b = sorted(random.sample(range(len(tour_copy)), 2))
    tour_copy[a:b + 1] = reversed(tour_copy[a:b + 1])
    return tour_copy



# https://deepwiki.com/rciemi/simulated-annealing-tsp-py/2.2-neighborhood-generation-strategies