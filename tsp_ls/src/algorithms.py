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

from .operators import vertex_switching, two_opt



def greedy_local_search():
    return None


# Naive approach :
# neighbor = two_opt(current)
# n_len = tour_length(neighbor, dist)

# optimized approach (with delta 2 opt) :

# for i in range(n - 1):
#     for j in range(i + 2, n):

#         if i == 0 and j == n - 1:
#             continue

#         delta = delta_2opt(current, dist, i, j)

#         if delta < 0:
#             apply_2opt(current, i, j)
#             current_len += delta
#             break


def simulated_annealing():
    return None





# optimized approach :
# i, j = sorted(random.sample(range(n), 2))

# delta = delta_2opt(current, dist, i, j)

# if delta < 0 or random.random() < math.exp(-delta / temp):
#     apply_2opt(current, i, j)
#     current_len += delta