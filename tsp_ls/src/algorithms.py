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


