"""
Genetic Programming evolutionary loop for symbolic regression.

Entry point: `run_gp(data, config)` → best Node found.
"""
import random
import math
import time
from dataclasses import dataclass, field

from src.tree import Node, ramped_half_and_half
from src.fitness import fitness, mse
from src.operators import (
    subtree_mutation, point_mutation, hoist_mutation,
    constant_folding, subtree_crossover,
)
from src.selection import (
    tournament_select_pair, elitist_survivor, over_selection,
)


# -- Configuration -------------------------------------------------------------

@dataclass
class GPConfig:
    # Population
    pop_size:           int   = 200
    max_generations:    int   = 300
    max_depth_init:     int   = 5
    max_depth:          int   = 8

    # Operator probabilities (must sum ≤ 1; remainder → reproduction)
    p_crossover:        float = 0.70
    p_subtree_mutation: float = 0.10
    p_point_mutation:   float = 0.10
    p_hoist_mutation:   float = 0.05

    # Selection
    tournament_k:       int   = 7
    elite_count:        int   = 5       # individuals preserved each generation

    # Fitness
    complexity_weight:  float = 0.005   # bloat penalty coefficient

    # Termination
    target_fitness:     float = 1e-6    # stop early if reached
    patience:           int   = 80      # generations without improvement

    # Constant range for initialisation
    const_range:        tuple = (-10.0, 10.0)

    # Verbosity  (0 = silent, 1 = summary per run, 2 = per-generation)
    verbose:            int   = 1

    # Restarts
    n_restarts:         int   = 3       # independent runs; best result kept


# -- Single run ----------------------------------------------------------------

def _single_run(data: list[tuple[float, float]], cfg: GPConfig,
                run_id: int = 0) -> tuple[Node, float]:
    """One independent GP run.  Returns (best_node, best_fitness)."""

    # -- Initialise population -------------------------------------------------
    population = [
        ramped_half_and_half(cfg.max_depth_init, cfg.const_range)
        for _ in range(cfg.pop_size)
    ]
    fitnesses = [
        fitness(ind, data, cfg.complexity_weight)
        for ind in population
    ]

    best_idx  = min(range(cfg.pop_size), key=lambda i: fitnesses[i])
    best_node = population[best_idx].clone()
    best_fit  = fitnesses[best_idx]

    no_improve = 0

    for gen in range(cfg.max_generations):
        # -- Early stop --------------------------------------------------------
        if best_fit <= cfg.target_fitness:
            break
        if no_improve >= cfg.patience:
            if cfg.verbose >= 2:
                print(f"  [run {run_id}] gen {gen}: patience exceeded — stopping")
            break

        # -- Produce children --------------------------------------------------
        children   = []
        child_fits = []

        while len(children) < cfg.pop_size:
            r = random.random()
            cumulative = 0.0

            cumulative += cfg.p_crossover
            if r < cumulative:
                p1, p2 = tournament_select_pair(population, fitnesses, cfg.tournament_k)
                c1, c2 = subtree_crossover(p1, p2, cfg.max_depth)
                for c in (c1, c2):
                    c = constant_folding(c)
                    children.append(c)
                    child_fits.append(fitness(c, data, cfg.complexity_weight))
                continue

            cumulative += cfg.p_subtree_mutation
            if r < cumulative:
                parent = tournament_select_pair(population, fitnesses, cfg.tournament_k)[0]
                c = subtree_mutation(parent, cfg.max_depth_init, cfg.const_range)
                c = constant_folding(c)
                children.append(c)
                child_fits.append(fitness(c, data, cfg.complexity_weight))
                continue

            cumulative += cfg.p_point_mutation
            if r < cumulative:
                parent = tournament_select_pair(population, fitnesses, cfg.tournament_k)[0]
                c = point_mutation(parent, cfg.const_range)
                children.append(c)
                child_fits.append(fitness(c, data, cfg.complexity_weight))
                continue

            cumulative += cfg.p_hoist_mutation
            if r < cumulative:
                parent = tournament_select_pair(population, fitnesses, cfg.tournament_k)[0]
                c = hoist_mutation(parent)
                children.append(c)
                child_fits.append(fitness(c, data, cfg.complexity_weight))
                continue

            # Reproduction (copy)
            parent = tournament_select_pair(population, fitnesses, cfg.tournament_k)[0]
            children.append(parent.clone())
            child_fits.append(fitness(parent, data, cfg.complexity_weight))

        # -- Survivor selection  (µ+λ with elitism) ----------------------------
        population, fitnesses = elitist_survivor(
            population, fitnesses,
            children[:cfg.pop_size], child_fits[:cfg.pop_size],
            cfg.pop_size,
        )

        # -- Track best --------------------------------------------------------
        gen_best_idx = min(range(cfg.pop_size), key=lambda i: fitnesses[i])
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit < best_fit - 1e-12:
            best_fit  = gen_best_fit
            best_node = population[gen_best_idx].clone()
            no_improve = 0
        else:
            no_improve += 1

        if cfg.verbose >= 2 and gen % 20 == 0:
            raw_mse = mse(best_node, data)
            print(f"  [run {run_id}] gen {gen:4d} | fit={best_fit:.6g} "
                  f"| mse={raw_mse:.6g} | size={best_node.size()}")

    return best_node, best_fit


# -- Multi-restart entry point -------------------------------------------------

def run_gp(data: list[tuple[float, float]],
           cfg: GPConfig | None = None) -> tuple[Node, float, dict]:
    """
    Run GP with optional restarts.

    Returns
    -------
    best_node  : Node — best expression found
    best_fit   : float — fitness of best node
    info       : dict  — run statistics
    """
    if cfg is None:
        cfg = GPConfig()

    overall_best_node = None
    overall_best_fit  = float('inf')

    start = time.time()

    for run in range(cfg.n_restarts):
        if cfg.verbose >= 1:
            print(f"  -- restart {run + 1}/{cfg.n_restarts} --")
        node, fit = _single_run(data, cfg, run_id=run + 1)
        if fit < overall_best_fit:
            overall_best_fit  = fit
            overall_best_node = node

        if cfg.verbose >= 1:
            raw_mse = mse(node, data)
            print(f"  → run {run+1} done | mse={raw_mse:.6g} | "
                  f"expr={node.to_string()}")

    elapsed = time.time() - start
    info = {
        'n_restarts':   cfg.n_restarts,
        'best_fitness': overall_best_fit,
        'best_mse':     mse(overall_best_node, data),
        'best_size':    overall_best_node.size(),
        'elapsed_sec':  elapsed,
    }

    return overall_best_node, overall_best_fit, info
