"""
run.py — TSP Local Search Experiment Runner
============================================

Runs all algorithm configurations on every .tsp file in a given folder,
repeating each run N_RUNS times. Results are saved as a CSV.

Usage:
    python -m src.run                                        # default: DB/bioalg-proj01-tsplib/
    python -m src.run --tsp_dir path/to/tsp/files
    python -m src.run --tsp_dir path/to/tsp --runs 10       # fewer runs for testing
    python -m src.run --out results/my_results.csv
    python -m src.run --tsp_file berlin52                    # single instance (no extension needed)
    python -m src.run --tsp_file berlin52.tsp --runs 5      # single instance with fewer runs
"""

import argparse
import csv
import os
import random
import time
from statistics import mean, stdev

from .tsp import read_tsplib_from_file, read_tsplib_dimension, tour_cost
from .tsp import (
    delta_cost_2opt,
    delta_cost_vertex_switch,
    delta_cost_or_opt,
)
from .init import random_tour
from .algorithms import (
    greedy_local_search_naive_best_improvement,
    greedy_local_search_naive_first_improvement,
    greedy_local_search_optimized,
    simulated_annealing_naive,
    simulated_annealing_optimized,
    make_cooling_schedule,
)
from .operators import (
    vertex_switching_neighbors,
    two_opt_neighbors,
    or_opt_neighbors,
    vertex_switching,
    two_opt,
    or_opt,
    generate_vertex_switching_moves,
    generate_2opt_moves,
    generate_or_opt_moves,
    apply_vertex_switching,
    apply_2opt,
    apply_or_opt,
)

# ─── Constants ────────────────────────────────────────────────────────────────

N_RUNS = 30           # repetitions per (instance, config)
SA_MAX_ITER = 50_000  # max iterations for SA optimized
SA_MIN_T = 1e-3

# Initial temperature for SA: scaled from instance size
def auto_initial_T(n):
    return max(100.0, n * 10.0)


# ─── Helpers ──────────────────────────────────────────────────────────────────

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


# ─── Configuration registry ───────────────────────────────────────────────────

def build_configs():
    """
    Returns a list of experiment configurations.
    Each config is a dict with keys:
        label, factory
    where factory(instance) -> callable() -> (tour, cost)
    """
    configs = []

    # ── 1. Greedy LS naive best-improvement, various operators ────────────────
    for op_name, neighbors_fn in [
        ("swap",     vertex_switching_neighbors),
        ("2-opt",    two_opt_neighbors),
        ("or-opt-1", or_opt_neighbors_k(1)),
    ]:
        label = f"gls_naive_best__{op_name}"

        def make(nfn=neighbors_fn):
            def factory(instance):
                def run():
                    return greedy_local_search_naive_best_improvement(
                        init_solution=lambda: random_tour(instance.n),
                        fitness=lambda t: tour_cost(t, instance),
                        get_neighbors=nfn,
                        max_iter=500,
                    )
                return run
            return factory

        configs.append({"label": label, "factory": make()})

    # ── 2. Greedy LS naive first-improvement, various operators ───────────────
    for op_name, neighbors_fn in [
        ("swap",     vertex_switching_neighbors),
        ("2-opt",    two_opt_neighbors),
        ("or-opt-1", or_opt_neighbors_k(1)),
    ]:
        label = f"gls_naive_first__{op_name}"

        def make(nfn=neighbors_fn):
            def factory(instance):
                def run():
                    return greedy_local_search_naive_first_improvement(
                        init_solution=lambda: random_tour(instance.n),
                        fitness=lambda t: tour_cost(t, instance),
                        get_neighbors=nfn,
                        max_iter=500,
                    )
                return run
            return factory

        configs.append({"label": label, "factory": make()})

    # ── 3. SA naive — operator comparison (exp cooling, medium alpha) ─────────
    for op_name, rand_nbr_fn in [
        ("swap",     vertex_switching),
        ("2-opt",    two_opt),
        ("or-opt-1", or_opt_single_k(1)),
        ("or-opt-2", or_opt_single_k(2)),
        ("or-opt-3", or_opt_single_k(3)),
    ]:
        label = f"sa_naive_op_cmp__{op_name}__exp_medium"

        def make(fn=rand_nbr_fn):
            def factory(instance):
                T0 = auto_initial_T(instance.n)

                def run():
                    cooling = make_cooling_schedule("exponential", T0=T0, alpha=0.995)
                    return simulated_annealing_naive(
                        init_solution=lambda: random_tour(instance.n),
                        fitness=lambda t: tour_cost(t, instance),
                        initial_temp=T0,
                        min_temp=SA_MIN_T,
                        update_temp=cooling,
                        random_neighbor=fn,
                    )
                return run
            return factory

        configs.append({"label": label, "factory": make()})

    # ── 4. SA naive — exponential cooling, swap operator, 3 alpha values ──────
    for alpha_label, alpha_val in [
        ("high",   0.999),
        ("medium", 0.995),
        ("low",    0.99),
    ]:
        label = f"sa_naive_exp__{alpha_label}__swap"

        def make(av=alpha_val):
            def factory(instance):
                T0 = auto_initial_T(instance.n)

                def run():
                    cooling = make_cooling_schedule("exponential", T0=T0, alpha=av)
                    return simulated_annealing_naive(
                        init_solution=lambda: random_tour(instance.n),
                        fitness=lambda t: tour_cost(t, instance),
                        initial_temp=T0,
                        min_temp=SA_MIN_T,
                        update_temp=cooling,
                        random_neighbor=vertex_switching,
                    )
                return run
            return factory

        configs.append({"label": label, "factory": make()})

    # ── 5. SA naive — polynomial cooling, swap operator, 3 alpha values ───────
    for alpha_label, alpha_val in [
        ("high",   4.0),
        ("medium", 3.0),
        ("low",    2.0),
    ]:
        label = f"sa_naive_poly__{alpha_label}__swap"

        def make(av=alpha_val):
            def factory(instance):
                T0 = auto_initial_T(instance.n)

                def run():
                    cooling = make_cooling_schedule("polynomial", T0=T0, alpha=av)
                    return simulated_annealing_naive(
                        init_solution=lambda: random_tour(instance.n),
                        fitness=lambda t: tour_cost(t, instance),
                        initial_temp=T0,
                        min_temp=SA_MIN_T,
                        update_temp=cooling,
                        random_neighbor=vertex_switching,
                    )
                return run
            return factory

        configs.append({"label": label, "factory": make()})

    # ── 6. SA naive — logarithmic cooling, swap operator, 3 alpha values ──────
    for alpha_label, alpha_val in [
        ("high",   2.0),
        ("medium", 1.0),
        ("low",    0.5),
    ]:
        label = f"sa_naive_log__{alpha_label}__swap"

        def make(av=alpha_val):
            def factory(instance):
                T0 = auto_initial_T(instance.n)

                def run():
                    cooling = make_cooling_schedule("logarithmic", T0=T0, alpha=av)
                    return simulated_annealing_naive(
                        init_solution=lambda: random_tour(instance.n),
                        fitness=lambda t: tour_cost(t, instance),
                        initial_temp=T0,
                        min_temp=SA_MIN_T,
                        update_temp=cooling,
                        random_neighbor=vertex_switching,
                    )
                return run
            return factory

        configs.append({"label": label, "factory": make()})

    # ── 7. GLS optimized — 3 operators ────────────────────────────────────────
    for op_name, gen_moves, delta_fn, apply_fn in [
        ("swap",   generate_vertex_switching_moves, lambda t, inst, m: delta_cost_vertex_switch(t, inst, m[0], m[1]), apply_vertex_switching),
        ("2-opt",  generate_2opt_moves,             lambda t, inst, m: delta_cost_2opt(t, inst, m[0], m[1]),          apply_2opt),
        ("or-opt", generate_or_opt_moves,           or_opt_delta_fn_k(1),                                             apply_or_opt),
    ]:
        label = f"gls_opt__{op_name}"

        def make(gm=gen_moves, df=delta_fn, af=apply_fn):
            def factory(instance):
                def run():
                    tour = random_tour(instance.n)
                    return greedy_local_search_optimized(
                        tour=tour,
                        instance=instance,
                        generate_moves=gm,
                        delta_fn=df,
                        apply_fn=af,
                        strategy="first",
                    )
                return run
            return factory

        configs.append({"label": label, "factory": make()})

    # ── 8. SA optimized — 3 operators, exp medium cooling ─────────────────────
    for op_name, gen_rand_move, delta_fn, apply_fn in [
        ("swap",   generate_random_move_factory("swap"),   lambda t, inst, m: delta_cost_vertex_switch(t, inst, m[0], m[1]), apply_vertex_switching),
        ("2-opt",  generate_random_move_factory("2-opt"),  lambda t, inst, m: delta_cost_2opt(t, inst, m[0], m[1]),          apply_2opt),
        ("or-opt", generate_random_move_factory("or-opt"), or_opt_delta_fn_k(1),                                             apply_or_opt),
    ]:
        label = f"sa_opt__{op_name}__exp_medium"

        def make(grm=gen_rand_move, df=delta_fn, af=apply_fn):
            def factory(instance):
                T0 = auto_initial_T(instance.n)

                def run():
                    tour = random_tour(instance.n)
                    cooling = make_cooling_schedule("exponential", T0=T0, alpha=0.995)
                    return simulated_annealing_optimized(
                        tour=tour,
                        instance=instance,
                        generate_random_move=grm,
                        delta_fn=df,
                        apply_fn=af,
                        T=T0,
                        min_T=SA_MIN_T,
                        update_T=cooling,
                        max_iter=SA_MAX_ITER,
                    )
                return run
            return factory

        configs.append({"label": label, "factory": make()})

    return configs


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_experiments(tsp_dir, out_path, n_runs, max_nodes=None, tsp_file=None):
    import pandas as pd

    all_tsp_files = sorted([
        f for f in os.listdir(tsp_dir)
        if f.endswith(".tsp")
    ])

    if not all_tsp_files:
        print(f"No .tsp files found in {tsp_dir}")
        return

    # Filter to a single file if requested
    if tsp_file is not None:
        fname = os.path.basename(tsp_file)
        if not fname.endswith(".tsp"):
            fname += ".tsp"
        all_tsp_files = [f for f in all_tsp_files if f == fname]
        if not all_tsp_files:
            print(f"File not found in {tsp_dir}: {fname}")
            return

    # Pre-filter by max_nodes
    tsp_files = []
    for f in all_tsp_files:
        fpath = os.path.join(tsp_dir, f)
        try:
            n = read_tsplib_dimension(fpath)
        except Exception as e:
            print(f"  [SKIP] {f}: cannot read header ({e})")
            continue
        if max_nodes is not None and n > max_nodes:
            print(f"  [SKIP] {os.path.splitext(f)[0]}  (n={n} > max_nodes={max_nodes})")
            continue
        tsp_files.append(f)

    configs = build_configs()
    total = len(tsp_files) * len(configs)
    done = 0
    print(f"\n{len(tsp_files)} instances retained, {len(configs)} configs → {total} experiment(s)\n")

    rows = []

    for tsp_file in tsp_files:
        fpath = os.path.join(tsp_dir, tsp_file)
        instance_name = os.path.splitext(tsp_file)[0]

        try:
            instance = read_tsplib_from_file(fpath)
        except Exception as e:
            print(f"  [SKIP] {tsp_file}: {e}")
            continue

        print(f"\n{'='*60}")
        print(f"Instance: {instance_name}  (n={instance.n})")
        print(f"{'='*60}")

        for cfg in configs:
            label = cfg["label"]
            factory = cfg["factory"]
            run_fn = factory(instance)

            costs = []
            times = []

            for rep in range(n_runs):
                t_start = time.perf_counter()
                try:
                    _, cost = run_fn()
                except Exception as e:
                    print(f"    [ERROR] {label} rep {rep}: {e}")
                    cost = float("nan")
                t_end = time.perf_counter()

                costs.append(cost)
                times.append(t_end - t_start)

            valid_costs = [c for c in costs if not (c != c)]  # filter nan
            if valid_costs:
                best_cost = min(valid_costs)
                mean_cost = mean(valid_costs)
                std_cost  = stdev(valid_costs) if len(valid_costs) > 1 else 0.0
            else:
                best_cost = mean_cost = std_cost = float("nan")

            mean_time = mean(times)
            std_time  = stdev(times) if len(times) > 1 else 0.0

            rows.append({
                "instance":    instance_name,
                "n":           instance.n,
                "config":      label,
                "best_cost":   round(best_cost, 2),
                "mean_cost":   round(mean_cost, 2),
                "std_cost":    round(std_cost, 2),
                "mean_time_s": round(mean_time, 4),
                "std_time_s":  round(std_time, 4),
            })

            done += 1
            print(f"  [{done}/{total}] {label}")
            print(f"    cost: best={best_cost:.1f}  mean={mean_cost:.1f}  std={std_cost:.1f}")
            print(f"    time: mean={mean_time:.3f}s  std={std_time:.3f}s")

    # Save results
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n✓ Results saved to {out_path}")
    print(df.head(10).to_string())


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TSP Local Search Experiment Runner")
    parser.add_argument(
        "--tsp_dir",
        default="DB/bioalg-proj01-tsplib",
        help="Directory containing .tsp files (default: DB/bioalg-proj01-tsplib)",
    )
    parser.add_argument(
        "--tsp_file",
        default=None,
        help="Run on a single instance, e.g. berlin52 or berlin52.tsp",
    )
    parser.add_argument(
        "--out",
        default="results/results.csv",
        help="Output CSV path (default: results/results.csv)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=N_RUNS,
        help=f"Number of repetitions per config (default: {N_RUNS})",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=None,
        help="Skip instances with more nodes than this value (default: no limit)",
    )
    args = parser.parse_args()

    print(f"TSP dir  : {args.tsp_dir}")
    print(f"TSP file : {args.tsp_file if args.tsp_file else 'all'}")
    print(f"Output   : {args.out}")
    print(f"Runs     : {args.runs}")
    print(f"Max nodes: {args.max_nodes if args.max_nodes else 'no limit'}")

    run_experiments(
        tsp_dir=args.tsp_dir,
        out_path=args.out,
        n_runs=args.runs,
        max_nodes=args.max_nodes,
        tsp_file=args.tsp_file,
    )


if __name__ == "__main__":
    main()