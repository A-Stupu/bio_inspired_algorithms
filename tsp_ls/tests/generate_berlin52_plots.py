# tests/generate_berlin52_plots.py
# ==================================
# Generates PNG visualizations for the berlin52 instance.
#
# Outputs in results/plots/:
#   - berlin52__compare__gls_vs_sa.png   : GLS 2-opt vs SA 2-opt side by side
#   - berlin52__multi_gls.png            : grid of GLS variants
#   - berlin52__multi_sa.png             : grid of SA variants
#   - berlin52__multi_all.png            : all configs (best run)
#
# Usage (from project root):
#     python -m tests.generate_berlin52_plots
#     python tests/generate_berlin52_plots.py

import sys
import os

# Allows running the script directly (python tests/generate_berlin52_plots.py)
# without having to use python -m tests.generate_berlin52_plots
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tsp import read_tsplib_from_file, tour_cost
from src.init import random_tour
from src.run import build_configs
from src.visualize import plot_compare, plot_multi_configs

# Config -------------------------------------------------------------------------

TSP_PATH  = "DB/bioalg-proj01-tsplib/berlin52.tsp"
OUT_DIR   = "results/plots"
N_RUNS    = 5       # runs per config to keep the best solution

# Configs to include in each figure
# (subsets of the labels defined in run.py::build_configs)
CONFIGS_GLS = [
    "gls_naive_best__swap",
    "gls_naive_best__2-opt",
    "gls_naive_first__swap",
    "gls_naive_first__2-opt",
    "gls_opt__swap",
    "gls_opt__2-opt",
    "gls_opt__or-opt",
]

CONFIGS_SA = [
    "sa_naive_op_cmp__swap__exp_medium",
    "sa_naive_op_cmp__2-opt__exp_medium",
    "sa_naive_op_cmp__or-opt-1__exp_medium",
    "sa_naive_exp__high__swap",
    "sa_naive_exp__medium__swap",
    "sa_naive_exp__low__swap",
    "sa_opt__swap__exp_medium",
    "sa_opt__2-opt__exp_medium",
    "sa_opt__or-opt__exp_medium",
]

# Configs for side-by-side comparison
CONFIG_LEFT  = "gls_opt__2-opt"
CONFIG_RIGHT = "sa_opt__2-opt__exp_medium"

# Helpers -------------------------------------------------------------------------

def best_of(run_fn, n_runs):
    """Runs run_fn n_runs times and returns the best (tour, cost)."""
    best_tour, best_cost = None, float("inf")
    for _ in range(n_runs):
        try:
            tour, cost = run_fn()
            if cost < best_cost:
                best_tour, best_cost = tour[:], cost
        except Exception as e:
            print(f"    [ERROR] {e}")
    return best_tour, best_cost

def run_configs(cfg_map, labels, instance, n_runs):
    """
    Runs a list of configs on the instance and returns
    [(label, tour, cost), ...] (missing or failed configs are ignored).
    """
    results = []
    for label in labels:
        if label not in cfg_map:
            print(f"  [SKIP] Unknown config: {label}")
            continue
        run_fn = cfg_map[label]["factory"](instance)
        tour, cost = best_of(run_fn, n_runs)
        if tour is not None:
            results.append((label, tour, cost))
            print(f"  ✓  {label:<45}  cost = {cost:,.0f}")
        else:
            print(f"  ✗  {label}  — all runs failed")
    return results

# Main --------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load the instance
    print(f"Loading {TSP_PATH} ...")
    instance = read_tsplib_from_file(TSP_PATH)
    instance.name = "berlin52"
    print(f"  n = {instance.n} vertices\n")

    # Build the config registry
    all_cfgs  = build_configs()
    cfg_map   = {c["label"]: c for c in all_cfgs}

    # 1. GLS Grid -----------------------------------------------------------------
    print("=== GLS Grid ===")
    gls_results = run_configs(cfg_map, CONFIGS_GLS, instance, N_RUNS)
    if gls_results:
        out = os.path.join(OUT_DIR, "berlin52__multi_gls.png")
        plot_multi_configs(gls_results, instance, cols=3, save_path=out)
    print()

    # 2. SA Grid ------------------------------------------------------------------
    print("=== SA Grid ===")
    sa_results = run_configs(cfg_map, CONFIGS_SA, instance, N_RUNS)
    if sa_results:
        out = os.path.join(OUT_DIR, "berlin52__multi_sa.png")
        plot_multi_configs(sa_results, instance, cols=3, save_path=out)
    print()

    # 3. All Configs Grid ------------------------------------------------------------
    print("=== All Configs Grid ===")
    all_results = run_configs(cfg_map, list(cfg_map.keys()), instance, N_RUNS)
    if all_results:
        out = os.path.join(OUT_DIR, "berlin52__multi_all.png")
        plot_multi_configs(all_results, instance, cols=4, save_path=out)
    print()

    # 4. Side-by-Side Comparison: GLS 2-opt vs SA 2-opt ----------------------------
    print("=== Side-by-Side Comparison ===")
    results_pair = run_configs(cfg_map, [CONFIG_LEFT, CONFIG_RIGHT], instance, N_RUNS)
    if len(results_pair) == 2:
        _, tour_left,  _ = results_pair[0]
        _, tour_right, _ = results_pair[1]
        out = os.path.join(OUT_DIR, "berlin52__compare__gls_vs_sa.png")
        plot_compare(
            tour_left, tour_right, instance,
            label_a=CONFIG_LEFT,
            label_b=CONFIG_RIGHT,
            save_path=out,
        )
    print()

    print(f"All PNGs are in: {OUT_DIR}/")

if __name__ == "__main__":
    main()
