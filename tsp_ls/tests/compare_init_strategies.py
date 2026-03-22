# tests/compare_init_strategies.py
# ==================================
# Compares the three initialization strategies (random, nearest_neighbor,
# greedy_edge) on the berlin52 instance, for GLS 2-opt and SA 2-opt.
#
# Outputs in results/plots/:
#   - berlin52__init_cmp__costs.png        : initial cost vs final cost per init
#   - berlin52__init_cmp__improvement.png  : % improvement by algorithm
#   - berlin52__init_cmp__tours.png        : grid of best tours per init
#
# Usage (from project root):
#     python -m tests.compare_init_strategies
#     python tests/compare_init_strategies.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.tsp import read_tsplib_from_file, tour_cost, delta_cost_2opt
from src.init import INIT_STRATEGIES
from src.operators import generate_2opt_moves, apply_2opt, generate_random_move_factory
from src.algorithms import (
    greedy_local_search_optimized,
    simulated_annealing_optimized,
    make_cooling_schedule,
)

# Config -----------------------------------------------------------------------

TSP_PATH = "DB/bioalg-proj01-tsplib/berlin52.tsp"
OUT_DIR  = "results/plots"
N_RUNS   = 30     # runs per (init, algo) — 30 required by the subject

SA_MAX_ITER = 50_000
SA_MIN_T    = 1e-3

ALGOS = ["GLS 2-opt", "SA 2-opt"]
INITS = list(INIT_STRATEGIES.keys())   # ["random", "nearest_neighbor", "greedy_edge"]

# Palette: one color per initialization strategy
COLORS = {
    "random":           "#78909C",
    "nearest_neighbor": "#42A5F5",
    "greedy_edge":      "#66BB6A",
}

INIT_LABELS_PRETTY = {
    "random":           "Random",
    "nearest_neighbor": "Nearest Neighbor",
    "greedy_edge":      "Greedy Edge",
}

# Runners ----------------------------------------------------------------------

def run_gls(instance, init_name):
    """One GLS 2-opt first-improvement run from init_name. Returns (init_cost, final_cost, tour)."""
    tour = INIT_STRATEGIES[init_name](instance)
    init_cost = tour_cost(tour, instance)
    final_tour, final_cost = greedy_local_search_optimized(
        tour=tour,
        instance=instance,
        generate_moves=generate_2opt_moves,
        delta_fn=lambda t, inst, m: delta_cost_2opt(t, inst, m[0], m[1]),
        apply_fn=apply_2opt,
        strategy="first",
    )
    return init_cost, final_cost, final_tour

def run_sa(instance, init_name):
    """One SA 2-opt run from init_name. Returns (init_cost, final_cost, tour)."""
    n = instance.n
    T0 = max(100.0, n * 10.0)
    tour = INIT_STRATEGIES[init_name](instance)
    init_cost = tour_cost(tour, instance)
    cooling = make_cooling_schedule("exponential", T0=T0, alpha=0.995)
    rand_move = generate_random_move_factory("2-opt")
    final_tour, final_cost = simulated_annealing_optimized(
        tour=tour,
        instance=instance,
        generate_random_move=rand_move,
        delta_fn=lambda t, inst, m: delta_cost_2opt(t, inst, m[0], m[1]),
        apply_fn=apply_2opt,
        T=T0,
        min_T=SA_MIN_T,
        update_T=cooling,
        max_iter=SA_MAX_ITER,
    )
    return init_cost, final_cost, final_tour

RUNNERS = {
    "GLS 2-opt": run_gls,
    "SA 2-opt":  run_sa,
}

# Collect Results --------------------------------------------------------------

def collect(instance, n_runs):
    """
    Runs n_runs times each (algo, init) combination.
    Returns a dict:
      results[algo][init] = {
          "init_costs":  [float, ...],
          "final_costs": [float, ...],
          "best_tour":   list,
          "best_cost":   float,
      }
    """
    results = {algo: {init: {"init_costs": [], "final_costs": [], "best_tour": None, "best_cost": float("inf")}
                      for init in INITS}
               for algo in ALGOS}

    total = len(ALGOS) * len(INITS) * n_runs
    done  = 0

    for algo in ALGOS:
        runner = RUNNERS[algo]
        for init in INITS:
            for _ in range(n_runs):
                init_cost, final_cost, tour = runner(instance, init)
                results[algo][init]["init_costs"].append(init_cost)
                results[algo][init]["final_costs"].append(final_cost)
                if final_cost < results[algo][init]["best_cost"]:
                    results[algo][init]["best_cost"] = final_cost
                    results[algo][init]["best_tour"] = tour[:]
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  {done}/{total}", end="\r", flush=True)

    print()
    return results

# Figures -----------------------------------------------------------------------

def fig_costs(results, instance_name, save_path):
    """
    Grouped bars: average initial cost and average final cost
    for each (algo, init) combination.
    """
    fig, axes = plt.subplots(1, len(ALGOS), figsize=(14, 5), sharey=False)

    for ax, algo in zip(axes, ALGOS):
        x      = np.arange(len(INITS))
        width  = 0.32
        inits_avg_init  = [mean(results[algo][i]["init_costs"])  for i in INITS]
        inits_avg_final = [mean(results[algo][i]["final_costs"]) for i in INITS]
        inits_std_final = [stdev(results[algo][i]["final_costs"]) if len(results[algo][i]["final_costs"]) > 1 else 0
                           for i in INITS]

        bars_init  = ax.bar(x - width / 2, inits_avg_init,  width, label="Initial Cost",
                            color=[COLORS[i] for i in INITS], alpha=0.4, edgecolor="black", linewidth=0.6)
        bars_final = ax.bar(x + width / 2, inits_avg_final, width, label="Final Cost",
                            color=[COLORS[i] for i in INITS], alpha=0.9, edgecolor="black", linewidth=0.6,
                            yerr=inits_std_final, capsize=4, error_kw={"linewidth": 1})

        ax.set_xticks(x)
        ax.set_xticklabels([INIT_LABELS_PRETTY[i] for i in INITS], fontsize=9)
        ax.set_ylabel("Tour Cost")
        ax.set_title(algo, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Annotate final cost
        for bar, val in zip(bars_final, inits_avg_final):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{val:,.0f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle(f"{instance_name} — Initial vs Final Cost by Initialization Strategy",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path}")

def fig_improvement(results, instance_name, save_path):
    """
    Bars: average % improvement by algorithm
    = (initial_cost - final_cost) / initial_cost * 100
    """
    fig, axes = plt.subplots(1, len(ALGOS), figsize=(12, 4.5), sharey=True)

    for ax, algo in zip(axes, ALGOS):
        x     = np.arange(len(INITS))
        width = 0.5
        improvements = []
        for i in INITS:
            avg_init  = mean(results[algo][i]["init_costs"])
            avg_final = mean(results[algo][i]["final_costs"])
            pct = 100 * (avg_init - avg_final) / avg_init if avg_init > 0 else 0
            improvements.append(pct)

        bars = ax.bar(x, improvements, width,
                      color=[COLORS[i] for i in INITS],
                      edgecolor="black", linewidth=0.6, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([INIT_LABELS_PRETTY[i] for i in INITS], fontsize=9)
        ax.set_ylabel("Improvement (%)")
        ax.set_title(algo, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(improvements) * 1.25 if improvements else 10)

        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # Color legend
    patches = [mpatches.Patch(color=COLORS[i], label=INIT_LABELS_PRETTY[i]) for i in INITS]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(f"{instance_name} — % Improvement by Algorithm\n"
                 f"by Initialization Strategy",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path}")

def fig_tours(results, instance, save_path):
    """
    Grid: best tour found for each (algo, init).
    Rows = algos, columns = inits.
    """
    n_rows = len(ALGOS)
    n_cols = len(INITS)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5.5, n_rows * 4.5))

    xs = np.array([c[0] for c in instance.vertex_coords])
    ys = np.array([c[1] for c in instance.vertex_coords])

    for row, algo in enumerate(ALGOS):
        for col, init in enumerate(INITS):
            ax = axes[row][col]
            tour = results[algo][init]["best_tour"]
            cost = results[algo][init]["best_cost"]
            color = COLORS[init]

            if tour is not None:
                n = len(tour)
                for k in range(n):
                    a, b = tour[k], tour[(k + 1) % n]
                    ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                            color=color, alpha=0.75, linewidth=1.1)
                ax.scatter(xs, ys, c="black", s=14, zorder=5)
                # Starting vertex in gold
                ax.scatter([xs[tour[0]]], [ys[tour[0]]], c="gold", s=70,
                           zorder=10, edgecolors="black", linewidths=0.7)

            title = f"{algo}\n{INIT_LABELS_PRETTY[init]}\ncost={cost:,.0f}"
            ax.set_title(title, fontsize=8.5, pad=4)
            ax.axis("off")

    fig.suptitle(f"{instance.name} — Best Tours by (Algorithm × Initialization)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path}")

def print_summary(results):
    """Prints a summary table in the terminal."""
    col_w = 20
    header = f"{'Init':<{col_w}}" + "".join(
        f"{'Init avg':>12}{'Final avg':>12}{'Std':>8}{'Best':>10}{'Imp%':>8}"
        for _ in ALGOS
    )
    algo_header = f"{'':<{col_w}}" + "".join(
        f"  {'── ' + algo + ' ──':^48}" for algo in ALGOS
    )

    print("\n" + "═" * (col_w + 40 * len(ALGOS)))
    print(algo_header)
    print("─" * (col_w + 40 * len(ALGOS)))

    for init in INITS:
        row = f"{INIT_LABELS_PRETTY[init]:<{col_w}}"
        for algo in ALGOS:
            d = results[algo][init]
            avg_init  = mean(d["init_costs"])
            avg_final = mean(d["final_costs"])
            std_final = stdev(d["final_costs"]) if len(d["final_costs"]) > 1 else 0
            best      = d["best_cost"]
            pct       = 100 * (avg_init - avg_final) / avg_init if avg_init > 0 else 0
            row += f"{avg_init:>12.0f}{avg_final:>12.0f}{std_final:>8.0f}{best:>10.0f}{pct:>7.1f}%"
        print(row)

    print("═" * (col_w + 40 * len(ALGOS)) + "\n")

# Main --------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {TSP_PATH} ...")
    instance = read_tsplib_from_file(TSP_PATH)
    instance.name = "berlin52"
    print(f"  n = {instance.n} vertices\n")

    print(f"Running: {N_RUNS} runs × {len(ALGOS)} algos × {len(INITS)} inits "
          f"= {N_RUNS * len(ALGOS) * len(INITS)} runs total\n")

    results = collect(instance, N_RUNS)

    print_summary(results)

    print("Generating figures ...")
    fig_costs(
        results, instance.name,
        save_path=os.path.join(OUT_DIR, "berlin52__init_cmp__costs.png"),
    )
    fig_improvement(
        results, instance.name,
        save_path=os.path.join(OUT_DIR, "berlin52__init_cmp__improvement.png"),
    )
    fig_tours(
        results, instance,
        save_path=os.path.join(OUT_DIR, "berlin52__init_cmp__tours.png"),
    )

    print(f"\n✓ Figures saved in: {OUT_DIR}/")

if __name__ == "__main__":
    main()
