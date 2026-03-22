# tests/plot_fitness_evolution.py
# ==================================
# Plots the evolution of fitness (tour cost) per iteration for each configuration,
# on a given TSP instance.
#
# Two figures are produced in results/plots/:
#   - berlin52__fitness_gls.png  : all GLS configs overlaid
#   - berlin52__fitness_sa.png   : all SA configs overlaid
#
# Each curve = average over N_RUNS runs, with ±std band in transparency.
#
# Usage (from project root):
#     python -m tests.plot_fitness_evolution
#     python -m tests.plot_fitness_evolution --tsp_file kroA100 --runs 10
#     python tests/plot_fitness_evolution.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import random
import numpy as np
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.tsp import (
    read_tsplib_from_file, tour_cost,
    delta_cost_2opt, delta_cost_vertex_switch, delta_cost_or_opt,
)
from src.init import INIT_STRATEGIES
from src.operators import (
    generate_2opt_moves, apply_2opt,
    generate_vertex_switching_moves, apply_vertex_switching,
    generate_or_opt_moves, apply_or_opt,
    generate_random_move_factory,
)
from src.algorithms import make_cooling_schedule
from src.fitness_logger import gls_with_history, sa_with_history

# Parameters --------------------------------------------------------------------

DEFAULT_TSP   = "DB/bioalg-proj01-tsplib/berlin52.tsp"
OUT_DIR       = "results/plots"
N_RUNS        = 30
SA_MAX_ITER   = 2_000
SA_MIN_T      = 1e-3
SA_LOG_EVERY  = 1    # one point every iteration → 500 points max
N_INTERP      = 300  # interpolation points to align curves

SEARCH_DIRS = [
    "DB/bioalg-proj01-tsplib",
    "DB/bioalg-proj01-dimacs-10k",
    "DB/bioalg-proj01-hz",
]

# Config Registry -----------------------------------------------------------------

def build_gls_configs():
    """
    List of GLS configs to plot.
    Each entry: (label, runner_factory)
    runner_factory(instance) -> callable() -> (tour, cost, history)
    """
    configs = []

    for strategy in ["first", "best"]:
        for op_name, gen_moves, delta_fn, apply_fn in [
            ("swap",   generate_vertex_switching_moves,
                       lambda t, inst, m: delta_cost_vertex_switch(t, inst, m[0], m[1]),
                       apply_vertex_switching),
            ("2-opt",  generate_2opt_moves,
                       lambda t, inst, m: delta_cost_2opt(t, inst, m[0], m[1]),
                       apply_2opt),
            ("or-opt", generate_or_opt_moves,
                       lambda t, inst, m: delta_cost_or_opt(t, inst, m[0], m[1]),
                       apply_or_opt),
        ]:
            label = f"GLS {strategy} / {op_name}"

            def make(s=strategy, gm=gen_moves, df=delta_fn, af=apply_fn):
                def factory(instance):
                    def run():
                        tour = INIT_STRATEGIES["random"](instance)
                        return gls_with_history(
                            tour=tour, instance=instance,
                            generate_moves=gm, delta_fn=df, apply_fn=af,
                            strategy=s,
                            max_iter=500,
                        )
                    return run
                return factory

            configs.append({"label": label, "factory": make()})

    return configs   # 6 configs : 2 stratégies × 3 opérateurs

def build_sa_configs():
    """
    List of SA configs to plot.
    """
    configs = []

    for sched_label, method, alpha in [
        ("exp α=0.999", "exponential", 0.999),
        ("exp α=0.995", "exponential", 0.995),
        ("exp α=0.99",  "exponential", 0.99),
        ("poly α=2",    "polynomial",  2.0),
        ("poly α=1",    "polynomial",  1.0),
        ("log α=1",     "logarithmic", 1.0),
        ("log α=0.5",   "logarithmic", 0.5),
    ]:
        label = f"SA 2-opt / {sched_label}"

        def make(m=method, a=alpha):
            def factory(instance):
                T0 = max(100.0, instance.n * 10.0)
                rand_move = generate_random_move_factory("2-opt")
                def run():
                    tour = INIT_STRATEGIES["random"](instance)
                    cooling = make_cooling_schedule(m, T0=T0, alpha=a)
                    return sa_with_history(
                        tour=tour, instance=instance,
                        generate_random_move=rand_move,
                        delta_fn=lambda t, inst, mv: delta_cost_2opt(t, inst, mv[0], mv[1]),
                        apply_fn=apply_2opt,
                        T=T0, min_T=SA_MIN_T, update_T=cooling,
                        max_iter=SA_MAX_ITER, log_every=1,
                    )
                return run
            return factory

        configs.append({"label": label, "factory": make()})

    return configs   # 7 configs : 3 exp + 2 poly + 2 log

# Collect History -----------------------------------------------------------------

def collect_histories(configs, instance, n_runs):
    """
    Runs each config n_runs times and collects histories.

    Returns:
        list of {
            "label":     str,
            "histories": [ [(iter, cost), ...], ... ]   # one list per run
        }
    """
    results = []
    total   = len(configs) * n_runs
    done    = 0

    for cfg in configs:
        label    = cfg["label"]
        run_fn   = cfg["factory"](instance)
        histories = []

        for _ in range(n_runs):
            try:
                _, _, history = run_fn()
                histories.append(history)
            except Exception as e:
                print(f"\n  [ERROR] {label}: {e}")
            done += 1
            print(f"  {done}/{total} — {label}", end="\r", flush=True)

        results.append({"label": label, "histories": histories})

    print()
    return results

# Interpolation / Alignment --------------------------------------------------------

def interpolate_histories(histories, n_points):
    """
    Aligns histories of different lengths on a common grid
    of n_points points (piecewise linear interpolation, last value held
    after the last recorded point).

    Args:
        histories : list of lists [(iter, cost), ...]
        n_points  : number of points in the common grid

    Returns:
        x_common : array (n_points,)
        matrix   : array (n_runs, n_points) — interpolated cost for each run
    """
    # Common bound: max of last iterations
    x_max = max(h[-1][0] for h in histories if h)
    if x_max == 0:
        x_max = 1
    x_common = np.linspace(0, x_max, n_points)

    matrix = np.zeros((len(histories), n_points))

    for i, history in enumerate(histories):
        xs = np.array([p[0] for p in history], dtype=float)
        ys = np.array([p[1] for p in history], dtype=float)
        # np.interp holds the last value beyond the last x (right=ys[-1])
        matrix[i] = np.interp(x_common, xs, ys)

    return x_common, matrix

# Plotting -----------------------------------------------------------------------

def _make_colormap(n):
    """Generates n distinct colors from a matplotlib colormap."""
    cmap = cm.get_cmap("tab20" if n <= 20 else "hsv", n)
    return [cmap(i) for i in range(n)]

def plot_fitness_curves(all_results, instance_name, title, save_path, n_interp=N_INTERP):
    """
    Plots all fitness evolution curves on the same axis.
    Average ± std over runs, using a common iteration grid.

    Args:
        all_results : output of collect_histories()
        instance_name : str
        title       : plot title
        save_path   : PNG path
        n_interp    : interpolation grid resolution
    """
    n_configs = len(all_results)
    colors    = _make_colormap(n_configs)

    fig, ax = plt.subplots(figsize=(12, 6))

    for cfg_data, color in zip(all_results, colors):
        label     = cfg_data["label"]
        histories = cfg_data["histories"]

        if not histories:
            continue

        x_common, matrix = interpolate_histories(histories, n_interp)
        mean_curve = matrix.mean(axis=0)
        std_curve  = matrix.std(axis=0)

        # Average final cost (for legend sorting)
        final_cost = mean_curve[-1]

        ax.plot(x_common, mean_curve,
                color=color, linewidth=1.5,
                label=f"{label}  (final={final_cost:,.0f})")
        ax.fill_between(x_common,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        color=color, alpha=0.12)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Tour Cost", fontsize=11)
    ax.set_title(f"{title}\n{instance_name}  —  average ± std over {len(all_results[0]['histories'])} runs",
                 fontsize=11)
    ax.grid(True, alpha=0.25)

    # Legend sorted by increasing final cost
    handles, labels = ax.get_legend_handles_labels()
    # Extract final costs from labels for sorting
    def _final(lbl):
        try:
            return float(lbl.split("final=")[1].replace(",", "").replace(")", ""))
        except Exception:
            return 0.0
    sorted_pairs = sorted(zip(labels, handles), key=lambda p: _final(p[0]))
    sorted_labels, sorted_handles = zip(*sorted_pairs) if sorted_pairs else ([], [])
    ax.legend(sorted_handles, sorted_labels,
              loc="upper right", fontsize=7.5,
              framealpha=0.85, ncol=max(1, n_configs // 12))

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path}")

def plot_fitness_subplots(all_results, instance_name, title, save_path,
                          n_cols=3, n_interp=N_INTERP):
    """
    Alternative: one subplot per config (useful when curves overlap too much).
    Each subplot shows individual runs in light gray + average in color.
    """
    n_configs = len(all_results)
    n_rows    = (n_configs + n_cols - 1) // n_cols
    colors    = _make_colormap(n_configs)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5, n_rows * 3.5),
                             sharex=False, sharey=False)
    axes_flat = np.array(axes).flatten()

    for idx, (cfg_data, color) in enumerate(zip(all_results, colors)):
        ax        = axes_flat[idx]
        label     = cfg_data["label"]
        histories = cfg_data["histories"]

        if not histories:
            ax.set_visible(False)
            continue

        x_common, matrix = interpolate_histories(histories, n_interp)
        mean_curve = matrix.mean(axis=0)

        # Individual runs in light gray
        for run_row in matrix:
            ax.plot(x_common, run_row, color="grey", linewidth=0.5, alpha=0.3)

        # Average in color
        ax.plot(x_common, mean_curve, color=color, linewidth=1.8)

        ax.set_title(f"{label}\nfinal={mean_curve[-1]:,.0f}", fontsize=7.5, pad=3)
        ax.set_xlabel("Iteration", fontsize=7)
        ax.set_ylabel("Cost", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    for idx in range(n_configs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"{title}  —  {instance_name}", fontsize=11, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path}")

# Main --------------------------------------------------------------------------

def find_tsp(tsp_file):
    fname = tsp_file if tsp_file.endswith(".tsp") else tsp_file + ".tsp"
    for d in SEARCH_DIRS:
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            return p
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsp_file", default="berlin52")
    parser.add_argument("--runs",     type=int, default=N_RUNS)
    parser.add_argument("--out_dir",  default=OUT_DIR)
    parser.add_argument("--subplots", action="store_true",
                        help="Also generate the individual subplots version")
    args = parser.parse_args()

    tsp_path = find_tsp(args.tsp_file)
    if tsp_path is None:
        # Try direct path
        if os.path.isfile(args.tsp_file):
            tsp_path = args.tsp_file
        else:
            print(f"[ERROR] File not found: {args.tsp_file}")
            return

    instance_name = os.path.splitext(os.path.basename(tsp_path))[0]
    instance      = read_tsplib_from_file(tsp_path)
    instance.name = instance_name
    print(f"Instance: {instance_name}  (n={instance.n})\n")

    # GLS -------------------------------------------------------------------------
    gls_configs = build_gls_configs()
    print(f"=== GLS: {len(gls_configs)} configs × {args.runs} runs ===")
    gls_results = collect_histories(gls_configs, instance, args.runs)

    plot_fitness_curves(
        gls_results, instance_name,
        title="GLS optimized — fitness evolution",
        save_path=os.path.join(args.out_dir, f"{instance_name}__fitness_gls.png"),
    )
    if args.subplots:
        plot_fitness_subplots(
            gls_results, instance_name,
            title="GLS optimized — fitness evolution (detail)",
            save_path=os.path.join(args.out_dir, f"{instance_name}__fitness_gls_detail.png"),
        )

    # SA --------------------------------------------------------------------------
    sa_configs = build_sa_configs()
    print(f"\n=== SA: {len(sa_configs)} configs × {args.runs} runs ===")
    sa_results = collect_histories(sa_configs, instance, args.runs)

    plot_fitness_curves(
        sa_results, instance_name,
        title="SA optimized — fitness evolution",
        save_path=os.path.join(args.out_dir, f"{instance_name}__fitness_sa.png"),
    )
    if args.subplots:
        plot_fitness_subplots(
            sa_results, instance_name,
            title="SA optimized — fitness evolution (detail)",
            save_path=os.path.join(args.out_dir, f"{instance_name}__fitness_sa_detail.png"),
        )

    print(f"\n✓ Figures saved in: {args.out_dir}/")

if __name__ == "__main__":
    main()
