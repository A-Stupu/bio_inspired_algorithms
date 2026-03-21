"""
visualize.py — TSP Solution Visualizer
=======================================

Functions to visualize TSP tours obtained by different algorithms.

Direct usage (CLI):
    python -m src.visualize --tsp_file berlin52 --configs gls_opt__2-opt sa_opt__2-opt__exp_medium
    python -m src.visualize --tsp_file berlin52 --all_configs
    python -m src.visualize --tsp_file berlin52 --configs gls_opt__2-opt --single

Python API:
    from src.visualize import plot_tour, plot_compare, plot_multi_configs
    from src.tsp import read_tsplib_from_file

    instance = read_tsplib_from_file("DB/bioalg-proj01-tsplib/berlin52.tsp")
    plot_tour(tour, instance, title="My Tour", save_path="tour.png")
    plot_compare(tour_a, tour_b, instance, label_a="GLS", label_b="SA")
    plot_multi_configs(results, instance)   # results = [(label, tour, cost), ...]
"""

import argparse
import os
import random
import math

import matplotlib
matplotlib.use("Agg")           # headless — save PNG without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Color palette ───────────────────────────────────────────────────────────

PALETTE = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#E91E63",  # pink
    "#795548",  # brown
]

# ─── Internal helpers ─────────────────────────────────────────────────────────

def _coords(instance):
    """Return xs, ys (numpy arrays) from instance coordinates."""
    xs = np.array([c[0] for c in instance.vertex_coords])
    ys = np.array([c[1] for c in instance.vertex_coords])
    return xs, ys

def _draw_tour(ax, tour, xs, ys, color, alpha=0.8, lw=1.2, label=None):
    """Draw the edges of a tour on a matplotlib axis."""
    n = len(tour)
    for k in range(n):
        a = tour[k]
        b = tour[(k + 1) % n]
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                color=color, alpha=alpha, linewidth=lw)
    # Legend via phantom line
    if label:
        ax.plot([], [], color=color, linewidth=lw, label=label)

def _scatter_cities(ax, xs, ys, color="black", zorder=5, s=18):
    """Display cities as points."""
    ax.scatter(xs, ys, c=color, s=s, zorder=zorder)

def _annotate_start(ax, tour, xs, ys):
    """Highlight the starting city (index 0 in the tour)."""
    start = tour[0]
    ax.scatter([xs[start]], [ys[start]], c="gold", s=80,
               zorder=10, edgecolors="black", linewidths=0.8)

def _style_ax(ax, title, instance_name, cost):
    """Apply common style to an axis."""
    ax.set_title(f"{title}\n{instance_name}  —  cost: {cost:,.0f}",
                 fontsize=10, pad=6)
    ax.axis("off")

def _save_or_show(fig, save_path, dpi=150):
    """Save the figure or display it."""
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)

# ─── Public API ───────────────────────────────────────────────────────────────

def plot_tour(tour, instance, title="TSP Tour", save_path=None, color="#2196F3"):
    """
    Visualize a single tour.

    Args:
        tour:        list of city indices (0-based)
        instance:    TSPInstance
        title:       plot title
        save_path:   output PNG path (None = interactive display)
        color:       edge color
    """
    from .tsp import tour_cost
    cost = tour_cost(tour, instance)
    xs, ys = _coords(instance)

    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_tour(ax, tour, xs, ys, color=color)
    _scatter_cities(ax, xs, ys)
    _annotate_start(ax, tour, xs, ys)
    _style_ax(ax, title, getattr(instance, "name", ""), cost)

    _save_or_show(fig, save_path)

def plot_compare(tour_a, tour_b, instance,
                 label_a="Tour A", label_b="Tour B",
                 save_path=None):
    """
    Compare two tours side by side.

    Args:
        tour_a, tour_b:  lists of city indices
        instance:        TSPInstance
        label_a, label_b: labels
        save_path:       output PNG path
    """
    from .tsp import tour_cost
    cost_a = tour_cost(tour_a, instance)
    cost_b = tour_cost(tour_b, instance)
    xs, ys = _coords(instance)
    inst_name = getattr(instance, "name", "")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, tour, cost, label, color in [
        (axes[0], tour_a, cost_a, label_a, PALETTE[0]),
        (axes[1], tour_b, cost_b, label_b, PALETTE[1]),
    ]:
        _draw_tour(ax, tour, xs, ys, color=color)
        _scatter_cities(ax, xs, ys)
        _annotate_start(ax, tour, xs, ys)
        _style_ax(ax, label, inst_name, cost)

    # Global title with delta
    delta = cost_b - cost_a
    sign = "+" if delta >= 0 else ""
    fig.suptitle(f"Comparison  |  Δcost {label_b} vs {label_a}: {sign}{delta:,.0f}",
                 fontsize=11, y=1.01)

    _save_or_show(fig, save_path)

def plot_multi_configs(results, instance, cols=3, save_path=None):
    """
    Display multiple tours (one per config) on a grid of subplots.

    Args:
        results:   list of tuples (label, tour, cost)
        instance:  TSPInstance
        cols:      number of columns in the grid
        save_path: output PNG path
    """
    n_plots = len(results)
    if n_plots == 0:
        print("No results to display.")
        return

    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 6, rows * 4.5))
    axes = np.array(axes).flatten()
    xs, ys = _coords(instance)
    inst_name = getattr(instance, "name", "")

    for idx, (label, tour, cost) in enumerate(results):
        ax = axes[idx]
        color = PALETTE[idx % len(PALETTE)]
        _draw_tour(ax, tour, xs, ys, color=color)
        _scatter_cities(ax, xs, ys)
        _annotate_start(ax, tour, xs, ys)
        _style_ax(ax, label, inst_name, cost)

    # Hide empty axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    # Global title: best config
    best_label, _, best_cost = min(results, key=lambda r: r[2])
    fig.suptitle(
        f"{inst_name}  —  {n_plots} configurations  |  "
        f"best: {best_label}  ({best_cost:,.0f})",
        fontsize=11, y=1.01
    )

    _save_or_show(fig, save_path)

def plot_cost_evolution(cost_history, label="SA", save_path=None, instance_name=""):
    """
    Display the evolution of cost over iterations (for instrumented SA).

    Args:
        cost_history: list of (iteration, cost)
        label:        config name
        save_path:    output PNG path
        instance_name: instance name
    """
    iters = [h[0] for h in cost_history]
    costs = [h[1] for h in cost_history]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(iters, costs, color=PALETTE[0], linewidth=1.2, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tour cost")
    ax.set_title(f"Cost evolution — {label}  ({instance_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)

# ─── CLI ───────────────────────────────────────────────────────────────────────

def _find_tsp_file(tsp_file, search_dirs):
    """Search for the .tsp file in available directories."""
    fname = tsp_file if tsp_file.endswith(".tsp") else tsp_file + ".tsp"
    for d in search_dirs:
        candidate = os.path.join(d, fname)
        if os.path.isfile(candidate):
            return candidate
    return None

def main():
    parser = argparse.ArgumentParser(description="TSP Tour Visualizer")
    parser.add_argument("--tsp_file", required=True,
                        help="Instance name (e.g., berlin52 or berlin52.tsp)")
    parser.add_argument("--tsp_dir", default=None,
                        help="Directory containing the .tsp (auto-detected if missing)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config labels to visualize (e.g., gls_opt__2-opt)")
    parser.add_argument("--all_configs", action="store_true",
                        help="Visualize all available configs")
    parser.add_argument("--single", action="store_true",
                        help="Single tour mode (first config only)")
    parser.add_argument("--compare", action="store_true",
                        help="Side-by-side comparison mode (2 configs required)")
    parser.add_argument("--out_dir", default="results/plots",
                        help="Output directory for PNGs (default: results/plots)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs to choose the best solution")
    args = parser.parse_args()

    # Local import (avoids circular import issues outside package)
    from .tsp import read_tsplib_from_file, tour_cost
    from .init import random_tour
    from .run import build_configs
    from .operators import (
        generate_vertex_switching_moves, generate_2opt_moves, generate_or_opt_moves,
        apply_vertex_switching, apply_2opt, apply_or_opt,
        generate_random_move_factory,
    )

    # ── Find the .tsp file ───────────────────────────────────────────────────────
    search_dirs = [
        args.tsp_dir or "",
        "DB/bioalg-proj01-tsplib",
        "DB/bioalg-proj01-dimacs-10k",
        "DB/bioalg-proj01-hz",
    ]
    tsp_path = _find_tsp_file(args.tsp_file, search_dirs)
    if tsp_path is None:
        print(f"[ERROR] File not found: {args.tsp_file}")
        return

    instance_name = os.path.splitext(os.path.basename(tsp_path))[0]
    instance = read_tsplib_from_file(tsp_path)
    instance.name = instance_name
    print(f"Instance loaded: {instance_name}  (n={instance.n})")

    # ── Build the list of configs to run ────────────────────────────────────────
    all_cfgs = build_configs()
    cfg_map = {c["label"]: c for c in all_cfgs}

    if args.all_configs:
        selected = list(cfg_map.keys())
    elif args.configs:
        selected = args.configs
    else:
        # Default: optimized 2-opt configs
        selected = [k for k in cfg_map if "2-opt" in k and "opt" in k]
        if not selected:
            selected = list(cfg_map.keys())[:4]

    print(f"Selected configs ({len(selected)}): {selected}")

    # ── Run each config ─────────────────────────────────────────────────────────
    results = []
    for label in selected:
        if label not in cfg_map:
            print(f"  [SKIP] Unknown config: {label}")
            continue

        factory = cfg_map[label]["factory"]
        run_fn = factory(instance)

        best_tour, best_cost = None, float("inf")
        for _ in range(args.runs):
            try:
                tour, cost = run_fn()
                if cost < best_cost:
                    best_tour, best_cost = tour[:], cost
            except Exception as e:
                print(f"  [ERROR] {label}: {e}")

        if best_tour is not None:
            results.append((label, best_tour, best_cost))
            print(f"  {label}  →  cost={best_cost:,.0f}")

    if not results:
        print("No results to visualize.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Display mode ────────────────────────────────────────────────────────────
    if args.single or len(results) == 1:
        label, tour, cost = results[0]
        save_path = os.path.join(args.out_dir, f"{instance_name}__{label}.png")
        plot_tour(tour, instance, title=label, save_path=save_path)

    elif args.compare:
        if len(results) < 2:
            print("[ERROR] --compare requires 2 configs.")
            return
        label_a, tour_a, _ = results[0]
        label_b, tour_b, _ = results[1]
        save_path = os.path.join(args.out_dir,
                                 f"{instance_name}__compare__{label_a}_vs_{label_b}.png")
        plot_compare(tour_a, tour_b, instance,
                     label_a=label_a, label_b=label_b,
                     save_path=save_path)

    else:
        # Multi-config grid
        save_path = os.path.join(args.out_dir, f"{instance_name}__multi.png")
        plot_multi_configs(results, instance, save_path=save_path)

    print(f"\n✓ Visualization complete. PNGs in: {args.out_dir}/")

if __name__ == "__main__":
    main()
