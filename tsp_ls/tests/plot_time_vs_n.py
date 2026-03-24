# tests/plot_time_vs_n.py
# ==================================
# Runs selected configurations on instances of varying sizes (e.g., from bioalg-proj01-hz)
# and plots execution time versus the number of vertices (n) to empirically
# analyze algorithmic time complexity.
#
# It produces a two-panel plot in results/plots/:
#   - Linear scale: intuitive view of time growth.
#   - Log-log scale: slope indicates the empirical polynomial complexity O(n^k).
#
# Usage (from project root):
#     python -m tests.plot_time_vs_n
#     python -m tests.plot_time_vs_n --tsp_dir DB/bioalg-proj01-hz --runs 5

import sys
import os
import argparse
import pandas as pd
import numpy as np

# Allows running the script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.run import run_experiments, build_configs

# Parameters --------------------------------------------------------------------

DEFAULT_DIR = "DB/bioalg-proj01-hz"
OUT_CSV     = "results/time_complexity_results.csv"
OUT_PLOT    = "results/plots/time_complexity_vs_n.png"
N_RUNS      = 5  # Lower default runs since we are testing scaling on potentially large instances

# Select a representative subset of configs to keep execution time reasonable
# We compare naive vs optimized to highlight the performance gain of Delta Evaluation
SELECTED_CONFIGS = [
    "gls_naive_first__2-opt",
    "gls_opt__2-opt",
    "sa_naive_exp__medium__swap",
    "sa_opt__2-opt__exp_medium"
]

COLORS = {
    "gls_naive_first__2-opt":     "#d62728", # Red
    "gls_opt__2-opt":             "#2ca02c", # Green
    "sa_naive_exp__medium__swap": "#ff7f0e", # Orange
    "sa_opt__2-opt__exp_medium":  "#1f77b4"  # Blue
}

# -------------------------------------------------------------------------------

def run_scaling_experiments(tsp_dir, runs):
    """Runs the selected configurations and saves the results to a CSV."""
    print(f"=== Running scaling experiments on {tsp_dir} ===")
    
    all_configs = build_configs()
    configs_to_run = [cfg for cfg in all_configs if cfg["label"] in SELECTED_CONFIGS]
    
    if not configs_to_run:
        print("[ERROR] None of the selected configs were found.")
        sys.exit(1)

    # Call the main runner programmatically
    run_experiments(
        tsp_dir=tsp_dir,
        out_path=OUT_CSV,
        n_runs=runs,
        max_nodes=250,
        tsp_file=None,
        configs=configs_to_run
    )

def plot_complexity(csv_path, save_path):
    """Reads the CSV and plots Time vs N in both linear and log-log scales."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[ERROR] CSV is empty.")
        return

    # Sort values by n for proper line plotting
    df = df.sort_values(by="n")
    
    # Create a figure with two subplots side-by-side
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 6))
    
    configs = df["config"].unique()
    
    for config in configs:
        subset = df[df["config"] == config]
        n_vals = subset["n"].values
        time_vals = subset["mean_time_s"].values
        
        # Skip if not enough data points
        if len(n_vals) < 2:
            continue
            
        color = COLORS.get(config, "#333333")
        
        # 1. Linear Plot
        ax_lin.plot(n_vals, time_vals, marker='o', linestyle='-', color=color, label=config, alpha=0.8)
        
        # 2. Log-Log Plot
        ax_log.plot(n_vals, time_vals, marker='o', linestyle='-', color=color, label=config, alpha=0.8)
        
        # Calculate empirical complexity (slope of log-log regression)
        # log(Time) = k * log(n) + c  =>  k is the slope
        # Filter out zero or negative times for log calculation
        valid_mask = (n_vals > 0) & (time_vals > 0)
        if sum(valid_mask) >= 2:
            log_n = np.log10(n_vals[valid_mask])
            log_t = np.log10(time_vals[valid_mask])
            slope, _ = np.polyfit(log_n, log_t, 1)
            
            # Annotate the slope on the log-log plot to help with the report interpretation
            # The slope 'k' means the empirical complexity is roughly O(n^k)
            ax_log.annotate(f"O(n^{slope:.2f})", 
                            xy=(n_vals[-1], time_vals[-1]),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=9, color=color, fontweight='bold')

    # Formatting Linear Plot
    ax_lin.set_title("Execution Time vs Number of Vertices (Linear Scale)", fontsize=11)
    ax_lin.set_xlabel("Number of vertices (n)", fontsize=10)
    ax_lin.set_ylabel("Mean Execution Time (s)", fontsize=10)
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(fontsize=8)

    # Formatting Log-Log Plot
    ax_log.set_title("Execution Time vs Number of Vertices (Log-Log Scale)", fontsize=11)
    ax_log.set_xlabel("Number of vertices (n)", fontsize=10)
    ax_log.set_ylabel("Mean Execution Time (s)", fontsize=10)
    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.grid(True, which="both", ls="--", alpha=0.3)
    ax_log.legend(fontsize=8)
    
    # Add an explanatory text box for the report interpretation
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax_log.text(0.05, 0.95, "Slope corresponds to 'k' in O(n^k)\n"
                            "Naive ~ O(n³)\n"
                            "Optimized ~ O(n²)", 
                transform=ax_log.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

    fig.suptitle("Empirical Time Complexity Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Complexity plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Time Complexity vs Number of Vertices")
    parser.add_argument("--tsp_dir", default=DEFAULT_DIR, help=f"Directory with TSP files (default: {DEFAULT_DIR})")
    parser.add_argument("--runs", type=int, default=N_RUNS, help=f"Runs per config (default: {N_RUNS})")
    parser.add_argument("--skip_run", action="store_true", help="Skip running experiments and just plot existing CSV")
    args = parser.parse_args()

    if not args.skip_run:
        run_scaling_experiments(args.tsp_dir, args.runs)
    else:
        print(f"Skipping experiments, reading from {OUT_CSV}...")

    print("\nGenerating plots...")
    plot_complexity(OUT_CSV, OUT_PLOT)

if __name__ == "__main__":
    main()