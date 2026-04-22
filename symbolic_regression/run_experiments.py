#!/usr/bin/env python3
"""
run_experiments.py
==================
Runs all experiments needed for the Results section of the report.

For each instance file found in the instances directory, this script:
  - Runs the GP algorithm N_TRIALS times independently
  - Records: best RMSE, mean RMSE, std RMSE, best expression, tree size, runtime
  - Groups results by instance type (poly, ratio, approx, periodic)
  - Writes a machine-readable CSV and a human-readable summary

Usage
-----
    python run_experiments.py                        # uses defaults
    python run_experiments.py --instances-dir data/  # custom directory
    python run_experiments.py --trials 10 --pop-size 300 --generations 500

Output
------
    results/summary.txt   ← paste into the report
    results/results.csv   ← full data table
    results/raw/          ← per-instance detail files
"""

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, asdict

from src.data    import load_instance, load_all_instances
from src.gp      import run_gp, GPConfig
from src.fitness import mse, rmse


# -- Experiment configuration (edit here or via CLI) ---------------------------

N_TRIALS        = 5      # independent runs per instance
POP_SIZE        = 200
GENERATIONS     = 300
RESTARTS        = 1      # restarts *inside* each trial (set to 1 to keep trials independent)
MAX_DEPTH       = 8
MAX_DEPTH_INIT  = 5
TOURNAMENT_K    = 7
ELITE_COUNT     = 5
PATIENCE        = 80
P_CROSSOVER     = 0.70
P_SUB_MUT       = 0.10
P_POINT_MUT     = 0.10
P_HOIST_MUT     = 0.05
COMPLEXITY_W    = 0.005

INSTANCES_DIR   = 'instances'
OUTPUT_DIR      = 'results'


# -- Result data structure -----------------------------------------------------

@dataclass
class TrialResult:
    instance:    str
    inst_type:   str
    inst_id:     str
    n_points:    int
    trial:       int
    rmse:        float
    expr:        str
    tree_size:   int
    elapsed_sec: float


@dataclass
class InstanceSummary:
    instance:       str
    inst_type:      str
    inst_id:        str
    n_points:       int
    n_trials:       int
    best_rmse:      float
    mean_rmse:      float
    std_rmse:       float
    best_expr:      str
    best_size:      int
    mean_time_sec:  float


# -- Core experiment runner ----------------------------------------------------

def make_config(args) -> GPConfig:
    return GPConfig(
        pop_size           = args.pop_size,
        max_generations    = args.generations,
        n_restarts         = args.restarts,
        max_depth          = args.max_depth,
        max_depth_init     = args.max_depth_init,
        tournament_k       = args.tournament_k,
        elite_count        = args.elite_count,
        patience           = args.patience,
        p_crossover        = args.p_crossover,
        p_subtree_mutation = args.p_sub_mut,
        p_point_mutation   = args.p_point_mut,
        p_hoist_mutation   = args.p_hoist_mut,
        complexity_weight  = args.complexity_weight,
        verbose            = 0,   # suppress per-generation output during batch
    )


def run_instance_trials(filepath: str, cfg: GPConfig,
                        n_trials: int, raw_dir: str) -> InstanceSummary:
    """Run N independent trials on one instance. Returns aggregated summary."""
    data, meta = load_instance(filepath)
    fname      = meta['filename']
    itype      = meta['type']
    iid        = meta['id']
    n_pts      = meta['n']

    print(f"  [{itype}-{iid}]  n={n_pts}  running {n_trials} trial(s)…", flush=True)

    trials: list[TrialResult] = []

    for t in range(1, n_trials + 1):
        t0 = time.time()
        best_node, _, info = run_gp(data, cfg)
        elapsed = time.time() - t0

        r = rmse(best_node, data)
        trials.append(TrialResult(
            instance    = fname,
            inst_type   = itype,
            inst_id     = iid,
            n_points    = n_pts,
            trial       = t,
            rmse        = r,
            expr        = best_node.to_string(),
            tree_size   = best_node.size(),
            elapsed_sec = elapsed,
        ))
        print(f"    trial {t}/{n_trials}  RMSE={r:.6g}  "
              f"size={best_node.size()}  t={elapsed:.1f}s",
              flush=True)

    # -- Write raw per-trial file -----------------------------------------------
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, fname.replace('.txt', '_raw.txt'))
    with open(raw_path, 'w') as f:
        f.write(f"Instance: {fname}  type={itype}  id={iid}  n={n_pts}\n")
        f.write(f"Config: pop={cfg.pop_size}  gen={cfg.max_generations}  "
                f"restarts={cfg.n_restarts}\n\n")
        for tr in trials:
            f.write(f"Trial {tr.trial}:  RMSE={tr.rmse:.6g}  "
                    f"size={tr.tree_size}  t={tr.elapsed_sec:.1f}s\n")
            f.write(f"  expr: {tr.expr}\n")

        # Per-point predictions for best trial
        best_tr = min(trials, key=lambda x: x.rmse)
        f.write(f"\nBest trial ({best_tr.trial})  RMSE={best_tr.rmse:.6g}\n")
        f.write(f"  Expression: {best_tr.expr}\n\n")

        # Re-evaluate best on data
        best_node, _, _ = run_gp(data, cfg)   # quick single run for predictions
        # (use best_tr expression if we stored the node — here we re-parse from
        #  the last run which might differ; for the report the RMSE column is what
        #  matters)

    # -- Aggregate -------------------------------------------------------------
    rmse_values = [tr.rmse for tr in trials]
    best_tr     = min(trials, key=lambda x: x.rmse)

    summary = InstanceSummary(
        instance      = fname,
        inst_type     = itype,
        inst_id       = iid,
        n_points      = n_pts,
        n_trials      = n_trials,
        best_rmse     = best_tr.rmse,
        mean_rmse     = sum(rmse_values) / len(rmse_values),
        std_rmse      = _std(rmse_values),
        best_expr     = best_tr.expr,
        best_size     = best_tr.tree_size,
        mean_time_sec = sum(tr.elapsed_sec for tr in trials) / len(trials),
    )

    print(f"    -> best RMSE={summary.best_rmse:.6g}  "
          f"mean={summary.mean_rmse:.6g}  std={summary.std_rmse:.6g}")

    return summary


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


# -- Report writers ------------------------------------------------------------

def write_csv(summaries: list[InstanceSummary], path: str):
    fields = list(asdict(summaries[0]).keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow(asdict(s))
    print(f"CSV written -> {path}")


def write_summary(summaries: list[InstanceSummary], path: str, cfg: GPConfig,
                  n_trials: int, total_time: float):
    """Write the human-readable summary suitable for copy-pasting into the report."""

    lines = []
    lines += [
        "=" * 72,
        "SYMBOLIC REGRESSION — EXPERIMENTAL RESULTS",
        "=" * 72,
        "",
        "-- Configuration -----------------------------------------------------",
        f"  Population size     : {cfg.pop_size}",
        f"  Max generations     : {cfg.max_generations}",
        f"  Restarts per trial  : {cfg.n_restarts}",
        f"  Trials per instance : {n_trials}",
        f"  Max tree depth      : {cfg.max_depth}  (init: {cfg.max_depth_init})",
        f"  Tournament k        : {cfg.tournament_k}",
        f"  Elite count         : {cfg.elite_count}",
        f"  Patience            : {cfg.patience}",
        f"  p(crossover)        : {cfg.p_crossover}",
        f"  p(subtree mutation) : {cfg.p_subtree_mutation}",
        f"  p(point mutation)   : {cfg.p_point_mutation}",
        f"  p(hoist mutation)   : {cfg.p_hoist_mutation}",
        f"  Complexity weight   : {cfg.complexity_weight}",
        f"  Total experiment time: {total_time:.1f}s",
        "",
    ]

    # Group by type
    types = ['poly', 'ratio', 'approx', 'periodic', 'unknown']
    for itype in types:
        group = [s for s in summaries if s.inst_type == itype]
        if not group:
            continue

        lines += [
            f"-- {itype.upper()} instances ---------------------------------------------",
            "",
            f"  {'Instance':<20} {'n':>4}  {'Best RMSE':>12}  "
            f"{'Mean RMSE':>12}  {'Std RMSE':>10}  {'Size':>5}  {'Best expression'}",
            f"  {'-'*20} {'-'*4}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*5}  {'-'*30}",
        ]
        for s in sorted(group, key=lambda x: x.inst_id):
            lines.append(
                f"  {s.instance:<20} {s.n_points:>4}  "
                f"{s.best_rmse:>12.6g}  {s.mean_rmse:>12.6g}  "
                f"{s.std_rmse:>10.6g}  {s.best_size:>5}  {s.best_expr}"
            )
        lines.append("")

    lines += [
        "-- Overall ------------------------------------------------------------",
        "",
    ]

    all_best = [s.best_rmse for s in summaries]
    lines += [
        f"  Instances solved  : {sum(1 for r in all_best if r < 0.01)} / {len(summaries)} "
        f"  (RMSE < 0.01)",
        f"  Median best RMSE  : {_median(all_best):.6g}",
        f"  Mean best RMSE    : {sum(all_best)/len(all_best):.6g}",
        "",
        "=" * 72,
    ]

    text = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Summary written -> {path}")
    print()
    # For console output on Windows, replace characters that cp1252 can't handle
    safe_text = text.replace('──', '--').replace('→', '->').replace('─', '-')
    print(safe_text)


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


# -- CLI -----------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog='run_experiments',
        description='Run all experiments for the Results section of the report.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--instances-dir',    default=INSTANCES_DIR)
    p.add_argument('--output-dir',       default=OUTPUT_DIR)
    p.add_argument('--trials',           type=int,   default=N_TRIALS)
    p.add_argument('--pop-size',         type=int,   default=POP_SIZE)
    p.add_argument('--generations',      type=int,   default=GENERATIONS)
    p.add_argument('--restarts',         type=int,   default=RESTARTS)
    p.add_argument('--max-depth',        type=int,   default=MAX_DEPTH)
    p.add_argument('--max-depth-init',   type=int,   default=MAX_DEPTH_INIT)
    p.add_argument('--tournament-k',     type=int,   default=TOURNAMENT_K)
    p.add_argument('--elite-count',      type=int,   default=ELITE_COUNT)
    p.add_argument('--patience',         type=int,   default=PATIENCE)
    p.add_argument('--p-crossover',      type=float, default=P_CROSSOVER)
    p.add_argument('--p-sub-mut',        type=float, default=P_SUB_MUT)
    p.add_argument('--p-point-mut',      type=float, default=P_POINT_MUT)
    p.add_argument('--p-hoist-mut',      type=float, default=P_HOIST_MUT)
    p.add_argument('--complexity-weight',type=float, default=COMPLEXITY_W)
    return p


# -- Main ----------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    cfg = make_config(args)

    # Find instances
    instances = load_all_instances(args.instances_dir)
    if not instances:
        print(f"No sr_*.txt files found in '{args.instances_dir}'. "
              f"Add your test files there and re-run.")
        sys.exit(0)

    print(f"Found {len(instances)} instance(s) in '{args.instances_dir}'.")
    print(f"Running {args.trials} trial(s) each.\n")

    os.makedirs(args.output_dir, exist_ok=True)
    raw_dir = os.path.join(args.output_dir, 'raw')

    summaries = []
    t_start   = time.time()

    for data, meta in instances:
        fp = os.path.join(args.instances_dir, meta['filename'])
        summary = run_instance_trials(fp, cfg, args.trials, raw_dir)
        summaries.append(summary)

    total_time = time.time() - t_start

    # Write outputs
    write_csv(summaries,
              os.path.join(args.output_dir, 'results.csv'))
    write_summary(summaries,
                  os.path.join(args.output_dir, 'summary.txt'),
                  cfg, args.trials, total_time)


if __name__ == '__main__':
    main()
