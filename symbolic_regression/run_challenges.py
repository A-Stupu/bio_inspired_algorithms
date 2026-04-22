#!/usr/bin/env python3
"""
run_challenges.py
=================
Dedicated runner for the challenge dataset (challenge_a, challenge_b, challenge_c).

Key differences from run_experiments.py
----------------------------------------
- Larger population and more generations by default (harder instances)
- More restarts (stochastic algorithm benefits from re-trying)
- Applies constant optimisation (hill-climbing) on the best tree found
  before reporting, to pin down irrational constants (pi, e, sqrt(2), ...)
- Groups output by challenge level (a / b / c) with separate commentary
- Writes results/challenges_summary.txt and results/challenges.csv

Usage
-----
    # Default parameters (recommended for the report):
    python run_challenges.py

    # Quick smoke test:
    python run_challenges.py --trials 2 --pop-size 100 --generations 100

    # Custom instance directory:
    python run_challenges.py --instances-dir path/to/challenges/
"""

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, asdict

from src.data      import load_instance, load_all_instances
from src.gp        import run_gp, GPConfig
from src.fitness   import mse, rmse
from src.operators import optimise_constants


# -- Default parameters (tuned for challenge instances) ------------------------

N_TRIALS        = 10    # more restarts -> better chance on hard instances
POP_SIZE        = 300   # larger population -> more diversity
GENERATIONS     = 500   # more generations -> deeper search
RESTARTS        = 1     # restarts inside each trial (kept at 1 for independent stats)
MAX_DEPTH       = 8
MAX_DEPTH_INIT  = 5
TOURNAMENT_K    = 7
ELITE_COUNT     = 5
PATIENCE        = 120   # more patience for hard instances
P_CROSSOVER     = 0.70
P_SUB_MUT       = 0.10
P_POINT_MUT     = 0.12  # slightly higher: constants need more fine-tuning
P_HOIST_MUT     = 0.05
COMPLEXITY_W    = 0.003  # softer penalty: allow larger trees on harder instances
CONST_OPT_STEPS = 60    # hill-climbing steps applied after GP

INSTANCES_DIR   = 'instances'
OUTPUT_DIR      = 'results'


# -- Data structures -----------------------------------------------------------

@dataclass
class ChallengeResult:
    instance:       str
    level:          str   # 'a', 'b', 'c'
    inst_id:        str
    n_points:       int
    n_trials:       int
    best_rmse:      float
    mean_rmse:      float
    std_rmse:       float
    best_rmse_after_opt: float   # after constant optimisation
    best_expr:      str
    best_expr_opt:  str          # expression after constant optimisation
    best_size:      int
    mean_time_sec:  float


# -- Helpers -------------------------------------------------------------------

def _std(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m)**2 for v in values) / (len(values) - 1))

def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2

def _parse_level(meta):
    """Extract challenge level (a/b/c) from instance type field."""
    t = meta.get('type', '')
    # type field is e.g. 'challenge_a' or 'challenge'
    for level in ('a', 'b', 'c'):
        if t.endswith(f'_{level}') or t == f'challenge_{level}':
            return level
    # Fallback: parse from filename
    fname = meta.get('filename', '')
    for level in ('_a_', '_b_', '_c_'):
        if level in fname:
            return level.strip('_')
    return '?'


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
        verbose            = 0,
    )


# -- Per-instance runner -------------------------------------------------------

def run_challenge_instance(filepath, cfg, n_trials, const_opt_steps, raw_dir):
    data, meta = load_instance(filepath)
    fname  = meta['filename']
    level  = _parse_level(meta)
    iid    = meta['id']
    n_pts  = meta['n']

    print(f"  [challenge_{level}-{iid}]  n={n_pts}  "
          f"running {n_trials} trial(s)...", flush=True)

    trial_rmse   = []
    trial_nodes  = []
    trial_times  = []

    best_node_overall = None
    best_rmse_overall = float('inf')

    for t in range(1, n_trials + 1):
        t0 = time.time()
        node, _, _ = run_gp(data, cfg)
        elapsed = time.time() - t0

        r = rmse(node, data)
        trial_rmse.append(r)
        trial_nodes.append(node)
        trial_times.append(elapsed)

        if r < best_rmse_overall:
            best_rmse_overall = r
            best_node_overall = node

        print(f"    trial {t}/{n_trials}  RMSE={r:.6g}  "
              f"size={node.size()}  t={elapsed:.1f}s", flush=True)

    # -- Constant optimisation on the best tree --------------------------------
    expr_before = best_node_overall.to_string()
    rmse_before = best_rmse_overall

    opt_node = optimise_constants(best_node_overall, data,
                                  n_steps=const_opt_steps)
    rmse_after = rmse(opt_node, data)
    expr_after = opt_node.to_string()

    improvement = rmse_before - rmse_after
    print(f"    -> best RMSE={rmse_before:.6g}  "
          f"after const-opt={rmse_after:.6g}  "
          f"(improvement: {improvement:+.6g})")

    # -- Write raw detail file -------------------------------------------------
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, fname.replace('.txt', '_challenge_raw.txt'))
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(f"Instance : {fname}  level={level}  id={iid}  n={n_pts}\n")
        f.write(f"Config   : pop={cfg.pop_size}  gen={cfg.max_generations}  "
                f"patience={cfg.patience}\n\n")
        for i, (r, nd, t) in enumerate(zip(trial_rmse, trial_nodes, trial_times), 1):
            f.write(f"Trial {i}: RMSE={r:.6g}  size={nd.size()}  t={t:.1f}s\n")
            f.write(f"  expr: {nd.to_string()}\n")
        f.write(f"\nBest before opt : RMSE={rmse_before:.6g}\n")
        f.write(f"  {expr_before}\n")
        f.write(f"\nBest after  opt : RMSE={rmse_after:.6g}\n")
        f.write(f"  {expr_after}\n")
        f.write("\nPredictions (after opt):\n")
        for x, y in data:
            pred = opt_node.evaluate(x)
            f.write(f"  x={x:8.4f}  y={y:10.4f}  f(x)={pred:10.4f}  "
                    f"err={abs(pred-y):.4g}\n")

    return ChallengeResult(
        instance            = fname,
        level               = level,
        inst_id             = iid,
        n_points            = n_pts,
        n_trials            = n_trials,
        best_rmse           = rmse_before,
        mean_rmse           = sum(trial_rmse) / len(trial_rmse),
        std_rmse            = _std(trial_rmse),
        best_rmse_after_opt = rmse_after,
        best_expr           = expr_before,
        best_expr_opt       = expr_after,
        best_size           = best_node_overall.size(),
        mean_time_sec       = sum(trial_times) / len(trial_times),
    )


# -- Output writers ------------------------------------------------------------

def write_csv(results, path):
    if not results:
        return
    fields = list(asdict(results[0]).keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"CSV written -> {path}")


def write_summary(results, path, cfg, n_trials, const_opt_steps, total_time):
    lines = [
        "=" * 72,
        "SYMBOLIC REGRESSION -- CHALLENGE RESULTS",
        "=" * 72,
        "",
        "-- Configuration --",
        f"  Population size       : {cfg.pop_size}",
        f"  Max generations       : {cfg.max_generations}",
        f"  Trials per instance   : {n_trials}",
        f"  Patience              : {cfg.patience}",
        f"  Complexity weight     : {cfg.complexity_weight}",
        f"  p(point mutation)     : {cfg.p_point_mutation}  (float-aware)",
        f"  Constant opt steps    : {const_opt_steps}",
        f"  Total experiment time : {total_time:.1f}s",
        "",
    ]

    for level in ('a', 'b', 'c'):
        group = sorted([r for r in results if r.level == level],
                       key=lambda r: r.inst_id)
        if not group:
            continue

        label = {'a': 'CHALLENGE A -- simple (should approximate well)',
                 'b': 'CHALLENGE B -- harder (some deviation expected)',
                 'c': 'CHALLENGE C -- bonus'}[level]

        lines += [
            f"-- {label} --",
            "",
            f"  {'Instance':<22} {'n':>4}  {'Best RMSE':>10}  "
            f"{'After opt':>10}  {'Mean':>10}  {'Std':>8}  {'Sz':>4}  Best expression (after opt)",
            f"  {'-'*22} {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*4}  {'-'*35}",
        ]
        for r in group:
            lines.append(
                f"  {r.instance:<22} {r.n_points:>4}  "
                f"{r.best_rmse:>10.5g}  {r.best_rmse_after_opt:>10.5g}  "
                f"{r.mean_rmse:>10.5g}  {r.std_rmse:>8.5g}  "
                f"{r.best_size:>4}  {r.best_expr_opt}"
            )
        lines.append("")

    # Overall stats
    all_best      = [r.best_rmse_after_opt for r in results]
    solved        = sum(1 for v in all_best if v < 0.01)
    good          = sum(1 for v in all_best if v < 0.1)

    lines += [
        "-- Overall --",
        "",
        f"  Total instances       : {len(results)}",
        f"  RMSE < 0.01  (solved) : {solved}",
        f"  RMSE < 0.10  (good)   : {good}",
        f"  Median best RMSE      : {_median(all_best):.5g}",
        "",
        "=" * 72,
    ]

    text = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Summary written -> {path}")
    print()
    # Safe console print (no unicode arrows)
    print(text)


# -- CLI -----------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog='run_challenges',
        description='Run challenge instances with constant optimisation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--instances-dir',     default=INSTANCES_DIR)
    p.add_argument('--output-dir',        default=OUTPUT_DIR)
    p.add_argument('--trials',            type=int,   default=N_TRIALS)
    p.add_argument('--pop-size',          type=int,   default=POP_SIZE)
    p.add_argument('--generations',       type=int,   default=GENERATIONS)
    p.add_argument('--restarts',          type=int,   default=RESTARTS)
    p.add_argument('--max-depth',         type=int,   default=MAX_DEPTH)
    p.add_argument('--max-depth-init',    type=int,   default=MAX_DEPTH_INIT)
    p.add_argument('--tournament-k',      type=int,   default=TOURNAMENT_K)
    p.add_argument('--elite-count',       type=int,   default=ELITE_COUNT)
    p.add_argument('--patience',          type=int,   default=PATIENCE)
    p.add_argument('--p-crossover',       type=float, default=P_CROSSOVER)
    p.add_argument('--p-sub-mut',         type=float, default=P_SUB_MUT)
    p.add_argument('--p-point-mut',       type=float, default=P_POINT_MUT)
    p.add_argument('--p-hoist-mut',       type=float, default=P_HOIST_MUT)
    p.add_argument('--complexity-weight', type=float, default=COMPLEXITY_W)
    p.add_argument('--const-opt-steps',   type=int,   default=CONST_OPT_STEPS)
    p.add_argument('--level',             default=None,
                   help="Run only instances of this level: a, b, or c.")
    return p


# -- Main ----------------------------------------------------------------------

def main():
    args   = build_parser().parse_args()
    cfg    = make_config(args)
    raw_dir = os.path.join(args.output_dir, 'raw')

    # Load only challenge instances
    all_instances = load_all_instances(args.instances_dir)
    instances = [
        (data, meta) for data, meta in all_instances
        if 'challenge' in meta.get('type', '') or
           'challenge' in meta.get('filename', '')
    ]

    # Optionally filter by level
    if args.level:
        instances = [
            (d, m) for d, m in instances
            if _parse_level(m) == args.level
        ]

    if not instances:
        print(f"No challenge instances found in '{args.instances_dir}'.")
        print("Make sure filenames contain 'challenge' (e.g. sr_challenge_a_01.txt).")
        sys.exit(0)

    print(f"Found {len(instances)} challenge instance(s).")
    print(f"Trials: {args.trials}  |  Pop: {args.pop_size}  |  "
          f"Gen: {args.generations}  |  Const-opt steps: {args.const_opt_steps}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    results   = []
    t_start   = time.time()

    for data, meta in instances:
        fp = os.path.join(args.instances_dir, meta['filename'])
        r  = run_challenge_instance(fp, cfg, args.trials,
                                    args.const_opt_steps, raw_dir)
        results.append(r)

    total_time = time.time() - t_start

    write_csv(results, os.path.join(args.output_dir, 'challenges.csv'))
    write_summary(results,
                  os.path.join(args.output_dir, 'challenges_summary.txt'),
                  cfg, args.trials, args.const_opt_steps, total_time)


if __name__ == '__main__':
    main()