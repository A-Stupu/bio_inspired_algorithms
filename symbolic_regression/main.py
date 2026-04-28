#!/usr/bin/env python3
"""
Symbolic Regression via Genetic Programming
============================================

Usage examples
--------------
# Run on a single instance with defaults:
    python main.py instances/sr_poly_4.txt

# Custom parameters:
    python main.py instances/sr_poly_4.txt \
        --pop-size 300 --generations 500 --restarts 5 \
        --complexity-weight 0.005 --verbose 2

# Run on all instances in a directory:
    python main.py --batch instances/

# Show help:
    python main.py --help
"""
import argparse
import os
import sys

from src.data import load_instance, load_all_instances
from src.gp   import run_gp, GPConfig
from src.fitness import mse, rmse


# Argument parsing

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='symbolic_regression',
        description='Genetic programming symbolic regression.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    p.add_argument('input', nargs='?',
                   help='Path to a single instance file (.txt) or '
                        'a directory (use with --batch).')
    p.add_argument('--batch', action='store_true',
                   help='Process all sr_*.txt files in the given directory.')

    # GP parameters
    p.add_argument('--pop-size',        type=int,   default=200,
                   help='Population size.')
    p.add_argument('--generations',     type=int,   default=300,
                   help='Maximum number of generations per restart.')
    p.add_argument('--restarts',        type=int,   default=3,
                   help='Number of independent restarts (best is kept).')
    p.add_argument('--max-depth',       type=int,   default=8,
                   help='Maximum tree depth after operators.')
    p.add_argument('--max-depth-init',  type=int,   default=5,
                   help='Maximum tree depth at initialisation.')
    p.add_argument('--tournament-k',    type=int,   default=7,
                   help='Tournament size for parent selection.')
    p.add_argument('--elite-count',     type=int,   default=5,
                   help='Number of elites preserved each generation.')
    p.add_argument('--patience',        type=int,   default=80,
                   help='Stop if no improvement for this many generations.')

    # Operator probabilities
    p.add_argument('--p-crossover',         type=float, default=0.70)
    p.add_argument('--p-subtree-mutation',  type=float, default=0.10)
    p.add_argument('--p-point-mutation',    type=float, default=0.10)
    p.add_argument('--p-hoist-mutation',    type=float, default=0.05)

    # Fitness
    p.add_argument('--complexity-weight', type=float, default=0.005,
                   help='Coefficient of the complexity penalty in the fitness.')

    # Output
    p.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                   help='Verbosity level (0=silent, 1=summary, 2=per-generation).')
    p.add_argument('--output-dir', type=str, default=None,
                   help='If given, write results to text files in this directory.')

    return p


# Single-instance runner

def run_instance(filepath: str, cfg: GPConfig, output_dir: str | None = None):
    data, meta = load_instance(filepath)

    print(f"\n{'='*60}")
    print(f"Instance : {meta['filename']}  "
          f"(type={meta['type']}, id={meta['id']}, n={meta['n']})")
    print(f"{'='*60}")

    best_node, best_fit, info = run_gp(data, cfg)

    expr  = best_node.to_string()
    error = rmse(best_node, data)

    print(f"\n-- Result ----------------------------------------------")
    print(f"  Expression : {expr}")
    print(f"  RMSE       : {error:.6g}")
    print(f"  Tree size  : {info['best_size']} nodes")
    print(f"  Time       : {info['elapsed_sec']:.1f}s")

    if cfg.verbose >= 1:
        print("\n  Predictions:")
        for x, y in data:
            pred = best_node.evaluate(x)
            print(f"    x={x:8.4f}  y={y:10.4f}  f(x)={pred:10.4f}  "
                  f"err={abs(pred - y):.4g}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_name = meta['filename'].replace('.txt', '_result.txt')
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, 'w') as f:
            f.write(f"Instance: {meta['filename']}\n")
            f.write(f"Expression: {expr}\n")
            f.write(f"RMSE: {error:.6g}\n")
            f.write(f"Tree size: {info['best_size']}\n")
            f.write(f"Time: {info['elapsed_sec']:.1f}s\n")
            f.write("\nPredictions:\n")
            for x, y in data:
                pred = best_node.evaluate(x)
                f.write(f"  x={x}  y={y}  f(x)={pred:.6g}\n")
        print(f"  → Results written to {out_path}")

    return expr, error, info


# Main

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit(0)

    cfg = GPConfig(
        pop_size           = args.pop_size,
        max_generations    = args.generations,
        n_restarts         = args.restarts,
        max_depth          = args.max_depth,
        max_depth_init     = args.max_depth_init,
        tournament_k       = args.tournament_k,
        elite_count        = args.elite_count,
        patience           = args.patience,
        p_crossover        = args.p_crossover,
        p_subtree_mutation = args.p_subtree_mutation,
        p_point_mutation   = args.p_point_mutation,
        p_hoist_mutation   = args.p_hoist_mutation,
        complexity_weight  = args.complexity_weight,
        verbose            = args.verbose,
    )

    if args.batch:
        instances = load_all_instances(args.input)
        print(f"Found {len(instances)} instance(s) in '{args.input}'.")
        for _, meta in instances:
            fp = os.path.join(args.input, meta['filename'])
            run_instance(fp, cfg, args.output_dir)
    else:
        run_instance(args.input, cfg, args.output_dir)


if __name__ == '__main__':
    main()
