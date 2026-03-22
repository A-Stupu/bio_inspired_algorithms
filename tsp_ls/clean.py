# clean.py
# ========
# Cleans generated files under results/ and results/plots.
#
# Usage (from tsp_ls/ or project root):
#   python -m clean
#   python clean.py --dry-run

import argparse
import os


def _collect_targets(root_dir):
    targets = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name == ".gitkeep":
                continue
            targets.append(os.path.join(dirpath, name))
    return targets


def main():
    parser = argparse.ArgumentParser(description="Clean results and plots")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show files to delete without removing them")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")

    if not os.path.isdir(results_dir):
        print("No results directory found.")
        return

    targets = _collect_targets(results_dir)
    if not targets:
        print("No files to remove.")
        return

    for path in targets:
        rel = os.path.relpath(path, base_dir)
        if args.dry_run:
            print(f"[DRY] {rel}")
        else:
            os.remove(path)
            print(f"Removed {rel}")


if __name__ == "__main__":
    main()
