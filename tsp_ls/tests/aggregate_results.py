# tests/aggregate_results.py
# ==========================
# Aggregate experiment results from results.csv.
#
# Produces:
#   1) Per-instance summary with relative metrics vs a baseline config (mandatory)
#   2) Optional group summary (by size, prefix, or explicit mapping)
#
# Usage (from tsp_ls/ or project root):
#   python -m tests.aggregate_results --in results/results.csv --baseline gls_opt__2-opt
#   python -m tests.aggregate_results --group_by n
#   python -m tests.aggregate_results --group_map results/groups.csv

import argparse
import csv
import os
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_float(val):
    if val is None or val == "" or val.lower() == "nan":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _read_group_map(path):
    mapping = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "instance" not in reader.fieldnames or "group" not in reader.fieldnames:
            raise ValueError("group_map must have headers: instance,group")
        for row in reader:
            inst = row.get("instance", "").strip()
            grp = row.get("group", "").strip()
            if inst and grp:
                mapping[inst] = grp
    return mapping


def _infer_prefix(instance_name):
    prefix = ""
    for ch in instance_name:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return prefix or instance_name


def _load_results(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(path, title, description_lines, table_fields, table_rows, plot_rel_path=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        for line in description_lines:
            f.write(f"{line}\n")
        f.write("\n")

        if plot_rel_path:
            f.write(f"![{title}]({plot_rel_path})\n\n")

        # Markdown table
        f.write("| " + " | ".join(table_fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(table_fields)) + " |\n")
        for row in table_rows:
            values = [str(row.get(k, "")) for k in table_fields]
            f.write("| " + " | ".join(values) + " |\n")


def _plot_per_instance(per_instance, out_path):
    instances = sorted({r["instance"] for r in per_instance})
    configs = sorted({r["config"] for r in per_instance})
    if not instances or not configs:
        return None

    data = {inst: {cfg: None for cfg in configs} for inst in instances}
    for row in per_instance:
        inst = row["instance"]
        cfg = row["config"]
        data[inst][cfg] = _to_float(row.get("mean_cost"))

    n_cols = 2 if len(instances) > 1 else 1
    n_rows = (len(instances) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(8, len(configs) * 0.6), n_rows * 4))
    axes = np.array(axes).flatten()

    x = np.arange(len(configs))
    for idx, inst in enumerate(instances):
        ax = axes[idx]
        ys = [data[inst][cfg] if data[inst][cfg] is not None else 0 for cfg in configs]
        ax.bar(x, ys, color="#4C78A8", alpha=0.85)
        ax.set_title(f"{inst} — mean_cost")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=90, fontsize=7)
        ax.set_ylabel("Mean cost")
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(instances), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_group_summary(group_rows, out_path):
    groups = sorted({r["group"] for r in group_rows})
    configs = sorted({r["config"] for r in group_rows})
    if not groups or not configs:
        return None

    data = {grp: {cfg: None for cfg in configs} for grp in groups}
    for row in group_rows:
        grp = row["group"]
        cfg = row["config"]
        data[grp][cfg] = _to_float(row.get("mean_rel_to_baseline"))

    n_cols = 2 if len(groups) > 1 else 1
    n_rows = (len(groups) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(8, len(configs) * 0.6), n_rows * 4))
    axes = np.array(axes).flatten()

    x = np.arange(len(configs))
    for idx, grp in enumerate(groups):
        ax = axes[idx]
        ys = [data[grp][cfg] if data[grp][cfg] is not None else 0 for cfg in configs]
        ax.bar(x, ys, color="#F58518", alpha=0.85)
        ax.set_title(f"{grp} — mean_rel_to_baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=90, fontsize=7)
        ax.set_ylabel("Mean rel to baseline")
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Aggregate TSP experiment results")
    parser.add_argument("--in", dest="in_path", default="results/results.csv",
                        help="Input results CSV (default: results/results.csv)")
    parser.add_argument("--out_instance", default="results/summary_per_instance.csv",
                        help="Output per-instance summary CSV")
    parser.add_argument("--out_group", default="results/summary_by_group.csv",
                        help="Output group summary CSV")
    parser.add_argument("--baseline", default="gls_opt__2-opt",
                        help="Baseline config label for relative metrics")
    parser.add_argument("--group_by", choices=["none", "n", "prefix"], default="n",
                        help="Grouping mode for the optional summary")
    parser.add_argument("--group_map", default=None,
                        help="CSV mapping with headers: instance,group")
    args = parser.parse_args()

    rows = _load_results(args.in_path)
    if not rows:
        print("No rows found.")
        return

    # Build per-instance baseline lookup
    baseline_by_instance = {}
    for row in rows:
        if row.get("config") == args.baseline:
            inst = row.get("instance")
            baseline_by_instance[inst] = _to_float(row.get("mean_cost"))

    # Per-instance summary (mandatory)
    per_instance = []
    for row in rows:
        inst = row.get("instance")
        baseline = baseline_by_instance.get(inst)
        mean_cost = _to_float(row.get("mean_cost"))

        rel_ratio = None
        rel_impr = None
        if baseline is not None and mean_cost is not None and baseline > 0:
            rel_ratio = mean_cost / baseline
            rel_impr = 100.0 * (baseline - mean_cost) / baseline

        per_instance.append({
            "instance": inst,
            "n": row.get("n"),
            "config": row.get("config"),
            "best_cost": row.get("best_cost"),
            "mean_cost": row.get("mean_cost"),
            "std_cost": row.get("std_cost"),
            "mean_time_s": row.get("mean_time_s"),
            "std_time_s": row.get("std_time_s"),
            "mean_init_cost": row.get("mean_init_cost", ""),
            "improvement_pct": row.get("improvement_pct", ""),
            "rel_mean_to_baseline": f"{rel_ratio:.6f}" if rel_ratio is not None else "",
            "rel_improvement_pct": f"{rel_impr:.3f}" if rel_impr is not None else "",
        })

    per_instance.sort(key=lambda r: (r["instance"], r["config"]))

    per_instance_fields = [
        "instance",
        "n",
        "config",
        "best_cost",
        "mean_cost",
        "std_cost",
        "mean_time_s",
        "std_time_s",
        "mean_init_cost",
        "improvement_pct",
        "rel_mean_to_baseline",
        "rel_improvement_pct",
    ]
    _write_csv(args.out_instance, per_instance_fields, per_instance)
    print(f"Wrote per-instance summary: {args.out_instance}")

    per_instance_plot = _plot_per_instance(
        per_instance,
        out_path=os.path.join("results", "plots", "summary_per_instance.png"),
    )

    per_instance_md = os.path.splitext(args.out_instance)[0] + ".md"
    _write_markdown_report(
        per_instance_md,
        title="Per-instance summary",
        description_lines=[
            "Columns:",
            "- instance: Instance name (filename without extension).",
            "- n: Number of nodes in the instance.",
            "- config: Configuration label used in the run.",
            "- best_cost: Best tour cost observed across runs.",
            "- mean_cost: Average tour cost across runs.",
            "- std_cost: Standard deviation of tour cost across runs.",
            "- mean_time_s: Average runtime per run (seconds).",
            "- std_time_s: Standard deviation of runtime (seconds).",
            "- mean_init_cost: Average initial tour cost (only for configs that log it).",
            "- improvement_pct: Percent improvement from mean_init_cost to mean_cost.",
            "- rel_mean_to_baseline: mean_cost / baseline_mean_cost (values < 1 are better).",
            "- rel_improvement_pct: Percent improvement vs baseline (positive is better).",
        ],
        table_fields=per_instance_fields,
        table_rows=per_instance,
        plot_rel_path="plots/summary_per_instance.png" if per_instance_plot else None,
    )
    print(f"Wrote per-instance report: {per_instance_md}")

    # Optional group summary
    if args.group_by == "none" and not args.group_map:
        print("Skipping group summary (group_by=none and no group_map).")
        return

    group_map = {}
    if args.group_map:
        group_map = _read_group_map(args.group_map)

    def resolve_group(inst, n_val):
        if inst in group_map:
            return group_map[inst]
        if args.group_by == "n":
            return f"n={n_val}"
        if args.group_by == "prefix":
            return _infer_prefix(inst)
        return "ungrouped"

    # Aggregate relative ratio by group + config
    grouped = {}
    for row in per_instance:
        inst = row["instance"]
        n_val = row["n"]
        group = resolve_group(inst, n_val)
        config = row["config"]

        ratio = _to_float(row.get("rel_mean_to_baseline"))
        time_s = _to_float(row.get("mean_time_s"))

        key = (group, config)
        if key not in grouped:
            grouped[key] = {"ratios": [], "times": []}
        if ratio is not None:
            grouped[key]["ratios"].append(ratio)
        if time_s is not None:
            grouped[key]["times"].append(time_s)

    group_rows = []
    for (group, config), agg in grouped.items():
        ratios = agg["ratios"]
        times = agg["times"]
        group_rows.append({
            "group": group,
            "config": config,
            "count_instances": len(ratios),
            "mean_rel_to_baseline": f"{mean(ratios):.6f}" if ratios else "",
            "mean_time_s": f"{mean(times):.6f}" if times else "",
        })

    group_rows.sort(key=lambda r: (r["group"], r["config"]))

    group_fields = [
        "group",
        "config",
        "count_instances",
        "mean_rel_to_baseline",
        "mean_time_s",
    ]
    _write_csv(args.out_group, group_fields, group_rows)
    print(f"Wrote group summary: {args.out_group}")

    group_plot = _plot_group_summary(
        group_rows,
        out_path=os.path.join("results", "plots", "summary_by_group.png"),
    )

    group_md = os.path.splitext(args.out_group)[0] + ".md"
    _write_markdown_report(
        group_md,
        title="Group summary",
        description_lines=[
            "Columns:",
            "- group: Group identifier (size, prefix, or explicit mapping).",
            "- config: Configuration label.",
            "- count_instances: Number of instances aggregated in this group.",
            "- mean_rel_to_baseline: Mean of rel_mean_to_baseline within the group.",
            "- mean_time_s: Mean runtime per run within the group (seconds).",
        ],
        table_fields=group_fields,
        table_rows=group_rows,
        plot_rel_path="plots/summary_by_group.png" if group_plot else None,
    )
    print(f"Wrote group report: {group_md}")


if __name__ == "__main__":
    main()
