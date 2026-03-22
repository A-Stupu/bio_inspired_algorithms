# Per-instance summary columns

- instance: Instance name (filename without extension).
- n: Number of nodes in the instance.
- config: Configuration label used in the run.
- best_cost: Best tour cost observed across runs.
- mean_cost: Average tour cost across runs.
- std_cost: Standard deviation of tour cost across runs.
- mean_time_s: Average runtime per run (seconds).
- std_time_s: Standard deviation of runtime (seconds).
- mean_init_cost: Average initial tour cost (only for configs that log it).
- improvement_pct: Percent improvement from mean_init_cost to mean_cost.
- rel_mean_to_baseline: mean_cost / baseline_mean_cost (values < 1 are better).
- rel_improvement_pct: Percent improvement vs baseline (positive is better).
