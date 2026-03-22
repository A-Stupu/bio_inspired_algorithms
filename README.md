# Bio-Inspired Algorithms — TSP Local Search

This repository contains local search experiments for the Traveling Salesperson Problem (TSP):
greedy local search (GLS) and simulated annealing (SA), with swap, 2-opt, and Or-opt operators.

Project code lives under [tsp_ls/src](tsp_ls/src) and experiments are driven by the CLI in
[tsp_ls/src/run.py](tsp_ls/src/run.py).

## Dataset setup

Place your TSPLIB .tsp instances under [tsp_ls/DB](tsp_ls/DB). See
[tsp_ls/DB/README.md](tsp_ls/DB/README.md) for suggested sources and folder layout.

## Dependencies (Windows/Linux)

Requirements file: [tsp_ls/requirements.txt](tsp_ls/requirements.txt)

Linux:

```bash
python3 -m pip install -r tsp_ls/requirements.txt
```

Windows (PowerShell):

```powershell
py -m pip install -r tsp_ls/requirements.txt
```

## Run experiments

Run all configs on a folder:

```bash
python -m src.run --tsp_dir DB/bioalg-proj01-tsplib
```

Run a single instance:

```bash
python -m src.run --tsp_file berlin52
```

Filter configs by CLI options:

```bash
python -m src.run --algo gls_opt --operator 2-opt
python -m src.run --algo sa_naive --schedule exp
python -m src.run --include 2-opt exp --exclude or-opt
python -m src.run --configs gls_opt__2-opt sa_opt__2-opt__exp_medium
```

## Plots and tests

Plot helpers are in [tsp_ls/tests](tsp_ls/tests), for example:

```bash
python -m tests.generate_berlin52_plots
python -m tests.plot_fitness_evolution --tsp_file berlin52
```

Basic unit tests:

```bash
python -m tests.test_tsp
```

## Aggregate metrics

Per-instance summary (mandatory for reporting):

```bash
python -m tests.aggregate_results --in results/results.csv --baseline gls_opt__2-opt
```

Group summary (optional):

```bash
python -m tests.aggregate_results --group_by n
python -m tests.aggregate_results --group_by prefix
python -m tests.aggregate_results --group_map results/groups.csv
```
