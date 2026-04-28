# Symbolic Regression - Genetic Programming

Project II for the Bio-Inspired Algorithms course (ESIEE Paris, 2026).

## Project Structure


```
symbolic_regression/
├── main.py                  <- CLI: run GP on one instance or in batch
├── run_experiments.py       <- Reproduce all results from the report
├── run_challenges.py        <- Dedicated runner for challenge datasets
├── src/
│   ├── tree.py              <- Node class + random tree generators
│   ├── fitness.py           <- MSE + complexity-penalised fitness
│   ├── operators.py         <- Mutation & crossover operators
│   ├── selection.py         <- Tournament, elitist, over-selection
│   ├── gp.py                <- Main GP evolutionary loop
│   └── data.py              <- Instance file loader
├── instances/               <- sr_*.txt data files (poly, ratio, approx, periodic, challenge)
├── results/                 <- Output directory (created on run)
│   ├── summary.txt              <- Human-readable results table (run_experiments)
│   ├── results.csv               <- Full data (one row per instance)
│   ├── challenges_summary.txt    <- Human-readable results table (run_challenges)
│   ├── challenges.csv            <- Full data for challenge runs
│   ├── plots/                    <- PNG plots per instance
│   └── raw/                      <- Per-instance trial detail files
├── tests/
│   └── test_all.py           <- Unit tests
└── README.md
```

## Run experiments

Run on a single instance:

```bash
python main.py instances/sr_poly_01.txt
```

Reproduce all first-dataset results (5 trials x 24 instances):

```bash
python run_experiments.py --trials 5 --pop-size 200 --generations 300
```

Reproduce all challenge results (10 trials x 8 instances):

```bash
python run_challenges.py
    --trials 10
```

Both scripts write their output to `results/`


| File                        | Contents                                                      |
|-----------------------------|---------------------------------------------------------------|
| `results/summary.txt`       | Formatted table grouped by instance type   |
| `results/results.csv`       | Full data (best/mean/std RMSE, expression, size, time)        |
| `results/challenges_summary.txt` | Same for challenge instances         |
| `results/challenges.csv`    | Full data for challenge runs                                  |
| `results/raw/*.txt`         | Per-instance trial-by-trial detail                            |
| `results/plots/*.png`       | Per-instance predicted vs actual fit plots                                      |



## Command-Line Options

`run_experiments.py`

```
python run_experiments.py [options]

  --instances-dir DIR     Directory containing sr_*.txt files      (default: instances/)
  --output-dir DIR        Where to write results                   (default: results/)
  --trials N              Independent trials per instance          (default: 5)
  --pop-size N            (same options as main.py above ...)
  --generations N
  ...
```

`run_challenges.py`

Dedicated runner for challenge instances (challenge_a, challenge_b, challenge_c). Key differences from `run_experiments.py`: larger population and more generations by default, float-aware point mutation, structural seeding, and post-evolutionary constant optimisation.

```
python run_challenges.py [options]
 
  --instances-dir DIR     Directory containing sr_challenge_*.txt  (default: instances/)
  --output-dir DIR        Where to write results                   (default: results/)
  --trials N              Independent trials per instance          (default: 10)
  --pop-size N            Population size µ                        (default: 300)
  --generations N         Maximum generations per restart          (default: 500)
  --restarts N            Independent restarts inside each trial   (default: 1)
  --max-depth N           Hard depth cap during search             (default: 7)
  --max-depth-init N      Max depth at initialisation              (default: 5)
  --tournament-k k        Tournament size for parent selection     (default: 7)
  --elite-count N         Individuals always preserved each gen.   (default: 5)
  --patience N            Early-stop if no improvement for N gen.  (default: 120)
  --p-crossover F         Subtree crossover probability            (default: 0.70)
  --p-sub-mut F           Subtree mutation probability             (default: 0.10)
  --p-point-mut F         Point mutation probability (float-aware) (default: 0.12)
  --p-hoist-mut F         Hoist mutation probability               (default: 0.05)
  --complexity-weight F   Penalty weight λ on tree size            (default: 0.01)
  --const-opt-steps N     Hill-climbing steps after GP             (default: 120)
  --level {a,b,c}         Run only instances of this level
```

The `--level` flag is useful for quick targeted runs:
 
```bash
python run_challenges.py --level c
```