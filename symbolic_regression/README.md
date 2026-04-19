# Symbolic Regression - Genetic Programming

Project II for the Bio-Inspired Algorithms course (ESIEE Paris, 2026).

## Project Structure

```
symbolic_regression/
├── main.py                  <- CLI: run GP on one instance or in batch
├── run_experiments.py       <- Reproduce all results from the report
├── src/
│   ├── tree.py              <- Node class + random tree generators
│   ├── fitness.py           <- MSE + complexity-penalised fitness
│   ├── operators.py         <- Mutation & crossover operators
│   ├── selection.py         <- Tournament, elitist, over-selection
│   ├── gp.py                <- Main GP evolutionary loop
│   └── data.py              <- Instance file loader
├── instances/               <- sr_*.txt data files
├── results/                 <- Output directory (created on run)
│   ├── summary.txt          <- Human-readable results table
│   ├── results.csv          <- Full data (one row per instance)
│   └── raw/                 <- Per-instance trial detail files
├── tests/
│   └── test_all.py          <- Unit tests (21 tests)
└── README.md
```

## Run experiments

Run on a single instance:

```bash
python main.py instances/sr_poly_01.txt
```

Reproduce all experimental results:

```bash
python run_experiments.py
```

Parameters used in the report:

```bash
python run_experiments.py \
    --trials 5 --pop-size 200 --generations 300
```


This runs every `sr_*.txt` file found in `instances/` with 5 independent trials each
and writes three output files to `results/`:

| File | Contents |
|---|---|
| `results/summary.txt` | Formatted table grouped by instance type |
| `results/results.csv` | Full data (best/mean/std RMSE, expression, size, time) |
| `results/raw/*.txt`   | Per-instance trial-by-trial detail |



## Command-Line Options



```
python run_experiments.py [options]

  --instances-dir DIR     Directory containing sr_*.txt files      (default: instances/)
  --output-dir DIR        Where to write results                   (default: results/)
  --trials N              Independent trials per instance          (default: 5)
  --pop-size N            (same options as main.py above ...)
  --generations N
  ...
```


## Algorithm

### Representation

Individuals are binary expression trees. The genotype is the tree structure; the phenotype
is the mathematical expression it encodes, output as a Python-evaluable string.

- Internal nodes: binary operators `+`, `-`, `*`, `/`
- Terminal nodes:
  - `x` - the variable
  - integer constants, evolved via Gaussian mutation
  - `x**a` for `a` in {2, 3, 4, 5} - power terms

### Fitness

```
fitness(f) = MSE(f, data) + complexity_weight * size(tree)
```

Lower is better. The complexity penalty discourages bloat without preventing the
algorithm from building larger expressions when they genuinely improve the fit.

### Variation operators

| Operator | Default probability | Description |
|---|---|---|
| Subtree crossover | 70% | Exchange a random subtree between two parents |
| Subtree mutation  | 10% | Replace a random node with a new random subtree |
| Point mutation    | 10% | Nudge a constant, change an operator, or alter an exponent |
| Hoist mutation    |  5% | Replace a node with one of its own subtrees (shrinks tree) |
| Reproduction      |  5% | Copy a parent unchanged |

Crossover is applied first (with `subtree_crossover`), and constant folding is run
on each resulting child to simplify constant sub-expressions before evaluation.

### Selection

- Parent selection: k-tournament (default k = 7)
- Survivor selection: (mu + lambda) elitist — combine parents and children, keep
  the best `pop_size` overall, with at least `elite_count` from the previous generation.

### Bloat control

Four complementary mechanisms limit tree growth:

1. `complexity_weight` penalty in the fitness function
2. `max_depth` hard cap enforced after every crossover
3. `hoist_mutation` operator, which actively reduces tree size
4. `constant_folding` simplifies sub-trees of constants into a single node

### Restarts and trials

`main.py --restarts N` runs N independent evolutions and returns the best result.
`run_experiments.py --trials N` runs the whole GP (including its restarts) N times
independently to estimate mean and standard deviation across runs.