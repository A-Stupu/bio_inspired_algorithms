# Symbolic Regression — Genetic Programming

Project II for the Bio-Inspired Algorithms course (ESIEE Paris, 2026).

## Requirements

Python ≥ 3.10 (uses `str | None` union syntax).  No external dependencies.

```
pip install pytest   # only needed to run the test suite
```

## Project Structure

```
symbolic_regression/
├── main.py                  ← CLI entry point
├── src/
│   ├── tree.py              ← Node class + random tree generators
│   ├── fitness.py           ← MSE + complexity-penalised fitness
│   ├── operators.py         ← Mutation & crossover operators
│   ├── selection.py         ← Tournament, elitist, over-selection
│   ├── gp.py                ← Main GP evolutionary loop
│   └── data.py              ← Instance file loader
├── instances/               ← sr_*.txt data files
├── tests/
│   └── test_all.py          ← Unit tests
└── README.md
```

## Running

### Single instance

```bash
python main.py instances/sr_poly_04.txt
```

### With custom parameters

```bash
python main.py instances/sr_poly_04.txt \
    --pop-size 300 --generations 500 --restarts 5 \
    --complexity-weight 0.005 --verbose 2
```

### Batch mode (all instances in a directory)

```bash
python main.py --batch instances/ --output-dir results/
```

### Full parameter reference

```
usage: symbolic_regression [-h] [--batch] [--pop-size N] [--generations N]
                            [--restarts N] [--max-depth N] [--max-depth-init N]
                            [--tournament-k N] [--elite-count N] [--patience N]
                            [--p-crossover P] [--p-subtree-mutation P]
                            [--p-point-mutation P] [--p-hoist-mutation P]
                            [--complexity-weight W] [--verbose {0,1,2}]
                            [--output-dir DIR]
                            [input]
```

### Run tests

```bash
python -m pytest tests/ -v
```

## Algorithm

### Representation

Individuals are binary expression trees (genotype = phenotype here).

- **Internal nodes**: binary operators `+`, `-`, `*`, `/`
- **Terminal nodes**:
  - `x` — the variable
  - constants (integer values by default, evolved via Gaussian mutation)
  - `x**a` for `a ∈ {2,3,4,5}` — power terms

### Fitness

```
fitness = MSE(f, data) + complexity_weight × size(tree)
```

Lower is better. The complexity penalty controls bloat.

### Variation operators

| Operator | Probability (default) | Description |
|---|---|---|
| Subtree crossover | 70 % | Exchange random subtrees between two parents |
| Subtree mutation  | 10 % | Replace a random node with a new random subtree |
| Point mutation    | 10 % | Modify a terminal (nudge constant, change operator) |
| Hoist mutation    |  5 % | Replace a node with one of its subtrees (shrinks tree) |
| Reproduction      |  5 % | Copy parent unchanged |

### Selection

- **Parent selection**: k-tournament (k=7 by default)
- **Survivor selection**: (µ+λ) elitist — keep the best `pop_size` individuals from parents ∪ children, with a guaranteed `elite_count` from the previous generation.

### Bloat control

- `hoist_mutation` reduces tree size directly.
- `constant_folding` simplifies constant sub-expressions.
- `complexity_weight` penalty in fitness.
- `max_depth` hard limit on tree depth after crossover.

### Restarts

The algorithm is run `--restarts` times independently; the best result overall is returned.
