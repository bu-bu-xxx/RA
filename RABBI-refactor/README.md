# RABBI Refactor (Non-invasive framework)

This folder provides a minimal, non-invasive framework around the existing code to improve:
- Dependency injection and composability
- Standardized data passing and results
- Testability and runnable CLI

It does not rename or change any function/class/variable in the original logic. Core files from the old project have been copied here so the folder is self-contained.

## Layout

- `framework/`
  - `di.py`: Policy registry and container to build sims/solvers.
  - `runner.py`: Orchestration for single/multi-k runs (with cache).
  - `results.py`: Standard result objects and metrics wrappers.
  - `viz.py`: Visualization adapter delegating to existing `plot.py`.
- `cli.py`: Simple CLI for single/multi/cache runs with optional plots.
- `examples/`: Quick examples.
- Copied originals: `solver.py`, `customer.py`, `env.py`, `read_params.py`, `main.py`, `plot.py`, `plot_offline_nplus1.py`, `params*.yml`.

## Quick Start

Create folders for data and pics (if not present):

```bash
mkdir -p RABBI-refactor/data/Y RABBI-refactor/data/shelve RABBI-refactor/data/pics
```

### Single run

```bash
python3 -m RABBI-refactor.cli single \
  --param RABBI-refactor/tests/params_min.yml \
  --solver RABBI \
  --seed 123
```

### Multi-k run

```bash
python3 -m RABBI-refactor.cli multi \
  --param RABBI-refactor/params5.yml \
  --y-prefix RABBI-refactor/data/Y/Y_matrix_params5 \
  --solvers OFFline NPlusOneLP \
  --plots multi_k_results multi_k_ratio multi_k_regret lp_x_benchmark_ratio \
  --save-dir RABBI-refactor/data/pics
```

### Multi-k with cache

```bash
python3 -m RABBI-refactor.cli cache \
  --param RABBI-refactor/params5.yml \
  --y-prefix RABBI-refactor/data/Y/Y_matrix_params5 \
  --solvers OFFline NPlusOneLP \
  --max-concurrency 4 \
  --shelve-dir RABBI-refactor/data/shelve \
  --plots multi_k_results multi_k_ratio \
  --save-dir RABBI-refactor/data/pics
```

## Plot keys
- `multi_k_results`: total reward vs k of multiple policies
- `multi_k_ratio`: policy/Offline reward ratio vs k
- `multi_k_regret`: regret vs Offline baseline
- `lp_x_benchmark_ratio`: LP-x benchmark ratio vs k

## Programmatic usage (example)

See `examples/run_multi_with_plots.py`:

```bash
python3 RABBI-refactor/examples/run_multi_with_plots.py
```

## Notes
- The framework adds the local refactor folder and (if needed) the old neighbor folder to `sys.path` to import original modules.
- The original files were copied here so you can delete the old project once you validate this folder runs end-to-end.

## Smoke Test (All Solvers)

Run a comprehensive smoke test covering all solvers (RABBI, OFFline, NPlusOneLP, TopKLP), including plot generation and cache:

```bash
# Ensure required directories exist
mkdir -p RABBI-refactor/data/Y RABBI-refactor/data/pics RABBI-refactor/data/shelve

# Multi-k with all solvers and plots
python3 -m RABBI-refactor.cli multi \
  --param RABBI-refactor/tests/params_min.yml \
  --y-prefix RABBI-refactor/data/Y/Y_matrix_params_min \
  --solvers RABBI OFFline NPlusOneLP TopKLP \
  --plots multi_k_results multi_k_ratio multi_k_regret lp_x_benchmark_ratio \
  --save-dir RABBI-refactor/data/pics

# Cache mode to create shelve artifacts
python3 -m RABBI-refactor.cli cache \
  --param RABBI-refactor/tests/params_min.yml \
  --y-prefix RABBI-refactor/data/Y/Y_matrix_params_min \
  --solvers RABBI OFFline NPlusOneLP TopKLP \
  --shelve-dir RABBI-refactor/data/shelve \
  --plots multi_k_results \
  --save-dir RABBI-refactor/data/pics

# Expected outputs
ls -1 RABBI-refactor/data/pics
ls -1 RABBI-refactor/data/shelve
```

This mirrors what the tests do in `tests/test_cli_and_multi_smoke.py`.
