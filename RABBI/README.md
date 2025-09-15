# RABBI (Non-invasive framework)

This folder provides a minimal, non-invasive framework around the existing code to improve:
- Dependency injection and composability
- Standardized data passing and results
- Testability and runnable CLI

It does not rename or change any function/class/variable in the original logic. Core logic is implemented under `framework/` so this folder is self-contained.

## Layout

- `framework/`
  - `cli.py`: CLI for single/multi/cache runs with optional plots.
  - `di.py`: Policy registry and container to build sims/solvers.
  - `runner.py`: Orchestration for single/multi-k runs (with cache).
  - `results.py`: Standard result objects and metrics wrappers.
  - `viz.py`: Visualization for multi-policy results.
  - `solver.py`: Policy implementations (RABBI, OFFline, NPlusOneLP, TopKLP).
  - `customer.py`: Customer simulator built on the environment.
  - `env.py`: Parameter loading and environment dynamics.
- `examples/`: Quick examples.
- `params/`: YAML configs (moved from top-level into this folder).

Notes:
- Plotting is handled directly in `framework/viz.py`.
- Legacy top-level modules (`customer.py`, `env.py`, `solver.py`, `cli.py`) have been removed in favor of the `framework/` package.

## Quick Start

Create folders for data and pics (if not present):

```bash
# Run from inside the RABBI folder
cd RABBI
mkdir -p data/Y data/shelve data/pics
```

### Single run

```bash
# Run from inside the RABBI folder
cd RABBI
python3 -m cli single \
  --param tests/params_min.yml \
  --solver RABBI \
  --seed 123
```

### Multi-k run

```bash
# Run from inside the RABBI folder
cd RABBI
python3 -m cli multi \
  --param params/params5.yml \
  --y-prefix data/Y/Y_matrix_params5 \
  --solvers OFFline NPlusOneLP \
  --plots multi_k_results multi_k_ratio multi_k_regret lp_x_benchmark_ratio \
  --save-dir data/pics \
  --save-prefix exp1_
```

### Multi-k with cache

```bash
# Run from inside the RABBI folder
cd RABBI
python3 -m cli cache \
  --param params/params5.yml \
  --y-prefix data/Y/Y_matrix_params5 \
  --solvers OFFline NPlusOneLP \
  --max-concurrency 4 \
  --shelve-dir data/shelve \
  --plots multi_k_results multi_k_ratio \
  --save-dir data/pics \
  --save-prefix exp1_
```

### Available solvers and CLI help

The `--solvers` argument validates and shows available choices in `-h`:

```bash
# Run from inside the RABBI folder
cd RABBI
python3 -m cli multi -h
```

Available solver names: RABBI, OFFline, NPlusOneLP, TopKLP

### Plot filename prefix

- Use `--save-prefix <prefix>` to prepend a prefix to generated image filenames.
- Example outputs: `exp1_multi_k_results.png`, `exp1_multi_k_ratio_results.png`,
  `exp1_multi_k_regret_results.png`, `exp1_lp_x_benchmark_ratio_vs_k.png`.

### Logging / Progress

Runs print progress to stdout:

- Start banner with parameters: `param`, `y_prefix`, `solvers`, `k_values`, `max_concurrency`, `seed`, task count
- Per-task schedule lines: `- task#<idx> solver=<name> k=<value>`
- Per-task completion lines: `[done] task#<idx> solver=<name> k=<value> total_reward=<sum> steps=<T>`
- `run_single` prints a start line and a completion summary.

This makes long multi-k runs observable in real time without extra flags.

## Plot keys
- `multi_k_results`: total reward vs k of multiple policies
- `multi_k_ratio`: policy/Offline reward ratio vs k
- `multi_k_regret`: regret vs Offline baseline
- `lp_x_benchmark_ratio`: LP-x benchmark ratio vs k

## Programmatic usage (example)

See `examples/run_multi_with_plots.py`:

```bash
python3 examples/run_multi_with_plots.py
```

## Notes
- The CLI is available via `python -m cli` from inside this folder. Programmatic usage can `import framework.runner` etc.

### Cache compatibility
- Shelve files created by older code may embed pickles importing modules that no longer exist. The runner ignores unreadable cache entries and recomputes missing results automatically, then writes back compatible entries.
- To force a clean state, clear cache files:

```bash
# Run from inside the RABBI folder
cd RABBI
python3 -m cli clear-cache --shelve-dir RABBI/data/shelve
# Or specific solvers
python3 -m cli clear-cache --solvers OFFline NPlusOneLP --shelve-dir data/shelve
```

### Clear-cache preview (no deletion)
- Use the preview flag to see what would be removed without deleting files.
- Flags: `--preview-clear` (primary), aliases: `--dry-run`, `--preview`.

Examples:

```bash
# Run from inside the RABBI folder
cd RABBI
# Preview all cache files in default directory
python3 -m cli clear-cache --preview-clear

# Preview for specific solvers only
python3 -m cli clear-cache --solvers OFFline NPlusOneLP --preview-clear

# Preview with a custom cache directory
python3 -m cli clear-cache --shelve-dir data/shelve --preview-clear
```

Output:
- Preview prints `Would remove:` followed by the matching shelve files (supports backends creating `.db/.dat/.dir/.bak`).
- Without preview, it prints `Removed:` and deletes those files.

## Smoke Test (All Solvers)

Run a comprehensive smoke test covering all solvers (RABBI, OFFline, NPlusOneLP, TopKLP), including plot generation and cache:

```bash
# Ensure required directories exist
mkdir -p data/Y data/pics data/shelve

# Multi-k with all solvers and plots
python3 -m cli multi \
  --param tests/params_min.yml \
  --y-prefix data/Y/Y_matrix_params_min \
  --solvers RABBI OFFline NPlusOneLP TopKLP \
  --plots multi_k_results multi_k_ratio multi_k_regret lp_x_benchmark_ratio \
  --save-dir data/pics

# Cache mode to create shelve artifacts
python3 -m cli cache \
  --param tests/params_min.yml \
  --y-prefix data/Y/Y_matrix_params_min \
  --solvers RABBI OFFline NPlusOneLP TopKLP \
  --shelve-dir data/shelve \
  --plots multi_k_results \
  --save-dir data/pics

# Expected outputs
ls -1 data/pics
ls -1 data/shelve
```

This mirrors what the tests do in `tests/test_cli_and_multi_smoke.py`.
