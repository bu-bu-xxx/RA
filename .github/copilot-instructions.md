# RABBI Framework - AI Agent Instructions

## Project Overview
RABBI is a non-invasive framework for dynamic pricing research implementing LP-based policies for multi-product resource allocation. The codebase simulates customer choice under resource constraints with multiple solvers (RABBI, OFFline, NPlusOneLP, TopKLP, Robust).

## Architecture

### Core Components (all under `RABBI/framework/`)
- **`env.py`**: Parameter loading from YAML configs, demand models (MNL/Linear), price grid generation
- **`customer.py`**: Customer choice simulator, generates Y matrices (realized choices) and Q matrices (offline probabilities)
- **`solver.py`**: All policy implementations inherit from `LPBasedPolicy` base class
  - `RABBI`: Resolves LP per period with current-time demand estimates
  - `OFFline`: Uses precomputed Q matrices (requires `compute_offline_Q()`)
  - `NPlusOneLP`: Neighborhood exploration with continuous relaxation
  - `TopKLP`: Top-k menu LP approach
  - `Robust`: Column generation via dual resolution (ellipsoid + separation oracle)
- **`di.py`**: Dependency injection container (`Container`) and policy registry (`PolicyRegistry`)
- **`runner.py`**: Orchestrates single/multi-k runs with caching (shelve) and parallel execution
- **`results.py`**: Standardized result objects (`RunResult`, `MultiKResult`) and metric computations
- **`viz.py`**: Generates comparison plots (reward vs k, ratios, regret, LP-x benchmark)
- **`cli.py`**: CLI entry point (`python -m cli {single,multi,cache,clear-cache}`)

### Data Flow
1. **Config** (YAML) → `ParamsLoader` → `Parameters` object with A (resource matrix), f (price matrix), B (budgets), T (horizon), demand model
2. **Simulation** → `Container.prepare_qy()` generates/loads Y (realized choices) and Q (offline probs) using memory-mapped `.npy` files
3. **Solver execution** → Policy calls `env.step(j, alpha)` each period, updates budgets/histories
4. **Multi-k scaling** → `scaling_list` in YAML controls k values; runner scales T and B by k
5. **Caching** → Shelve stores solver results keyed by `(solver_name, k_val, config_signature)`; Y/Q arrays persist separately

## Critical Workflows

### Running Experiments
```bash
cd RABBI  # ALWAYS work from RABBI/ directory
python3 -m cli single --param params/params5.yml --solver RABBI --seed 42
python3 -m cli multi --param params/params5.yml --qy-prefix data/QY/qy_params5 \
  --solvers OFFline Robust --plots multi_k_results --plot-dir data/pics
python3 -m cli cache --param params/params5.yml --qy-prefix data/QY/qy_params5 \
  --solvers RABBI OFFline NPlusOneLP TopKLP Robust --shelve-dir data/shelve
./run_cache.sh 7  # Convenience wrapper for params7.yml with logging
```

### Testing
```bash
cd RABBI
python3 -m pytest tests/  # Run all smoke tests
python3 -m pytest tests/test_runner_smoke.py::test_run_single_smoke  # Single test
```

### Clearing Cache (important when changing `scaling_list`)
```bash
cd RABBI
python3 -m cli clear-cache --shelve-dir data/shelve  # Clears all
python3 -m cli clear-cache --solvers Robust --preview-clear  # Preview only
rm -rf data/QY/* data/shelve/* data/pics/*  # Nuclear option
```

## Key Conventions

### Adding New Solvers
1. Inherit from `LPBasedPolicy` in `framework/solver.py`
2. Implement `__init__(self, env, debug=False)` and `run(self)` methods
3. Register in `PolicyRegistry._registry` in `framework/di.py`
4. Add to CLI choices automatically via `PolicyRegistry.available_names()`

### YAML Configuration
- **Required fields**: `product_number`, `resource_number`, `resource_matrix`, `price_set_matrix`, `horizon`, `budget`, `scaling_list`, `demand_model`
- **Demand models**: `MNL` (requires `d`, `mu`, `u0`, `gamma`) or `LINEAR` (requires `psi`, `theta`)
- **Robust solver**: Add `robust: {eta: 1e-8, sep_eps: 1e-6}` section for dual feasibility/oracle tolerances
- Use `null` in `price_set_matrix` to allow skipping products (handled by `allow_skip` masks)

### Memory Management
- Y/Q arrays can be huge (T × m × n); always use `--qy-prefix` for multi-k runs
- `Container.prepare_qy()` uses file locking + `mmap_mode='r'` to avoid loading full arrays
- Cached params strip `Y`, `Q`, `x_history` to keep shelve files small
- `--debug` flag populates `x_history` for deep traces (only use for single runs)

### File Naming Patterns
- Params: `params/params<N>.yml` where N is numeric suffix
- Y/Q files: `{prefix}_k{N}_Y.npy` and `{prefix}_k{N}_Q.npy` (or no suffix for k=1.0)
- Plots: `{prefix}multi_k_results.png`, `{prefix}multi_k_ratio_results.png`, `{prefix}lp_x_benchmark_ratio_vs_k.png`
- Shelve: `{shelve_prefix}<solver>_<param_key>.db` (varies by dbm backend)

### Debugging Tips
- Use `--debug` flag for single runs to see per-period alpha selections and LP solutions
- Check `params.no_sell_cnt` to see how often inventory constraints forced no-sell prices
- Robust solver tracks `A_prime_size_history` showing column counts per period
- Plot `lp_x_benchmark_ratio` to see how LP solution quality correlates with performance

## Common Pitfalls

1. **Running from wrong directory**: CLI assumes `python -m cli` from `RABBI/` folder
2. **Mismatched scaling_list**: Clearing only shelve but not Y/Q causes dimension mismatches in plots → clear all artifacts
3. **OFFline requires Q**: Must call `compute_offline_Q()` or use `--qy-prefix` with cached Q
4. **Large m values**: Full enumeration of price combinations scales as `∏|P_i|`; use TopKLP/Robust for large action spaces
5. **Config signature changes**: Modifying YAML (except `scaling_list`) invalidates cache; use `clear-cache` or delete shelve files

## Project-Specific Patterns

- **Non-invasive design**: Framework wraps existing logic without renaming core functions/classes
- **Policy registry pattern**: Solvers self-register via string keys; CLI dynamically resolves classes
- **Standardized results**: All solvers return params objects with `reward_history`, `b_history`, `alpha_history`
- **Parallel execution**: `run_multi_k_with_cache()` uses `concurrent.futures` with configurable `max_concurrency`
- **Lazy Q computation**: Y matrix generated first; Q computed only when needed (OFFline solver or benchmarks)
- **Config signature hashing**: Cache keys include JSON-sorted config to detect parameter changes

## Mathematics Note
See `RABBI/Robust-RABBI.md` for full algorithm specification. Key formulas:
- **Revenue coefficient**: $r_\alpha = \sum_i f_{i,\alpha} p_{i,\alpha}$
- **Resource coefficient**: $c_{k,\alpha} = \sum_i A_{i,k} p_{i,\alpha}$
- **Reduced cost** (Robust): $\bar{c}_\alpha = r_\alpha - \sum_k c_{k,\alpha}\lambda_k - \mu$
- **MNL demand**: $p_i(\alpha) = \frac{d_i \exp(-\gamma f_{i,\alpha} / \mu)}{\sum_j d_j \exp(-\gamma f_{j,\alpha} / \mu) + \exp(u_0 / \mu)}$
